"""
Module 6 — EEM: Excitation-Emission Matrix Visualization
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io_utils import load_spectrum_file, to_excel_download
from utils.plot_utils import spectral_colorscale

st.set_page_config(page_title="EEM | PL Analyzer", layout="wide", page_icon="🗺")
st.title("🗺 EEM — Excitation-Emission Matrix")
st.markdown("""
여러 여기 파장에서 측정한 PL 스펙트럼을 2D contour/heatmap으로 시각화합니다.
각 파일에 해당하는 여기 파장을 입력하세요.
""")

# ── Upload ──────────────────────────────────────────────────────────────────
eem_files = st.file_uploader("PL 파일들 (여기 파장별로 각 1개 파일)",
                              type=['csv','txt','xlsx','xls'],
                              accept_multiple_files=True)

if not eem_files:
    st.info("여러 여기 파장에서 측정한 PL 스펙트럼 파일들을 업로드하세요.")
    st.stop()

# ── Excitation wavelength input ─────────────────────────────────────────────
st.subheader("⚙️ 여기 파장 및 설정")
exc_df_init = pd.DataFrame({
    '파일': [f.name for f in eem_files],
    '여기 파장 (nm)': [float(300 + i*20) for i in range(len(eem_files))],
    '포함 여부': [True] * len(eem_files)
})
exc_df = st.data_editor(exc_df_init, use_container_width=True, key='eem_table')

col1, col2, col3 = st.columns(3)
normalize_eem = col1.checkbox("정규화 (각 스펙트럼 최대값 = 1)", value=False)
mask_rayleigh = col2.checkbox("Rayleigh 산란선 마스킹", value=True)
mask_raman = col3.checkbox("Raman 산란선 마스킹", value=False)

col4, col5 = st.columns(2)
rayleigh_bw = col4.slider("Rayleigh 마스킹 폭 (±nm)", 0, 30, 10) if mask_rayleigh else 10
plot_type = col5.radio("플롯 타입", ["Contour (filled)", "Heatmap", "3D Surface"], horizontal=True)

if st.button("🔄 EEM 생성", type="primary"):
    # Load and organize data
    spectra_by_exc = {}
    all_emission_wl = set()

    for _, row in exc_df.iterrows():
        if not row['포함 여부']: continue
        f = next((x for x in eem_files if x.name == row['파일']), None)
        if f is None: continue
        f.seek(0)
        wl, inten = load_spectrum_file(f)
        if wl is None: continue
        exc = float(row['여기 파장 (nm)'])

        if mask_rayleigh:
            mask = np.abs(wl - exc) > rayleigh_bw
            wl = wl[mask]; inten = inten[mask]
        if mask_raman:
            # Water Raman at ~3400 cm-1
            raman_wl = 1e7 / (1e7/exc - 3400)
            mask_r = np.abs(wl - raman_wl) > rayleigh_bw
            wl = wl[mask_r]; inten = inten[mask_r]

        if normalize_eem and inten.max() > 0:
            inten = inten / inten.max()

        spectra_by_exc[exc] = (wl, inten)
        all_emission_wl.update(wl.tolist())

    if not spectra_by_exc:
        st.error("유효한 데이터 없음"); st.stop()

    # Common emission grid
    em_wl = np.linspace(min(all_emission_wl), max(all_emission_wl), 500)
    exc_list = sorted(spectra_by_exc.keys())
    Z = np.zeros((len(exc_list), len(em_wl)))

    for i, exc in enumerate(exc_list):
        wl_s, inten_s = spectra_by_exc[exc]
        Z[i] = np.interp(em_wl, wl_s, inten_s, left=0, right=0)

    # ── Plotting ──────────────────────────────────────────────────────────────
    if plot_type == "Contour (filled)":
        fig_eem = go.Figure(go.Contour(
            x=em_wl, y=exc_list, z=Z,
            colorscale=spectral_colorscale(),
            contours=dict(showlabels=True, labelfont=dict(size=10, color='white')),
            colorbar=dict(title='Intensity', title_side='right')
        ))
    elif plot_type == "Heatmap":
        fig_eem = go.Figure(go.Heatmap(
            x=em_wl, y=exc_list, z=Z,
            colorscale=spectral_colorscale(),
            colorbar=dict(title='Intensity')
        ))
    else:  # 3D Surface
        fig_eem = go.Figure(go.Surface(
            x=em_wl, y=exc_list, z=Z,
            colorscale=spectral_colorscale(),
            colorbar=dict(title='Intensity')
        ))
        fig_eem.update_layout(
            scene=dict(
                xaxis_title='Emission (nm)',
                yaxis_title='Excitation (nm)',
                zaxis_title='Intensity',
                bgcolor='rgba(18,18,28,0.95)'
            )
        )

    # Rayleigh line
    if mask_rayleigh and plot_type != "3D Surface":
        fig_eem.add_trace(go.Scatter(
            x=exc_list, y=exc_list, mode='lines',
            name='Rayleigh line (λ_em = λ_exc)',
            line=dict(color='rgba(255,255,255,0.3)', dash='dash', width=1)
        ))

    if plot_type != "3D Surface":
        fig_eem.update_layout(
            xaxis_title='Emission Wavelength (nm)',
            yaxis_title='Excitation Wavelength (nm)',
        )

    fig_eem.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(18,18,28,0.95)',
        plot_bgcolor='rgba(18,18,28,0.95)',
        title='Excitation-Emission Matrix (EEM)',
        height=600,
        font=dict(family='monospace', size=13)
    )
    st.plotly_chart(fig_eem, use_container_width=True)

    # ── Diagonal (same exc/em) plot ───────────────────────────────────────────
    with st.expander("📈 개별 스펙트럼 오버레이", expanded=False):
        from utils.plot_utils import make_figure, style_axes, rainbow_colors
        colors_eem = rainbow_colors(len(exc_list))
        fig_ovl = make_figure(title="PL Spectra at Different Excitation Wavelengths")
        style_axes(fig_ovl, "Emission Wavelength (nm)", "Intensity (a.u.)")
        for i, exc in enumerate(exc_list):
            wl_s, inten_s = spectra_by_exc[exc]
            fig_ovl.add_trace(go.Scatter(x=wl_s, y=inten_s, name=f'λ_exc = {exc:.0f} nm',
                                          line=dict(color=colors_eem[i], width=2)))
        st.plotly_chart(fig_ovl, use_container_width=True)

    # ── Export ────────────────────────────────────────────────────────────────
    eem_df = pd.DataFrame(Z, index=[f'exc_{e:.0f}' for e in exc_list], columns=em_wl.round(2))
    excel_bytes = to_excel_download({'EEM Matrix': eem_df.reset_index().rename(columns={'index': 'Exc\\Em(nm)'})})
    st.download_button("📥 EEM 데이터 Excel", data=excel_bytes, file_name="eem_matrix.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
