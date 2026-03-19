"""
Module 3 — Spectral Tools: Raman Subtraction & eV Conversion
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io_utils import load_spectrum_file, load_multiple_files, to_excel_download
from utils.fitting_utils import raman_shift_to_wavelength, intensity_jacobian_transform, wavelength_to_eV
from utils.plot_utils import make_figure, style_axes, COLORS, rainbow_colors

st.set_page_config(page_title="Spectral Tools | PL Analyzer", layout="wide", page_icon="🔧")
st.title("🔧 Spectral Tools — Raman Subtraction & eV Conversion")

tool_tab1, tool_tab2 = st.tabs(["🚫 Raman Peak 제거", "⚡ eV 축 변환 (Jacobian)"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Raman Subtraction
# ═══════════════════════════════════════════════════════════════════════════════
with tool_tab1:
    st.markdown("""
    **Raman 산란 피크 제거** — 여기 파장에서의 순수 용매(또는 blank) 스펙트럼을 빼거나,
    Raman peak 위치를 직접 지정해 Gaussian 으로 피팅 후 제거합니다.
    """)

    method = st.radio("제거 방법",
                       ["📂 Blank 스펙트럼 차감", "🎯 Raman peak 위치 지정 후 제거"],
                       horizontal=True)

    col_l, col_r = st.columns(2)
    with col_l:
        pl_files = st.file_uploader("PL 스펙트럼 파일(들)",
                                     type=['csv','txt','xlsx','xls'],
                                     accept_multiple_files=True, key='raman_pl')
        excitation_nm = st.number_input("여기 파장 (nm)", value=365.0, step=0.5, format="%.1f")

    with col_r:
        if method.startswith("📂"):
            blank_file = st.file_uploader("Blank / 용매 스펙트럼", type=['csv','txt','xlsx','xls'], key='blank')
            scale_blank = st.slider("Blank 배율 (α)", 0.0, 2.0, 1.0, 0.05,
                                    help="PL = PL_sample - α × blank")
        else:
            st.markdown("**Raman shift 범위 설정**")
            # Common Raman shifts for solvents
            solvent = st.selectbox("용매 (참고용)", ["직접 입력", "Water (OH stretch ~3400 cm⁻¹)",
                                                     "Ethanol", "Toluene", "Chloroform", "DMSO"])
            preset_shifts = {
                "직접 입력": [],
                "Water (OH stretch ~3400 cm⁻¹)": [3400],
                "Ethanol": [2900, 3400],
                "Toluene": [1003, 1030, 3060],
                "Chloroform": [667, 3019],
                "DMSO": [670, 2911, 2994],
            }
            default_shifts = preset_shifts.get(solvent, [])
            shifts_str = st.text_input("Raman shift 값들 (cm⁻¹, 쉼표 구분)",
                                        value=', '.join(map(str, default_shifts)) if default_shifts else "3400")
            peak_width_nm = st.slider("제거 창 너비 (nm)", 2, 50, 15)

    if not pl_files:
        st.info("PL 스펙트럼을 업로드하세요")
    else:
        spectra = load_multiple_files(pl_files)
        colors = rainbow_colors(len(spectra))
        blank_wl, blank_inten = None, None

        if method.startswith("📂") and 'blank_file' in dir() and blank_file is not None:
            blank_wl, blank_inten = load_spectrum_file(blank_file)

        if st.button("🔄 Raman 제거 실행", type="primary"):
            fig_r = make_figure(title="Raman Subtraction")
            style_axes(fig_r, "Wavelength (nm)", "Intensity (a.u.)")
            results = {}

            for i, s in enumerate(spectra):
                wl = s['wavelength']
                inten = s['intensity'].copy()
                inten_orig = inten.copy()

                if method.startswith("📂"):
                    if blank_wl is not None:
                        blank_interp = np.interp(wl, blank_wl, blank_inten)
                        inten = inten - scale_blank * blank_interp
                        inten = np.clip(inten, 0, None)
                    else:
                        st.warning("Blank 파일을 업로드하세요")
                else:
                    # Gaussian subtraction per Raman peak
                    try:
                        shifts = [float(x.strip()) for x in shifts_str.split(',') if x.strip()]
                    except:
                        shifts = [3400.0]

                    from scipy.optimize import curve_fit
                    from utils.fitting_utils import gaussian

                    for shift in shifts:
                        raman_wl = raman_shift_to_wavelength(excitation_nm, shift)
                        mask = (wl >= raman_wl - peak_width_nm) & (wl <= raman_wl + peak_width_nm)
                        if mask.sum() < 5:
                            continue
                        x_peak = wl[mask]
                        y_peak = inten[mask]
                        try:
                            popt, _ = curve_fit(gaussian, x_peak, y_peak,
                                                p0=[y_peak.max(), raman_wl, peak_width_nm/3],
                                                maxfev=5000)
                            inten[mask] -= gaussian(x_peak, *popt)
                        except Exception:
                            # Fallback: linear interpolation across the peak
                            idx_lo = np.searchsorted(wl, raman_wl - peak_width_nm)
                            idx_hi = np.searchsorted(wl, raman_wl + peak_width_nm)
                            if idx_lo < idx_hi:
                                inten[idx_lo:idx_hi] = np.linspace(inten[idx_lo], inten[idx_hi],
                                                                     idx_hi - idx_lo)
                    inten = np.clip(inten, 0, None)

                fig_r.add_trace(go.Scatter(x=wl, y=inten_orig, name=f'{s["name"]} (raw)',
                                           line=dict(color=colors[i], width=1, dash='dot'), opacity=0.4))
                fig_r.add_trace(go.Scatter(x=wl, y=inten, name=f'{s["name"]} (corrected)',
                                           line=dict(color=colors[i], width=2)))
                results[s['name']] = pd.DataFrame({'Wavelength (nm)': wl, 'Corrected PL': inten, 'Raw PL': inten_orig})

            # Mark Raman positions
            if not method.startswith("📂"):
                try:
                    shifts = [float(x.strip()) for x in shifts_str.split(',') if x.strip()]
                    for shift in shifts:
                        raman_wl = raman_shift_to_wavelength(excitation_nm, shift)
                        fig_r.add_vline(x=raman_wl,
                                        line=dict(color='rgba(255,100,100,0.5)', dash='dash', width=1.5),
                                        annotation_text=f"Raman\n{shift:.0f} cm⁻¹\n{raman_wl:.1f} nm",
                                        annotation_font_color='rgba(255,150,150,0.8)')
                except: pass

            st.plotly_chart(fig_r, use_container_width=True)

            # Raman position table
            if not method.startswith("📂"):
                try:
                    shifts = [float(x.strip()) for x in shifts_str.split(',') if x.strip()]
                    raman_df = pd.DataFrame({
                        'Raman Shift (cm⁻¹)': shifts,
                        'Raman Emission λ (nm)': [raman_shift_to_wavelength(excitation_nm, s) for s in shifts]
                    })
                    st.markdown("**Raman peak 위치 계산 결과:**")
                    st.dataframe(raman_df, use_container_width=True)
                except: pass

            # Download
            if results:
                excel_bytes = to_excel_download({k[:31]: v for k, v in results.items()})
                st.download_button("📥 보정 데이터 Excel", data=excel_bytes,
                                   file_name="raman_subtracted.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: eV Conversion with Jacobian
# ═══════════════════════════════════════════════════════════════════════════════
with tool_tab2:
    st.markdown(r"""
    **파장 → 에너지 축 변환 (Jacobian Transform)**

    파장(nm) 축에서 에너지(eV) 축으로 단순 변환하면 스펙트럼이 왜곡됩니다.
    Jacobian 보정으로 올바른 에너지 분포를 얻습니다:

    $$I(E) = I(\lambda) \cdot \left|\frac{d\lambda}{dE}\right| = I(\lambda) \cdot \frac{\lambda^2}{hc}$$
    """)

    ev_files = st.file_uploader("스펙트럼 파일(들)",
                                 type=['csv','txt','xlsx','xls'],
                                 accept_multiple_files=True, key='ev_files')
    apply_jacobian = st.checkbox("Jacobian transform 적용", value=True,
                                  help="적용 안 하면 단순 축 변환만 수행")
    normalize_ev = st.checkbox("최대값 정규화", value=False, key='ev_norm')
    show_comparison = st.checkbox("변환 전/후 비교 표시", value=True)

    if ev_files:
        spectra_ev = load_multiple_files(ev_files)
        colors_ev = rainbow_colors(len(spectra_ev))

        fig_ev = make_figure(title="eV Axis Spectra" + (" (Jacobian applied)" if apply_jacobian else ""))
        style_axes(fig_ev, "Energy (eV)", "Intensity (a.u.)" + (" [Jacobian corrected]" if apply_jacobian else ""))

        results_ev = {}
        for i, s in enumerate(spectra_ev):
            wl = s['wavelength']
            inten = s['intensity']

            if apply_jacobian:
                energy, inten_ev = intensity_jacobian_transform(wl, inten)
            else:
                energy = wavelength_to_eV(wl)
                inten_ev = inten.copy()
                idx = np.argsort(energy)
                energy = energy[idx]
                inten_ev = inten_ev[idx]

            if normalize_ev and inten_ev.max() > 0:
                inten_ev /= inten_ev.max()

            fig_ev.add_trace(go.Scatter(x=energy, y=inten_ev, name=s['name'],
                                         line=dict(color=colors_ev[i], width=2)))
            results_ev[s['name']] = pd.DataFrame({
                'Energy (eV)': energy,
                'Intensity': inten_ev,
                'Wavelength (nm)': 1239.84 / energy
            })

        st.plotly_chart(fig_ev, use_container_width=True)

        if show_comparison and spectra_ev:
            st.subheader("변환 전/후 비교")
            from plotly.subplots import make_subplots
            fig_cmp = make_subplots(rows=1, cols=2,
                                    subplot_titles=["원래 스펙트럼 (nm)", "변환 후 (eV)"])
            for i, s in enumerate(spectra_ev):
                wl = s['wavelength']
                inten = s['intensity']
                inten_plot = inten / inten.max() if normalize_ev else inten

                energy, inten_ev = intensity_jacobian_transform(wl, inten) if apply_jacobian else (wavelength_to_eV(wl), inten)
                if not apply_jacobian:
                    idx = np.argsort(energy); energy = energy[idx]; inten_ev = inten_ev[idx]
                if normalize_ev and inten_ev.max() > 0: inten_ev /= inten_ev.max()

                fig_cmp.add_trace(go.Scatter(x=wl, y=inten_plot, name=s['name'],
                                              line=dict(color=colors_ev[i], width=2)), row=1, col=1)
                fig_cmp.add_trace(go.Scatter(x=energy, y=inten_ev, name=s['name'],
                                              showlegend=False,
                                              line=dict(color=colors_ev[i], width=2)), row=1, col=2)

            fig_cmp.update_layout(template='plotly_dark', paper_bgcolor='rgba(18,18,28,0.95)',
                                   plot_bgcolor='rgba(18,18,28,0.95)')
            fig_cmp.update_xaxes(title_text="Wavelength (nm)", row=1, col=1)
            fig_cmp.update_xaxes(title_text="Energy (eV)", row=1, col=2)
            st.plotly_chart(fig_cmp, use_container_width=True)

        if results_ev:
            excel_bytes = to_excel_download({k[:31]: v for k, v in results_ev.items()})
            st.download_button("📥 eV 변환 데이터 Excel", data=excel_bytes,
                               file_name="pl_eV_converted.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
