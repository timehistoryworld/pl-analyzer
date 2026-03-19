"""
Module 2 — PLQY: Photoluminescence Quantum Yield (Comparative Method)
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io_utils import load_spectrum_file, to_excel_download
from utils.plot_utils import make_figure, style_axes, COLORS

st.set_page_config(page_title="PLQY | PL Analyzer", layout="wide", page_icon="💡")
st.title("💡 PLQY — Photoluminescence Quantum Yield")
st.markdown("""
**비교법 (Comparative Method)** — Williams et al. (1983)

$$\\Phi_{sample} = \\Phi_{ref} \\times \\frac{I_{sample}}{I_{ref}} \\times \\frac{A_{ref}}{A_{sample}} \\times \\left(\\frac{n_{sample}}{n_{ref}}\\right)^2$$

> Abs < 0.1 (희석 용액) 권장. 같은 용매 사용 시 굴절률 항 = 1.
""")

# ── Layout: two columns for sample / reference ─────────────────────────────
st.subheader("📁 데이터 업로드")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔵 Sample")
    abs_samp_file = st.file_uploader("Absorbance — Sample", type=['csv','txt','xlsx','xls'], key='abs_s')
    pl_samp_file  = st.file_uploader("PL Spectrum — Sample",  type=['csv','txt','xlsx','xls'], key='pl_s')

with col2:
    st.markdown("#### 🟠 Reference")
    abs_ref_file  = st.file_uploader("Absorbance — Reference", type=['csv','txt','xlsx','xls'], key='abs_r')
    pl_ref_file   = st.file_uploader("PL Spectrum — Reference",  type=['csv','txt','xlsx','xls'], key='pl_r')

st.subheader("⚙️ 측정 파라미터")
c1, c2, c3, c4 = st.columns(4)
excitation_nm = c1.number_input("여기 파장 (nm)", value=365.0, step=0.5, format="%.1f")
plqy_ref      = c2.number_input("Reference PLQY (Φ_ref)", value=0.95, min_value=0.001, max_value=1.0, step=0.01, format="%.4f")
n_ratio       = c3.number_input("굴절률 비 (n_samp/n_ref)", value=1.0, step=0.001, format="%.4f",
                                 help="동일 용매면 1.0. 다른 용매면 해당 값 입력")
with c4:
    st.markdown("**PL 적분 범위 (nm)**")
    pl_range_lo = st.number_input("from", value=400.0, format="%.1f", key='pl_lo')
    pl_range_hi = st.number_input("to",   value=800.0, format="%.1f", key='pl_hi')

# ── Multiple measurements support ──────────────────────────────────────────
st.subheader("📊 다중 농도/샘플 PLQY (선택)")
multi_mode = st.checkbox("다중 샘플 모드 활성화 (농도별 PLQY 계산)", value=False)
multi_pl_files, concentrations, conc_unit = [], [], "µM"
if multi_mode:
    multi_pl_files = st.file_uploader("PL 파일들 (농도 순서대로)",
                                       type=['csv','txt','xlsx','xls'],
                                       accept_multiple_files=True, key='multi_pl')
    conc_unit = st.text_input("농도 단위", value="µM")
    if multi_pl_files:
        conc_df = pd.DataFrame({'파일': [f.name for f in multi_pl_files],
                                 f'농도 ({conc_unit})': [0.0]*len(multi_pl_files)})
        edited = st.data_editor(conc_df, use_container_width=True, key='conc_table')
        concentrations = edited[f'농도 ({conc_unit})'].tolist()

# ── Calculate ───────────────────────────────────────────────────────────────
if st.button("🔄 PLQY 계산", type="primary"):
    # Load all files
    abs_s_wl, abs_s_inten = load_spectrum_file(abs_samp_file)
    pl_s_wl,  pl_s_inten  = load_spectrum_file(pl_samp_file)
    abs_r_wl, abs_r_inten = load_spectrum_file(abs_ref_file)
    pl_r_wl,  pl_r_inten  = load_spectrum_file(pl_ref_file)

    missing = []
    if abs_s_wl is None: missing.append("Abs Sample")
    if pl_s_wl  is None: missing.append("PL Sample")
    if abs_r_wl is None: missing.append("Abs Reference")
    if pl_r_wl  is None: missing.append("PL Reference")
    if missing:
        st.error(f"누락된 파일: {', '.join(missing)}")
        st.stop()

    # Abs at excitation wavelength
    A_samp = float(np.interp(excitation_nm, abs_s_wl, abs_s_inten))
    A_ref  = float(np.interp(excitation_nm, abs_r_wl, abs_r_inten))

    # PL integration
    def integrate_pl(wl, inten, lo, hi):
        mask = (wl >= lo) & (wl <= hi)
        if mask.sum() < 2:
            return np.trapz(inten, wl)
        return np.trapz(inten[mask], wl[mask])

    PL_samp = integrate_pl(pl_s_wl, pl_s_inten, pl_range_lo, pl_range_hi)
    PL_ref  = integrate_pl(pl_r_wl, pl_r_inten, pl_range_lo, pl_range_hi)

    # PLQY
    if A_samp <= 0 or A_ref <= 0:
        st.error("흡광도 값이 0 이하입니다. 파일을 확인하세요.")
        st.stop()

    plqy_samp = plqy_ref * (PL_samp / PL_ref) * (A_ref / A_samp) * (n_ratio**2)

    # ── Results display ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 계산 결과")

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    col_r1.metric("A_sample (λ_exc)", f"{A_samp:.4f}")
    col_r2.metric("A_reference (λ_exc)", f"{A_ref:.4f}")
    col_r3.metric("PL ratio (samp/ref)", f"{PL_samp/PL_ref:.4f}")
    col_r4.metric("**PLQY (Φ)**", f"{plqy_samp*100:.2f} %",
                  delta=f"{'⚠ Abs > 0.1' if A_samp > 0.1 else 'Abs OK'}")

    if A_samp > 0.1:
        st.warning(f"Sample 흡광도 ({A_samp:.3f}) > 0.1 — inner filter effect 보정을 고려하세요.")

    st.markdown(f"""
    | 파라미터 | 값 |
    |---|---|
    | Φ_ref | {plqy_ref:.4f} |
    | A_sample @ {excitation_nm:.1f} nm | {A_samp:.4f} |
    | A_reference @ {excitation_nm:.1f} nm | {A_ref:.4f} |
    | PL integral (sample) | {PL_samp:.4f} |
    | PL integral (reference) | {PL_ref:.4f} |
    | 굴절률 비 (n_s/n_r)² | {n_ratio**2:.4f} |
    | **PLQY (Φ_sample)** | **{plqy_samp:.4f} ({plqy_samp*100:.2f}%)** |
    """)

    # ── Plots ──────────────────────────────────────────────────────────────
    tab_a, tab_b = st.tabs(["Absorbance", "PL Spectra"])
    with tab_a:
        fig_abs = make_figure(title="Absorbance Spectra")
        style_axes(fig_abs, "Wavelength (nm)", "Absorbance")
        fig_abs.add_trace(go.Scatter(x=abs_s_wl, y=abs_s_inten, name='Sample',
                                     line=dict(color=COLORS[0], width=2)))
        fig_abs.add_trace(go.Scatter(x=abs_r_wl, y=abs_r_inten, name='Reference',
                                     line=dict(color=COLORS[1], width=2)))
        fig_abs.add_vline(x=excitation_nm,
                          line=dict(color='rgba(255,255,100,0.6)', dash='dash', width=1.5),
                          annotation_text=f"λ_exc = {excitation_nm:.0f} nm",
                          annotation_font_color='rgba(255,255,100,0.8)')
        st.plotly_chart(fig_abs, use_container_width=True)

    with tab_b:
        fig_pl = make_figure(title="PL Spectra (raw)")
        style_axes(fig_pl, "Wavelength (nm)", "PL Intensity (a.u.)")
        mask_s = (pl_s_wl >= pl_range_lo) & (pl_s_wl <= pl_range_hi)
        mask_r = (pl_r_wl >= pl_range_lo) & (pl_r_wl <= pl_range_hi)
        fig_pl.add_trace(go.Scatter(x=pl_s_wl, y=pl_s_inten, name='Sample',
                                    line=dict(color=COLORS[0], width=2)))
        fig_pl.add_trace(go.Scatter(x=pl_r_wl, y=pl_r_inten, name='Reference',
                                    line=dict(color=COLORS[1], width=2)))
        # Shade integration region
        fig_pl.add_vrect(x0=pl_range_lo, x1=pl_range_hi,
                         fillcolor='rgba(100,200,100,0.06)',
                         line=dict(color='rgba(100,200,100,0.3)', width=1),
                         annotation_text="Integration range")
        st.plotly_chart(fig_pl, use_container_width=True)

    # ── Multi-sample mode ──────────────────────────────────────────────────
    if multi_mode and multi_pl_files:
        st.markdown("---")
        st.subheader("📊 다중 샘플 PLQY")
        from utils.io_utils import load_spectrum_file as lsf
        multi_results = []
        fig_multi = make_figure(title="PLQY vs Concentration")
        style_axes(fig_multi, f"Concentration ({conc_unit})", "PLQY (Φ)")

        from utils.plot_utils import rainbow_colors
        m_colors = rainbow_colors(len(multi_pl_files))
        for i, (f, conc) in enumerate(zip(multi_pl_files, concentrations)):
            wl_m, inten_m = lsf(f)
            if wl_m is None: continue
            pl_m = integrate_pl(wl_m, inten_m, pl_range_lo, pl_range_hi)
            plqy_m = plqy_ref * (pl_m / PL_ref) * (A_ref / A_samp) * (n_ratio**2)
            multi_results.append({'File': f.name, f'Conc ({conc_unit})': conc, 'PLQY': plqy_m, 'PLQY (%)': plqy_m*100})

        if multi_results:
            res_df = pd.DataFrame(multi_results)
            fig_multi.add_trace(go.Scatter(
                x=res_df[f'Conc ({conc_unit})'], y=res_df['PLQY (%)'],
                mode='markers+lines',
                marker=dict(size=10, color=COLORS[0], line=dict(color='white', width=1.5)),
                line=dict(color=COLORS[0], width=2)
            ))
            st.plotly_chart(fig_multi, use_container_width=True)
            st.dataframe(res_df, use_container_width=True)
            excel_bytes = to_excel_download({'PLQY Results': res_df})
            st.download_button("📥 다중 PLQY Excel", data=excel_bytes,
                               file_name="plqy_multi.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Download single result
    result_df = pd.DataFrame([{
        'Parameter': p, 'Value': v
    } for p, v in [
        ('Excitation (nm)', excitation_nm),
        ('A_sample', A_samp), ('A_reference', A_ref),
        ('PL integral sample', PL_samp), ('PL integral reference', PL_ref),
        ('PLQY_ref', plqy_ref), ('n_ratio^2', n_ratio**2),
        ('PLQY_sample', plqy_samp), ('PLQY_sample (%)', plqy_samp*100),
    ]])
    excel_bytes = to_excel_download({'PLQY': result_df})
    st.download_button("📥 결과 Excel 다운로드", data=excel_bytes,
                       file_name="plqy_result.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
