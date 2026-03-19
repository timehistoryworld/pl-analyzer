"""
Module 4 — Stern-Volmer Quenching Analysis
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io_utils import load_multiple_files, to_excel_download_dict as to_excel_download
from utils.fitting_utils import stern_volmer_linear, stern_volmer_modified, stern_volmer_combined
from utils.plot_utils import make_figure, style_axes, COLORS, rainbow_colors

@st.cache_data(show_spinner=False)
def _cached_sv_fit(concs: tuple, i0_i: tuple, model_name: str, p0: tuple):
    """Stern-Volmer curve_fit 캐시."""
    from scipy.optimize import curve_fit as _cf
    from utils.fitting_utils import stern_volmer_linear, stern_volmer_modified, stern_volmer_combined
    concs_arr = np.array(concs)
    i0_i_arr  = np.array(i0_i)
    model_map = {
        "선형": stern_volmer_linear,
        "Modified": stern_volmer_modified,
        "Combined": stern_volmer_combined,
    }
    fn = model_map[model_name]
    popt, pcov = _cf(fn, concs_arr, i0_i_arr, p0=list(p0),
                     maxfev=10000, bounds=(0, np.inf))
    return popt, pcov

st.set_page_config(page_title="Stern-Volmer | PL Analyzer", layout="wide", page_icon="📉")
st.title("📉 Stern-Volmer — Quenching Analysis")
st.markdown("""
다중 농도 파일 업로드 → 각 농도별 PL 강도 추출 → Stern-Volmer plot 생성 및 파라미터 추출
""")

# ── Upload & sample info ────────────────────────────────────────────────────
st.subheader("📁 파일 업로드")
sv_files = st.file_uploader("PL 파일들 (quencher 농도 순서대로)",
                              type=['csv','txt','xlsx','xls'],
                              accept_multiple_files=True)

if not sv_files:
    st.info("여러 quencher 농도에서 측정한 PL 파일들을 업로드하세요 (농도 오름차순 권장)")
    st.stop()

spectra = load_multiple_files(sv_files)
if not spectra:
    st.error("파일 로딩 실패"); st.stop()

st.success(f"{len(spectra)}개 파일 로드 완료")

# ── Sample parameter table ──────────────────────────────────────────────────
st.subheader("⚙️ 샘플 파라미터")
col1, col2, col3 = st.columns(3)
conc_unit = col1.text_input("Quencher 농도 단위", value="µM")
fluor_lifetime_ns = col2.number_input("형광 수명 τ₀ (ns, optional)", value=0.0, min_value=0.0,
                                       help="0이면 kq 계산 건너뜀")
int_method = col3.radio("PL 강도 추출 방법",
                          ["피크 최대값", "적분 (범위 지정)", "특정 파장에서 강도"])

pl_lo, pl_hi, int_wl = 400.0, 800.0, 520.0
if int_method == "적분 (범위 지정)":
    cc1, cc2 = st.columns(2)
    pl_lo = cc1.number_input("적분 시작 (nm)", value=400.0, format="%.1f")
    pl_hi = cc2.number_input("적분 끝 (nm)",   value=800.0, format="%.1f")
elif int_method == "특정 파장에서 강도":
    int_wl = st.number_input("파장 (nm)", value=520.0, format="%.1f")

# ── Concentration input table ────────────────────────────────────────────────
st.subheader("📊 농도 입력")
init_df = pd.DataFrame({
    '파일': [s['name'] for s in spectra],
    f'[Quencher] ({conc_unit})': [0.0] + [float(i) for i in range(1, len(spectra))],
    '포함 여부': [True] * len(spectra)
})
edited_df = st.data_editor(init_df, use_container_width=True, key='sv_table')

# ── Extract intensities ─────────────────────────────────────────────────────
def extract_intensity(wl, inten, method, lo=400, hi=800, target_wl=520):
    if method == "피크 최대값":
        return float(np.max(inten))
    elif method == "적분 (범위 지정)":
        mask = (wl >= lo) & (wl <= hi)
        if mask.sum() < 2: return float(np.trapz(inten, wl))
        return float(np.trapz(inten[mask], wl[mask]))
    else:
        return float(np.interp(target_wl, wl, inten))

concs, intensities, names = [], [], []
for _, row in edited_df.iterrows():
    if not row['포함 여부']: continue
    s = next((x for x in spectra if x['name'] == row['파일']), None)
    if s is None: continue
    I = extract_intensity(s['wavelength'], s['intensity'],
                          int_method, pl_lo, pl_hi, int_wl)
    concs.append(float(row[f'[Quencher] ({conc_unit})']))
    intensities.append(I)
    names.append(row['파일'])

if len(concs) < 2:
    st.warning("최소 2개 이상의 데이터 포인트가 필요합니다")
    st.stop()

concs = np.array(concs)
intensities = np.array(intensities)
I0 = intensities[concs == concs.min()][0] if (concs == concs.min()).any() else intensities[0]
I0_I = I0 / intensities  # I₀/I ratio

# ── Fitting models ──────────────────────────────────────────────────────────
st.subheader("📈 Stern-Volmer Plot & Fitting")
sv_model = st.selectbox("피팅 모델", [
    "선형 SV (dynamic 또는 static quenching)",
    "Modified SV (two-population, 위로 굽은 경우)",
    "Combined SV (dynamic + static, 위로 굽은 경우)",
    "모두 표시 (비교)"
])

tab_sv, tab_data, tab_spectra = st.tabs(["SV Plot", "데이터 테이블", "원시 스펙트럼"])

with tab_sv:
    fig_sv = make_figure(title="Stern-Volmer Plot")
    style_axes(fig_sv, f"[Quencher] ({conc_unit})", "I₀ / I")

    # Data points
    fig_sv.add_trace(go.Scatter(
        x=concs, y=I0_I, mode='markers',
        name='Data',
        marker=dict(size=10, color='white', line=dict(color=COLORS[0], width=2))
    ))

    c_fit = np.linspace(concs.min(), concs.max(), 300)
    fit_results = []

    model_short = {"선형 SV": "선형", "Modified SV": "Modified", "Combined SV": "Combined"}

    def try_fit(model_func, p0, name, color):
        try:
            popt, pcov = _cached_sv_fit(
                tuple(concs.tolist()), tuple(I0_I.tolist()),
                model_short.get(name, "선형"), tuple(p0)
            )
            perr = np.sqrt(np.diag(pcov))
            y_fit = model_func(c_fit, *popt)
            residuals = I0_I - model_func(concs, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((I0_I - I0_I.mean())**2)
            r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
            fig_sv.add_trace(go.Scatter(x=c_fit, y=y_fit, name=name,
                                         line=dict(color=color, width=2.5, dash='dash')))
            return popt, perr, r2
        except Exception as e:
            st.warning(f"{name} 피팅 실패: {e}")
            return None, None, None

    models_to_run = []
    if "선형" in sv_model or "모두" in sv_model:
        models_to_run.append(("선형 SV", stern_volmer_linear, [I0, 1e-3], COLORS[0]))
    if "Modified" in sv_model or "모두" in sv_model:
        models_to_run.append(("Modified SV", stern_volmer_modified, [I0, 1e-3, 0.8], COLORS[1]))
    if "Combined" in sv_model or "모두" in sv_model:
        models_to_run.append(("Combined SV", stern_volmer_combined, [I0, 1e-3, 1e-3], COLORS[2]))

    for model_name, model_func, p0, color in models_to_run:
        popt, perr, r2 = try_fit(model_func, p0, model_name, color)
        if popt is not None:
            fit_results.append((model_name, popt, perr, r2))

    st.plotly_chart(fig_sv, use_container_width=True)

    # Results
    for model_name, popt, perr, r2 in fit_results:
        with st.expander(f"📋 {model_name} 결과 (R² = {r2:.5f})", expanded=True):
            if "선형" in model_name:
                I0_fit, Ksv = popt
                I0_err, Ksv_err = perr
                st.markdown(f"""
                | 파라미터 | 값 |
                |---|---|
                | I₀ | {I0_fit:.4g} ± {I0_err:.4g} |
                | **Ksv** | **{Ksv:.4g} ± {Ksv_err:.4g} {conc_unit}⁻¹** |
                | R² | {r2:.6f} |
                """)
                if fluor_lifetime_ns > 0:
                    tau_s = fluor_lifetime_ns * 1e-9
                    conc_M_factor = 1.0  # assume input is already in M; adjust if µM etc.
                    kq = Ksv / tau_s
                    st.markdown(f"| kq (bimolecular quenching constant) | {kq:.4g} M⁻¹s⁻¹ |")

            elif "Modified" in model_name:
                I0_fit, Ksv, fa = popt
                st.markdown(f"""
                | 파라미터 | 값 |
                |---|---|
                | I₀ | {I0_fit:.4g} ± {perr[0]:.4g} |
                | **Ksv** | **{Ksv:.4g} ± {perr[1]:.4g} {conc_unit}⁻¹** |
                | **fa (accessible fraction)** | **{fa:.4f} ± {perr[2]:.4f}** |
                | R² | {r2:.6f} |
                """)

            elif "Combined" in model_name:
                I0_fit, Kd, Ka = popt
                st.markdown(f"""
                | 파라미터 | 값 |
                |---|---|
                | I₀ | {I0_fit:.4g} ± {perr[0]:.4g} |
                | **Kd (dynamic)** | **{Kd:.4g} ± {perr[1]:.4g} {conc_unit}⁻¹** |
                | **Ka (static)** | **{Ka:.4g} ± {perr[2]:.4g} {conc_unit}⁻¹** |
                | R² | {r2:.6f} |
                """)

with tab_data:
    data_df = pd.DataFrame({
        '파일': names,
        f'[Quencher] ({conc_unit})': concs,
        'I (PL intensity)': intensities,
        'I₀ / I': I0_I,
    })
    st.dataframe(data_df, use_container_width=True)
    excel_bytes = to_excel_download({'SV Data': data_df})
    st.download_button("📥 SV 데이터 Excel", data=excel_bytes,
                       file_name="stern_volmer.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tab_spectra:
    colors_sp = rainbow_colors(len(spectra))
    fig_sp = make_figure(title="PL Spectra (all concentrations)")
    style_axes(fig_sp, "Wavelength (nm)", "PL Intensity (a.u.)")
    for i, s in enumerate(spectra):
        fig_sp.add_trace(go.Scatter(x=s['wavelength'], y=s['intensity'],
                                     name=s['name'], line=dict(color=colors_sp[i], width=2)))
    st.plotly_chart(fig_sp, use_container_width=True)
