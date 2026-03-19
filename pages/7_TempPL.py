"""
Module 7 — Temperature-Dependent PL Analysis
Varshni, Bose-Einstein, thermal quenching, linewidth vs T
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io_utils import load_multiple_files, to_excel_download_dict as to_excel_download
from utils.fitting_utils import (varshni, bose_einstein, pl_quenching,
                                  fwhm_from_gaussian_sigma, fit_peak_gaussian, detect_peaks)
from utils.plot_utils import make_figure, style_axes, rainbow_colors, COLORS

@st.cache_data(show_spinner=False)
def _cached_varshni(temps: tuple, y_data: tuple, p0: tuple):
    from scipy.optimize import curve_fit as _cf
    from utils.fitting_utils import varshni
    t = np.array(temps); y = np.array(y_data)
    popt, pcov = _cf(varshni, t, y, p0=list(p0), maxfev=10000)
    return popt, pcov

@st.cache_data(show_spinner=False)
def _cached_bose_einstein(temps: tuple, y_data: tuple, p0: tuple):
    from scipy.optimize import curve_fit as _cf
    from utils.fitting_utils import bose_einstein
    t = np.array(temps); y = np.array(y_data)
    popt, pcov = _cf(bose_einstein, t, y, p0=list(p0), maxfev=10000)
    return popt, pcov

@st.cache_data(show_spinner=False)
def _cached_pl_quenching(temps: tuple, integrals: tuple, p0: tuple):
    from scipy.optimize import curve_fit as _cf
    from utils.fitting_utils import pl_quenching
    t = np.array(temps); y = np.array(integrals)
    popt, pcov = _cf(pl_quenching, t, y,
                     p0=list(p0), bounds=([0,0,0],[2,1e6,5]), maxfev=20000)
    return popt, pcov

st.set_page_config(page_title="Temp PL | PL Analyzer", layout="wide", page_icon="🌡")
st.title("🌡 Temperature-Dependent PL Analysis")
st.markdown("온도별 PL 스펙트럼 비교, peak position/FWHM vs T 피팅 (Varshni + Bose-Einstein + 열적 소광)")

# ── Upload ──────────────────────────────────────────────────────────────────
temp_files = st.file_uploader("PL 파일들 (온도별 각 1개)",
                               type=['csv','txt','xlsx','xls'], accept_multiple_files=True)

if not temp_files:
    st.info("온도별로 측정한 PL 스펙트럼 파일들을 업로드하세요.")
    st.stop()

spectra = load_multiple_files(temp_files)
if not spectra:
    st.error("파일 로딩 실패"); st.stop()

st.success(f"{len(spectra)}개 파일 로드")

# ── Temperature input table ─────────────────────────────────────────────────
st.subheader("🌡 온도 입력")
temp_df_init = pd.DataFrame({
    '파일': [s['name'] for s in spectra],
    '온도 (K)': [float(100 + i*20) for i in range(len(spectra))],
    '포함 여부': [True] * len(spectra)
})
temp_df = st.data_editor(temp_df_init, use_container_width=True, key='temp_table')

col1, col2, col3 = st.columns(3)
normalize_t = col1.checkbox("최대값 정규화", value=True)
peak_method = col2.radio("피크 위치 추출 방법", ["최대값 파장", "Gaussian 피팅"])
gauss_window = col3.slider("Gaussian 피팅 창 (nm)", 10, 150, 60) if peak_method == "Gaussian 피팅" else 60

# ── Process ─────────────────────────────────────────────────────────────────
temps, peak_positions, fwhms, pl_integrals, labels = [], [], [], [], []

for _, row in temp_df.iterrows():
    if not row['포함 여부']: continue
    s = next((x for x in spectra if x['name'] == row['파일']), None)
    if s is None: continue

    T = float(row['온도 (K)'])
    wl = s['wavelength']
    inten = s['intensity']

    # Integral
    pl_integrals.append(np.trapz(inten, wl))

    if peak_method == "Gaussian 피팅":
        peak_idx = np.argmax(inten)
        result = fit_peak_gaussian(wl, inten, peak_idx, window_nm=gauss_window)
        if result:
            popt, _, _, _ = result
            peak_positions.append(popt[1])  # center
            fwhms.append(fwhm_from_gaussian_sigma(popt[2]))
        else:
            peak_positions.append(wl[np.argmax(inten)])
            fwhms.append(np.nan)
    else:
        peak_positions.append(float(wl[np.argmax(inten)]))
        # Estimate FWHM from data
        half_max = inten.max() / 2
        above = wl[inten >= half_max]
        fwhms.append(float(above[-1] - above[0]) if len(above) > 1 else np.nan)

    temps.append(T)
    labels.append(row['파일'])

temps = np.array(temps)
peak_positions = np.array(peak_positions)
fwhms = np.array(fwhms)
pl_integrals = np.array(pl_integrals)

# Convert peak position nm → eV for energy fitting
hc = 1239.84193
peak_eV = hc / peak_positions

idx_sort = np.argsort(temps)
temps = temps[idx_sort]
peak_positions = peak_positions[idx_sort]
peak_eV = peak_eV[idx_sort]
fwhms = fwhms[idx_sort]
pl_integrals = pl_integrals[idx_sort]
labels = [labels[i] for i in idx_sort]

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_spectra, tab_peak, tab_fwhm, tab_quench = st.tabs([
    "📈 스펙트럼", "🎯 Peak Position vs T", "📏 FWHM vs T", "🔥 열적 소광"
])

colors_T = rainbow_colors(len(spectra))

with tab_spectra:
    fig_sp = make_figure(title="Temperature-Dependent PL Spectra")
    style_axes(fig_sp, "Wavelength (nm)", "Intensity (a.u.)")
    loaded_spectra = {s['name']: s for s in spectra}

    for i, (lbl, T_val) in enumerate(zip(labels, temps)):
        s = loaded_spectra.get(lbl)
        if s is None: continue
        inten = s['intensity'].copy()
        if normalize_t and inten.max() > 0: inten /= inten.max()
        fig_sp.add_trace(go.Scatter(x=s['wavelength'], y=inten,
                                     name=f'{T_val:.0f} K',
                                     line=dict(color=colors_T[i % len(colors_T)], width=2)))
    st.plotly_chart(fig_sp, use_container_width=True)

with tab_peak:
    st.subheader("Peak Position vs Temperature — Varshni & Bose-Einstein Fitting")

    fit_axis = st.radio("피팅 에너지 축", ["Wavelength (nm)", "Energy (eV)"], horizontal=True)
    y_data = peak_positions if fit_axis == "Wavelength (nm)" else peak_eV
    y_label = "Peak Wavelength (nm)" if fit_axis == "Wavelength (nm)" else "Peak Energy (eV)"

    fig_peak = make_figure(title="Peak Position vs Temperature")
    style_axes(fig_peak, "Temperature (K)", y_label)
    fig_peak.add_trace(go.Scatter(x=temps, y=y_data, mode='markers',
                                   name='Data',
                                   marker=dict(size=10, color='white',
                                               line=dict(color=COLORS[0], width=2))))

    T_fit = np.linspace(temps.min(), temps.max(), 300)

    col_v, col_b = st.columns(2)
    with col_v:
        st.markdown("**Varshni 피팅**")
        fit_varshni = st.checkbox("Varshni 피팅 실행", value=True)
        if fit_varshni:
            try:
                E0_init = y_data[temps == temps.min()][0] if y_data.size else y_data[0]
                popt_v, pcov_v = _cached_varshni(
                    tuple(temps.tolist()), tuple(y_data.tolist()),
                    (E0_init, 1e-4, 100)
                )
                y_v = varshni(T_fit, *popt_v)
                fig_peak.add_trace(go.Scatter(x=T_fit, y=y_v, name='Varshni fit',
                                              line=dict(color=COLORS[1], width=2.5, dash='dash')))
                residuals_v = y_data - varshni(temps, *popt_v)
                r2_v = 1 - np.sum(residuals_v**2) / np.sum((y_data - y_data.mean())**2)
                perr_v = np.sqrt(np.diag(pcov_v))
                st.markdown(f"""
                | 파라미터 | 값 |
                |---|---|
                | E₀ | {popt_v[0]:.5f} ± {perr_v[0]:.5f} |
                | α | {popt_v[1]:.4e} ± {perr_v[1]:.4e} |
                | β (K) | {popt_v[2]:.2f} ± {perr_v[2]:.2f} |
                | R² | {r2_v:.5f} |
                """)
            except Exception as e:
                st.warning(f"Varshni 피팅 실패: {e}")

    with col_b:
        st.markdown("**Bose-Einstein 피팅**")
        fit_bose = st.checkbox("Bose-Einstein 피팅 실행", value=True)
        if fit_bose:
            try:
                E0_init = y_data[temps == temps.min()][0] if y_data.size else y_data[0]
                popt_b, pcov_b = _cached_bose_einstein(
                    tuple(temps.tolist()), tuple(y_data.tolist()),
                    (E0_init, 0.01, 150)
                )
                y_b = bose_einstein(T_fit, *popt_b)
                fig_peak.add_trace(go.Scatter(x=T_fit, y=y_b, name='Bose-Einstein fit',
                                              line=dict(color=COLORS[2], width=2.5, dash='dot')))
                residuals_b = y_data - bose_einstein(temps, *popt_b)
                r2_b = 1 - np.sum(residuals_b**2) / np.sum((y_data - y_data.mean())**2)
                perr_b = np.sqrt(np.diag(pcov_b))
                st.markdown(f"""
                | 파라미터 | 값 |
                |---|---|
                | E₀ | {popt_b[0]:.5f} ± {perr_b[0]:.5f} |
                | a_B | {popt_b[1]:.4e} ± {perr_b[1]:.4e} |
                | θ_B (K) | {popt_b[2]:.2f} ± {perr_b[2]:.2f} |
                | R² | {r2_b:.5f} |
                """)
            except Exception as e:
                st.warning(f"Bose-Einstein 피팅 실패: {e}")

    st.plotly_chart(fig_peak, use_container_width=True)

with tab_fwhm:
    st.subheader("FWHM vs Temperature")
    valid = ~np.isnan(fwhms)
    if valid.sum() < 2:
        st.warning("FWHM 데이터가 부족합니다. Gaussian 피팅 방법을 시도해보세요.")
    else:
        fig_fwhm = make_figure(title="FWHM vs Temperature")
        style_axes(fig_fwhm, "Temperature (K)", "FWHM (nm)")
        fig_fwhm.add_trace(go.Scatter(x=temps[valid], y=fwhms[valid], mode='markers+lines',
                                       name='FWHM data',
                                       marker=dict(size=10, color='white',
                                                   line=dict(color=COLORS[3], width=2)),
                                       line=dict(color=COLORS[3], width=2)))

        # Linear fit
        if st.checkbox("선형 피팅 (Γ vs T)", value=True):
            from scipy.stats import linregress
            slope, intercept, r_val, _, _ = linregress(temps[valid], fwhms[valid])
            T_fit_fwhm = np.linspace(temps[valid].min(), temps[valid].max(), 200)
            fig_fwhm.add_trace(go.Scatter(x=T_fit_fwhm, y=slope*T_fit_fwhm + intercept,
                                           name=f'Linear fit (slope={slope:.4f} nm/K)',
                                           line=dict(color='rgba(255,200,100,0.8)', dash='dash', width=2)))
            st.markdown(f"**선형 피팅**: Γ(T) = {intercept:.3f} + {slope:.4f}·T &nbsp; (R² = {r_val**2:.5f})")

        st.plotly_chart(fig_fwhm, use_container_width=True)

with tab_quench:
    st.subheader("열적 소광 분석 — Integrated PL Intensity vs T")
    fig_q = make_figure(title="Thermal Quenching")
    style_axes(fig_q, "Temperature (K)", "Integrated PL Intensity (a.u.)")
    fig_q.update_yaxes(type='log')
    norm_integral = pl_integrals / pl_integrals.max()
    fig_q.add_trace(go.Scatter(x=temps, y=norm_integral, mode='markers',
                                name='Data',
                                marker=dict(size=10, color='white',
                                            line=dict(color=COLORS[4], width=2))))

    if st.checkbox("열적 소광 피팅: I(T) = I₀ / (1 + A·exp(-Ea/kBT))", value=True):
        try:
            popt_q, pcov_q = _cached_pl_quenching(
                tuple(temps.tolist()), tuple(norm_integral.tolist()),
                (1.0, 1.0, 0.1)
            )
            T_fit_q = np.linspace(temps.min(), temps.max(), 300)
            I_fit_q = pl_quenching(T_fit_q, *popt_q)
            fig_q.add_trace(go.Scatter(x=T_fit_q, y=I_fit_q, name='Thermal quenching fit',
                                        line=dict(color=COLORS[1], width=2.5, dash='dash')))
            perr_q = np.sqrt(np.diag(pcov_q))
            residuals_q = norm_integral - pl_quenching(temps, *popt_q)
            r2_q = 1 - np.sum(residuals_q**2) / np.sum((norm_integral - norm_integral.mean())**2)
            st.markdown(f"""
            | 파라미터 | 값 |
            |---|---|
            | I₀ | {popt_q[0]:.4f} ± {perr_q[0]:.4f} |
            | A (pre-exponential factor) | {popt_q[1]:.4g} ± {perr_q[1]:.4g} |
            | **Ea (eV)** | **{popt_q[2]:.4f} ± {perr_q[2]:.4f}** |
            | R² | {r2_q:.5f} |
            """)
            st.info(f"🔑 Activation energy Ea = {popt_q[2]*1000:.1f} meV — non-radiative 채널의 활성화 에너지")
        except Exception as e:
            st.warning(f"열적 소광 피팅 실패: {e}")

    st.plotly_chart(fig_q, use_container_width=True)

    # Summary table + download
    summary_df = pd.DataFrame({
        'File': labels, 'Temperature (K)': temps,
        'Peak λ (nm)': peak_positions.round(3),
        'Peak E (eV)': peak_eV.round(5),
        'FWHM (nm)': fwhms.round(3),
        'Integrated PL': pl_integrals,
        'Norm. Integrated PL': norm_integral.round(6)
    })
    st.markdown("**요약 테이블**")
    st.dataframe(summary_df, use_container_width=True)
    excel_bytes = to_excel_download({'Temperature PL': summary_df})
    st.download_button("📥 온도 의존성 데이터 Excel", data=excel_bytes,
                       file_name="temp_pl_analysis.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
