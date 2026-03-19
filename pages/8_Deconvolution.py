"""
Module 8 — Spectral Deconvolution: Multi-Gaussian/Voigt Fitting
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
from utils.fitting_utils import (gaussian, voigt_approx, fwhm_from_gaussian_sigma, detect_peaks)
from utils.plot_utils import make_figure, style_axes, COLORS, rainbow_colors

@st.cache_data(show_spinner="피팅 중...")
def _cached_deconv_fit(wl_bytes: bytes, inten_bytes: bytes,
                        p0: tuple, bounds_lo: tuple, bounds_hi: tuple,
                        n_comp: int, peak_shape: str, n_baseline: int):
    """Deconvolution curve_fit — 동일 데이터+파라미터면 캐시 반환."""
    from scipy.optimize import curve_fit as _cf
    from utils.fitting_utils import gaussian, voigt_approx
    wl    = np.frombuffer(wl_bytes,    dtype=float)
    inten = np.frombuffer(inten_bytes, dtype=float)

    n_params_per = 4 if peak_shape == "Voigt (pseudo)" else 3

    def model(x, *params):
        components     = params[:n_comp * n_params_per]
        baseline_params = params[n_comp * n_params_per:]
        y = np.zeros_like(x, dtype=float)
        for k in range(n_comp):
            p = components[k * n_params_per:(k+1) * n_params_per]
            if peak_shape == "Gaussian":
                amp, cen, sig = p
                y += gaussian(x, amp, abs(cen), abs(sig))
            else:
                amp, cen, sig, gam = p
                y += voigt_approx(x, amp, cen, abs(sig), abs(gam))
        if n_baseline == 1:
            y += baseline_params[0]
        elif n_baseline == 2:
            y += baseline_params[0] + baseline_params[1] * x
        return y

    popt, pcov = _cf(model, wl, inten, p0=list(p0),
                     bounds=(list(bounds_lo), list(bounds_hi)), maxfev=100000)
    return popt, pcov

st.set_page_config(page_title="Deconvolution | PL Analyzer", layout="wide", page_icon="🔬")
st.title("🔬 Spectral Deconvolution — Multi-component Fitting")
st.markdown("여러 Gaussian/Voigt 성분으로 스펙트럼을 분해합니다. 피크 수와 초기값을 대화형으로 조정하세요.")

# ── Upload ──────────────────────────────────────────────────────────────────
col_up, col_set = st.columns([2, 1])
with col_up:
    deconv_files = st.file_uploader("스펙트럼 파일(들)",
                                     type=['csv','txt','xlsx','xls'], accept_multiple_files=True)
with col_set:
    peak_shape = st.radio("피크 모양", ["Gaussian", "Voigt (pseudo)"], horizontal=True)
    n_components = st.number_input("피크 성분 수", min_value=1, max_value=10, value=2, step=1)
    baseline_mode = st.radio("기준선", ["없음", "상수", "선형"], horizontal=True)
    normalize_dc = st.checkbox("최대값 정규화", value=False)
    show_res = st.checkbox("잔차 표시", value=True)

if not deconv_files:
    st.info("스펙트럼 파일을 업로드하세요.")
    st.stop()

spectra = load_multiple_files(deconv_files)
if not spectra:
    st.error("파일 로딩 실패"); st.stop()

sel_file = st.selectbox("분석할 파일", [s['name'] for s in spectra])
s = next(x for x in spectra if x['name'] == sel_file)
wl = s['wavelength']
inten = s['intensity'].copy()
if normalize_dc and inten.max() > 0: inten /= inten.max()

# ── Fitting range ─────────────────────────────────────────────────────────────
st.subheader("📐 피팅 범위")
c1, c2 = st.columns(2)
wl_lo = c1.number_input("시작 λ (nm)", value=float(wl.min()), format="%.2f")
wl_hi = c2.number_input("끝 λ (nm)",   value=float(wl.max()), format="%.2f")
mask = (wl >= wl_lo) & (wl <= wl_hi)
wl_fit = wl[mask]; inten_fit = inten[mask]

# ── Auto-detect peaks as initial guess ───────────────────────────────────────
auto_peaks_idx, auto_peaks_wl, auto_peaks_inten = detect_peaks(wl_fit, inten_fit, prominence=0.05)

# ── Initial value table ───────────────────────────────────────────────────────
st.subheader("🎛 성분 초기값 설정")

n = int(n_components)
init_data = {'성분': [f'P{i+1}' for i in range(n)],
             'Amplitude': [], 'Center (nm)': [], 'Sigma (nm)': []}
if peak_shape == "Voigt (pseudo)":
    init_data['Gamma (nm)'] = []

for i in range(n):
    # Pre-fill from auto-detected peaks if available
    if i < len(auto_peaks_wl):
        amp0 = float(auto_peaks_inten[i])
        cen0 = float(auto_peaks_wl[i])
    else:
        amp0 = float(inten_fit.max()) * (0.5 ** i)
        cen0 = float(wl_fit.mean()) + i * 20
    init_data['Amplitude'].append(amp0)
    init_data['Center (nm)'].append(cen0)
    init_data['Sigma (nm)'].append(15.0)
    if peak_shape == "Voigt (pseudo)":
        init_data['Gamma (nm)'].append(10.0)

init_df = pd.DataFrame(init_data)
edited_init = st.data_editor(init_df, use_container_width=True, key='deconv_init')

# Constraints
st.subheader("🔒 파라미터 제약 (선택)")
with st.expander("제약 조건 설정"):
    fix_centers = st.checkbox("Center 고정 (초기값 그대로)", value=False)
    fix_sigmas  = st.checkbox("Sigma 고정", value=False)
    center_tol  = st.slider("Center 허용 범위 ±(nm)", 0.0, 100.0, 20.0, 1.0) if not fix_centers else 0.0

# Fitting
if st.button("🔄 피팅 실행", type="primary"):
    n_params_per = 4 if peak_shape == "Voigt (pseudo)" else 3
    n_baseline   = {'없음': 0, '상수': 1, '선형': 2}[baseline_mode]

    # Build p0 and bounds
    p0, lo, hi = [], [], []
    for i in range(n):
        row  = edited_init.iloc[i]
        amp0 = float(row['Amplitude'])
        cen0 = float(row['Center (nm)'])
        sig0 = float(row['Sigma (nm)'])
        p0  += [amp0, cen0, sig0]
        lo  += [0,
                cen0 if fix_centers else max(wl_lo, cen0 - center_tol),
                0.1  if not fix_sigmas else sig0 * 0.99]
        hi  += [amp0 * 10,
                cen0 if fix_centers else min(wl_hi, cen0 + center_tol),
                300  if not fix_sigmas else sig0 * 1.01]
        if peak_shape == "Voigt (pseudo)":
            gam0 = float(row['Gamma (nm)'])
            p0.append(gam0); lo.append(0.1); hi.append(200)

    if n_baseline >= 1:
        p0.append(0.0); lo.append(-inten_fit.max()); hi.append(inten_fit.max())
    if n_baseline == 2:
        p0.append(0.0); lo.append(-1); hi.append(1)

    # Rebuild model for result evaluation (캐시 함수가 내부에서 동일 모델 사용)
    def model(x, *params):
        y = np.zeros_like(x, dtype=float)
        for k in range(n):
            p = params[k * n_params_per:(k+1) * n_params_per]
            if peak_shape == "Gaussian":
                amp, cen, sig = p
                y += gaussian(x, amp, abs(cen), abs(sig))
            else:
                amp, cen, sig, gam = p
                y += voigt_approx(x, amp, cen, abs(sig), abs(gam))
        bp = params[n * n_params_per:]
        if n_baseline == 1: y += bp[0]
        elif n_baseline == 2: y += bp[0] + bp[1] * x
        return y

    try:
        popt, pcov = _cached_deconv_fit(
            wl_fit.tobytes(), inten_fit.tobytes(),
            tuple(p0), tuple(lo), tuple(hi),
            n, peak_shape, n_baseline
        )
        perr = np.sqrt(np.diag(pcov))
        wl_fine = np.linspace(wl_fit.min(), wl_fit.max(), 2000)
        y_total = model(wl_fine, *popt)
        y_total_data = model(wl_fit, *popt)
        residuals = inten_fit - y_total_data
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((inten_fit - inten_fit.mean())**2)
        r2 = 1 - ss_res / ss_tot

        # ── Plot ──────────────────────────────────────────────────────────────
        if show_res:
            fig = make_subplots(rows=2, cols=1, row_heights=[0.78, 0.22],
                                shared_xaxes=True, vertical_spacing=0.04)
        else:
            fig = make_subplots(rows=1, cols=1)

        comp_colors = rainbow_colors(n)
        result_rows = []
        total_area = 0

        # Data
        fig.add_trace(go.Scatter(x=wl_fit, y=inten_fit, name='Data', mode='markers',
                                  marker=dict(color='rgba(180,180,255,0.5)', size=4)), row=1, col=1)

        areas = []
        for k in range(n):
            p = popt[k * n_params_per:(k+1) * n_params_per]
            pe = perr[k * n_params_per:(k+1) * n_params_per]

            if peak_shape == "Gaussian":
                amp, cen, sig = p
                amp_e, cen_e, sig_e = pe
                y_comp = gaussian(wl_fine, amp, cen, abs(sig))
                fwhm = fwhm_from_gaussian_sigma(sig)
                area = amp * abs(sig) * np.sqrt(2 * np.pi)
                row_dict = {
                    'Component': f'P{k+1}',
                    'Amplitude': f'{amp:.4g} ± {amp_e:.4g}',
                    'Center (nm)': f'{cen:.3f} ± {cen_e:.3f}',
                    'Sigma (nm)': f'{abs(sig):.3f} ± {sig_e:.3f}',
                    'FWHM (nm)': f'{fwhm:.3f}',
                    'Area': f'{area:.4g}',
                }
            else:
                amp, cen, sig, gam = p
                amp_e, cen_e, sig_e, gam_e = pe
                y_comp = voigt_approx(wl_fine, amp, cen, abs(sig), abs(gam))
                fwhm = fwhm_from_gaussian_sigma(sig)  # approximate
                area = np.trapz(y_comp, wl_fine)
                row_dict = {
                    'Component': f'P{k+1}',
                    'Amplitude': f'{amp:.4g} ± {amp_e:.4g}',
                    'Center (nm)': f'{cen:.3f} ± {cen_e:.3f}',
                    'Sigma (nm)': f'{abs(sig):.3f} ± {sig_e:.3f}',
                    'Gamma (nm)': f'{abs(gam):.3f} ± {gam_e:.3f}',
                    'FWHM (approx, nm)': f'{fwhm:.3f}',
                    'Area': f'{area:.4g}',
                }

            areas.append(area)
            total_area += area

            # Hex fill color
            import colorsys
            hue = 0.75 - 0.65 * k / max(n-1, 1)
            rgb = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
            fill_rgba = f'rgba({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)},0.12)'

            fig.add_trace(go.Scatter(x=wl_fine, y=y_comp,
                                      name=f'P{k+1} ({popt[k*n_params_per+1]:.1f} nm)',
                                      line=dict(color=comp_colors[k], width=1.5, dash='dot'),
                                      fill='tozeroy', fillcolor=fill_rgba), row=1, col=1)
            result_rows.append(row_dict)

        # Total fit
        fig.add_trace(go.Scatter(x=wl_fine, y=y_total, name='Total fit',
                                  line=dict(color='white', width=2.5, dash='dash')), row=1, col=1)

        if show_res:
            fig.add_trace(go.Scatter(x=wl_fit, y=residuals, name='Residuals', mode='lines',
                                      line=dict(color='rgba(232,92,76,0.8)', width=1.5),
                                      fill='tozeroy', fillcolor='rgba(232,92,76,0.1)'), row=2, col=1)
            fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.3)', dash='dash'), row=2, col=1)
            fig.update_xaxes(title_text="Wavelength (nm)", row=2, col=1)
            fig.update_yaxes(title_text="Residuals", row=2, col=1)

        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(18,18,28,0.95)',
                           plot_bgcolor='rgba(18,18,28,0.95)',
                           title=f"Spectral Deconvolution — {sel_file}  (R² = {r2:.5f})",
                           height=600, font=dict(family='monospace', size=13))
        fig.update_xaxes(title_text="Wavelength (nm)", row=1, col=1)
        fig.update_yaxes(title_text="Intensity (a.u.)", row=1, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # Area fraction
        for i, row_d in enumerate(result_rows):
            row_d['Area fraction (%)'] = f'{areas[i]/total_area*100:.1f}'

        st.markdown(f"**R² = {r2:.6f}**")
        result_df = pd.DataFrame(result_rows)
        st.dataframe(result_df, use_container_width=True)

        # Area fraction bar chart
        fig_area = go.Figure(go.Bar(
            x=[f'P{i+1}' for i in range(n)],
            y=[a/total_area*100 for a in areas],
            marker=dict(color=comp_colors, line=dict(color='white', width=1)),
            text=[f'{a/total_area*100:.1f}%' for a in areas],
            textposition='outside'
        ))
        fig_area.update_layout(template='plotly_dark', paper_bgcolor='rgba(18,18,28,0.95)',
                                plot_bgcolor='rgba(18,18,28,0.95)',
                                title='Area Fraction per Component (%)',
                                yaxis_title='Area Fraction (%)', height=300,
                                font=dict(family='monospace', size=13))
        st.plotly_chart(fig_area, use_container_width=True)

        # Download
        export_dict = {
            'Fit Parameters': result_df,
            'Fit Curve': pd.DataFrame({'Wavelength': wl_fine, 'Total Fit': y_total}),
            'Data+Residuals': pd.DataFrame({'Wavelength': wl_fit, 'Data': inten_fit, 'Residuals': residuals}),
        }
        for k in range(n):
            p = popt[k * n_params_per:(k+1) * n_params_per]
            if peak_shape == "Gaussian":
                amp, cen, sig = p
                comp_y = gaussian(wl_fine, amp, cen, abs(sig))
            else:
                amp, cen, sig, gam = p
                comp_y = voigt_approx(wl_fine, amp, cen, abs(sig), abs(gam))
            export_dict[f'Component P{k+1}'] = pd.DataFrame({'Wavelength': wl_fine, f'P{k+1}': comp_y})

        excel_bytes = to_excel_download({k[:31]: v for k, v in export_dict.items()})
        st.download_button("📥 디컨볼루션 결과 Excel", data=excel_bytes,
                           file_name="deconvolution.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    except Exception as e:
        st.error(f"피팅 실패: {e}\n성분 수나 초기값을 조정해보세요.")
        import traceback
        with st.expander("오류 상세"):
            st.code(traceback.format_exc())
