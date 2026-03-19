"""
Module 1 — Basics: Averaging, Peak Analysis, Gaussian Fitting
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io_utils import load_multiple_files, interpolate_to_common_grid, to_excel_download
from utils.fitting_utils import detect_peaks, fit_peak_gaussian, gaussian, multi_gaussian, fwhm_from_gaussian_sigma
from utils.plot_utils import make_figure, style_axes, add_spectrum, COLORS, rainbow_colors
from scipy.optimize import curve_fit

st.set_page_config(page_title="Basics | PL Analyzer", layout="wide", page_icon="📊")
st.title("📊 Basics — Peak Analysis & Gaussian Fitting")
st.markdown("다중 스펙트럼 평균화, 피크 자동 감지, 단일/다중 Gaussian 피팅")

# ── File upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader("스펙트럼 파일 업로드 (여러 파일 가능)",
                             type=['csv', 'txt', 'xlsx', 'xls'],
                             accept_multiple_files=True)

if not uploaded:
    st.info("CSV / TXT / Excel 파일을 업로드하세요. 첫 번째 열: 파장(nm), 두 번째 열: 강도")
    st.stop()

spectra = load_multiple_files(uploaded)
if not spectra:
    st.error("파일 로딩 실패")
    st.stop()

st.success(f"{len(spectra)}개 파일 로드 완료")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 분석 설정")
    normalize = st.checkbox("최대값 정규화", value=False)
    baseline_correct = st.checkbox("기준선 보정 (최솟값 차감)", value=False)

    st.subheader("피크 감지")
    prominence = st.slider("Prominence 임계값", 0.01, 0.5, 0.05, 0.01,
                           help="피크 두드러짐 기준 (정규화 후 상대값)")
    min_dist = st.slider("최소 피크 간격 (nm)", 1, 100, 15)

    st.subheader("Gaussian 피팅")
    fit_window = st.slider("피팅 창 너비 (nm)", 10, 200, 60)
    n_gauss = st.number_input("다중 Gaussian 성분 수", 1, 8, 1, 1)
    show_residuals = st.checkbox("잔차 표시", value=True)

# ── Process spectra ───────────────────────────────────────────────────────────
colors = rainbow_colors(len(spectra))

def process(s, normalize, baseline_correct):
    wl = s['wavelength'].copy()
    inten = s['intensity'].copy()
    if baseline_correct:
        inten -= inten.min()
    if normalize:
        mx = inten.max()
        if mx > 0:
            inten /= mx
    return wl, inten

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 스펙트럼 보기", "➕ 평균 & 합산", "🔍 피크 감지", "〰️ Gaussian 피팅"])

# ─────────── Tab 1: Spectrum viewer ──────────────────────────────────────────
with tab1:
    fig = make_figure(title="Loaded Spectra")
    style_axes(fig, "Wavelength (nm)", "Intensity (a.u.)")
    for i, s in enumerate(spectra):
        wl, inten = process(s, normalize, baseline_correct)
        add_spectrum(fig, wl, inten, name=s['name'], color=colors[i])
    st.plotly_chart(fig, use_container_width=True)

    # Table summary
    rows = []
    for s in spectra:
        wl, inten = process(s, normalize, baseline_correct)
        rows.append({
            'File': s['name'],
            'λ range (nm)': f"{wl.min():.1f} – {wl.max():.1f}",
            'Points': len(wl),
            'Peak λ (nm)': f"{wl[np.argmax(inten)]:.2f}",
            'Max intensity': f"{inten.max():.4g}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ─────────── Tab 2: Averaging ─────────────────────────────────────────────────
with tab2:
    st.subheader("스펙트럼 평균 & 합산")

    # Select files to average
    file_names = [s['name'] for s in spectra]
    selected = st.multiselect("평균할 파일 선택", file_names, default=file_names)

    sel_spectra = [s for s in spectra if s['name'] in selected]
    if len(sel_spectra) < 1:
        st.warning("파일을 선택하세요")
    else:
        common_wl, interp_list = interpolate_to_common_grid(sel_spectra)
        stack = np.array(interp_list)

        avg_inten = np.mean(stack, axis=0)
        std_inten = np.std(stack, axis=0)
        sum_inten = np.sum(stack, axis=0)

        fig2 = make_figure(title="Averaged Spectrum")
        style_axes(fig2, "Wavelength (nm)", "Intensity (a.u.)")

        # Individual spectra (faint)
        for i, (s, inten) in enumerate(zip(sel_spectra, interp_list)):
            wl_proc, inten_proc = process({'wavelength': common_wl, 'intensity': inten},
                                           normalize, baseline_correct)
            fig2.add_trace(go.Scatter(
                x=wl_proc, y=inten_proc, name=s['name'], mode='lines',
                line=dict(color=colors[i % len(colors)], width=1, dash='dot'),
                opacity=0.4
            ))

        # Average ± std
        avg_plot = avg_inten.copy()
        if baseline_correct: avg_plot -= avg_plot.min()
        if normalize and avg_plot.max() > 0: avg_plot /= avg_plot.max()

        fig2.add_trace(go.Scatter(
            x=common_wl, y=avg_plot, name='Average',
            mode='lines', line=dict(color='#FFFFFF', width=2.5)
        ))
        fig2.add_trace(go.Scatter(
            x=np.concatenate([common_wl, common_wl[::-1]]),
            y=np.concatenate([avg_plot + std_inten, (avg_plot - std_inten)[::-1]]),
            fill='toself', fillcolor='rgba(255,255,255,0.08)',
            line=dict(color='rgba(255,255,255,0)'), name='±1 SD'
        ))
        st.plotly_chart(fig2, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            df_avg = pd.DataFrame({'Wavelength (nm)': common_wl,
                                   'Average': avg_inten,
                                   'Std Dev': std_inten,
                                   'Sum': sum_inten})
            excel_bytes = to_excel_download({'Average': df_avg})
            st.download_button("📥 평균 데이터 Excel 다운로드", data=excel_bytes,
                               file_name="pl_average.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ─────────── Tab 3: Peak detection ────────────────────────────────────────────
with tab3:
    st.subheader("피크 자동 감지")
    sel_file = st.selectbox("파일 선택", [s['name'] for s in spectra])
    s_sel = next(s for s in spectra if s['name'] == sel_file)
    wl, inten = process(s_sel, normalize, baseline_correct)

    peaks_idx, peaks_wl, peaks_inten = detect_peaks(wl, inten, prominence, min_dist)

    show_deriv = st.checkbox("미분 스펙트럼 표시", value=False)
    order = st.radio("미분 차수", [1, 2], horizontal=True) if show_deriv else 1

    fig3 = make_figure(title=f"Peak Detection — {sel_file}")
    if show_deriv:
        deriv = np.gradient(inten, wl)
        if order == 2:
            deriv = np.gradient(deriv, wl)
        deriv_norm = deriv / np.max(np.abs(deriv) + 1e-30)
        fig3.add_trace(go.Scatter(x=wl, y=deriv_norm, name=f'{order}차 미분',
                                  line=dict(color='rgba(232,184,76,0.7)', dash='dot', width=1.5)))

    style_axes(fig3, "Wavelength (nm)", "Intensity (a.u.)")
    add_spectrum(fig3, wl, inten, name=sel_file, color=COLORS[0])

    # Peak markers
    fig3.add_trace(go.Scatter(
        x=peaks_wl, y=peaks_inten, mode='markers+text',
        marker=dict(color='#FF4444', size=10, symbol='triangle-up',
                    line=dict(color='white', width=1.5)),
        text=[f"{w:.1f}" for w in peaks_wl],
        textposition='top center',
        textfont=dict(size=11, color='#FF8888'),
        name=f'{len(peaks_idx)} Peaks'
    ))
    st.plotly_chart(fig3, use_container_width=True)

    if len(peaks_idx) > 0:
        peak_df = pd.DataFrame({
            'Peak #': range(1, len(peaks_idx)+1),
            'Wavelength (nm)': peaks_wl.round(2),
            'Intensity': peaks_inten.round(4),
            'Rel. Intensity (%)': (peaks_inten / peaks_inten.max() * 100).round(1)
        })
        st.dataframe(peak_df, use_container_width=True)
        excel_bytes = to_excel_download({'Peaks': peak_df})
        st.download_button("📥 피크 데이터 Excel", data=excel_bytes,
                           file_name="pl_peaks.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ─────────── Tab 4: Gaussian fitting ──────────────────────────────────────────
with tab4:
    st.subheader("Gaussian 피팅")
    sel_file2 = st.selectbox("파일 선택 ", [s['name'] for s in spectra], key='gauss_file')
    s_sel2 = next(s for s in spectra if s['name'] == sel_file2)
    wl2, inten2 = process(s_sel2, normalize, baseline_correct)

    st.markdown(f"**{int(n_gauss)}개 Gaussian 성분** 으로 피팅합니다")

    # Initial guesses UI
    st.markdown("**초기값 설정** (각 성분별)")
    p0 = []
    bounds_lo, bounds_hi = [], []

    for k in range(int(n_gauss)):
        with st.expander(f"Gaussian #{k+1} 초기값", expanded=(k == 0)):
            c1, c2, c3 = st.columns(3)
            default_cen = float(wl2[np.argmax(inten2)]) + k * 20
            amp0 = c1.number_input(f"Amplitude", value=float(inten2.max()) / n_gauss,
                                    key=f'amp_{k}', format="%.4f")
            cen0 = c2.number_input(f"Center (nm)", value=default_cen,
                                    key=f'cen_{k}', format="%.2f")
            sig0 = c3.number_input(f"Sigma (nm)", value=15.0,
                                    key=f'sig_{k}', format="%.2f", min_value=0.1)
            p0 += [amp0, cen0, sig0]
            bounds_lo += [0, wl2.min(), 0.5]
            bounds_hi += [inten2.max() * 5, wl2.max(), 300]

    if st.button("🔄 피팅 실행", type="primary"):
        try:
            popt, pcov = curve_fit(multi_gaussian, wl2, inten2,
                                    p0=p0,
                                    bounds=(bounds_lo, bounds_hi),
                                    maxfev=50000)
            perr = np.sqrt(np.diag(pcov))

            wl_fine = np.linspace(wl2.min(), wl2.max(), 2000)
            fit_total = multi_gaussian(wl_fine, *popt)
            residuals = inten2 - multi_gaussian(wl2, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((inten2 - inten2.mean())**2)
            r2 = 1 - ss_res / ss_tot

            # Plot
            fig4 = make_figure(title=f"Gaussian Fitting — {sel_file2}  (R² = {r2:.5f})")
            style_axes(fig4, "Wavelength (nm)", "Intensity (a.u.)")
            add_spectrum(fig4, wl2, inten2, name='Data', color='rgba(180,180,255,0.8)')
            fig4.add_trace(go.Scatter(x=wl_fine, y=fit_total, name='Total fit',
                                      line=dict(color='white', width=2.5, dash='dash')))

            comp_colors = rainbow_colors(int(n_gauss))
            result_rows = []
            total_area = 0
            comp_areas = []

            for k in range(int(n_gauss)):
                amp, cen, sig = popt[3*k], popt[3*k+1], popt[3*k+2]
                amp_err, cen_err, sig_err = perr[3*k], perr[3*k+1], perr[3*k+2]
                fwhm = fwhm_from_gaussian_sigma(sig)
                area = amp * abs(sig) * np.sqrt(2 * np.pi)
                comp_areas.append(area)
                total_area += area

                comp_y = gaussian(wl_fine, amp, cen, sig)
                fig4.add_trace(go.Scatter(
                    x=wl_fine, y=comp_y,
                    name=f'G{k+1} ({cen:.1f} nm)',
                    line=dict(color=comp_colors[k], width=1.5, dash='dot'),
                    fill='tozeroy',
                    fillcolor=f'rgba({",".join(str(int(c*255)) for c in __import__("colorsys").hsv_to_rgb(0.75-0.65*k/max(int(n_gauss)-1,1), 0.85, 0.95))},0.10)'
                ))
                result_rows.append({
                    'Component': f'G{k+1}',
                    'Amplitude': f'{amp:.4g} ± {amp_err:.4g}',
                    'Center (nm)': f'{cen:.3f} ± {cen_err:.3f}',
                    'Sigma (nm)': f'{abs(sig):.3f} ± {sig_err:.3f}',
                    'FWHM (nm)': f'{fwhm:.3f}',
                    'Area': f'{area:.4g}',
                })

            # Residuals subplot
            if show_residuals:
                from plotly.subplots import make_subplots
                fig4_with_res = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25],
                                              shared_xaxes=True, vertical_spacing=0.05)
                for trace in fig4.data:
                    fig4_with_res.add_trace(trace, row=1, col=1)
                fig4_with_res.add_trace(
                    go.Scatter(x=wl2, y=residuals, name='Residuals',
                               mode='lines', line=dict(color='rgba(232,92,76,0.8)', width=1.5),
                               fill='tozeroy', fillcolor='rgba(232,92,76,0.1)'),
                    row=2, col=1
                )
                fig4_with_res.add_hline(y=0, line=dict(color='rgba(255,255,255,0.3)', dash='dash'), row=2, col=1)
                fig4_with_res.update_layout(**{k: v for k, v in
                                               {'template': 'plotly_dark',
                                                'font': dict(family='monospace', size=13),
                                                'paper_bgcolor': 'rgba(18,18,28,0.95)',
                                                'plot_bgcolor': 'rgba(18,18,28,0.95)',
                                                'title': fig4.layout.title}.items()})
                st.plotly_chart(fig4_with_res, use_container_width=True)
            else:
                st.plotly_chart(fig4, use_container_width=True)

            st.markdown(f"**R² = {r2:.6f}**")

            # Add area fraction
            for i, row in enumerate(result_rows):
                row['Area fraction (%)'] = f'{comp_areas[i]/total_area*100:.1f}'
            result_df = pd.DataFrame(result_rows)
            st.dataframe(result_df, use_container_width=True)

            excel_bytes = to_excel_download({
                'Fit Parameters': result_df,
                'Fit Curve': pd.DataFrame({'Wavelength': wl_fine, 'Total Fit': fit_total}),
                'Residuals': pd.DataFrame({'Wavelength': wl2, 'Residuals': residuals})
            })
            st.download_button("📥 피팅 결과 Excel", data=excel_bytes,
                               file_name="pl_gaussian_fit.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.error(f"피팅 실패: {e}\n초기값을 조정해보세요.")
