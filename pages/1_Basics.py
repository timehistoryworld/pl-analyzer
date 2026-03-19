"""
Module 1 — Basics: Averaging, Peak Analysis, Gaussian Fitting
Gaussian 피팅: 1G / 2G / 3G 별도 탭으로 분리
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import colorsys, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io_utils import load_multiple_files, interpolate_spectra, to_excel_download_dict as to_excel_download
from utils.fitting_utils import detect_peaks, gaussian, multi_gaussian, fwhm_from_gaussian_sigma
from utils.plot_utils import make_figure, style_axes, add_spectrum, COLORS, rainbow_colors
from scipy.optimize import curve_fit

st.set_page_config(page_title="Basics | PL Analyzer", layout="wide", page_icon="📊")

# ── Cached fitting ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Gaussian 피팅 중...")
def _cached_gaussian_fit(wl_b: bytes, inten_b: bytes,
                          p0: tuple, bounds_lo: tuple, bounds_hi: tuple):
    wl    = np.frombuffer(wl_b,    dtype=float)
    inten = np.frombuffer(inten_b, dtype=float)
    popt, pcov = curve_fit(multi_gaussian, wl, inten,
                            p0=list(p0),
                            bounds=(list(bounds_lo), list(bounds_hi)),
                            maxfev=80000)
    return popt, pcov

# ── Helpers ───────────────────────────────────────────────────────────────────
def process(s, normalize, baseline_correct):
    wl    = s['wavelength'].copy()
    inten = s['intensity'].copy()
    if baseline_correct:
        inten -= inten.min()
    if normalize:
        mx = inten.max()
        if mx > 0:
            inten /= mx
    return wl, inten

def comp_fill_color(k, n, alpha=0.12):
    hue = 0.75 - 0.65 * k / max(n - 1, 1)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return f'rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})'

def run_gaussian_fit(wl, inten, n_gauss, p0_list, bounds_lo, bounds_hi,
                     show_residuals, file_label):
    """
    공통 피팅 실행 + 결과 플롯 + 테이블 + Excel 다운로드.
    n_gauss: 1, 2, 3
    """
    try:
        popt, pcov = _cached_gaussian_fit(
            wl.tobytes(), inten.tobytes(),
            tuple(p0_list), tuple(bounds_lo), tuple(bounds_hi)
        )
        perr = np.sqrt(np.diag(pcov))
    except Exception as e:
        st.error(f"피팅 실패: {e}\n초기값을 조정해보세요.")
        return

    wl_fine   = np.linspace(wl.min(), wl.max(), 2000)
    fit_total = multi_gaussian(wl_fine, *popt)
    residuals = inten - multi_gaussian(wl, *popt)
    ss_res    = np.sum(residuals**2)
    ss_tot    = np.sum((inten - inten.mean())**2)
    r2        = 1 - ss_res / (ss_tot + 1e-30)
    chi2_red  = np.sum((residuals**2) / np.maximum(np.abs(inten), 1e-10)) / max(len(inten) - len(p0_list), 1)

    comp_colors = rainbow_colors(n_gauss)

    # ── Build figure ──────────────────────────────────────────────────────────
    if show_residuals:
        fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25],
                            shared_xaxes=True, vertical_spacing=0.04)
        data_row, res_row = 1, 2
    else:
        fig = make_subplots(rows=1, cols=1)
        data_row, res_row = 1, None

    # Data
    fig.add_trace(go.Scatter(
        x=wl, y=inten, name='Data', mode='markers',
        marker=dict(color='rgba(160,170,255,0.5)', size=3)
    ), row=data_row, col=1)

    result_rows, comp_areas = [], []
    total_area = 0

    for k in range(n_gauss):
        amp, cen, sig = popt[3*k], popt[3*k+1], popt[3*k+2]
        amp_e, cen_e, sig_e = perr[3*k], perr[3*k+1], perr[3*k+2]
        fwhm = fwhm_from_gaussian_sigma(sig)
        area = amp * abs(sig) * np.sqrt(2 * np.pi)
        comp_areas.append(area)
        total_area += area

        comp_y = gaussian(wl_fine, amp, cen, sig)
        fig.add_trace(go.Scatter(
            x=wl_fine, y=comp_y,
            name=f'G{k+1}  {cen:.1f} nm',
            line=dict(color=comp_colors[k], width=1.8, dash='dot'),
            fill='tozeroy',
            fillcolor=comp_fill_color(k, n_gauss)
        ), row=data_row, col=1)

        result_rows.append({
            'Component':        f'G{k+1}',
            'Amplitude':        f'{amp:.4g} ± {amp_e:.4g}',
            'Center (nm)':      f'{cen:.3f} ± {cen_e:.3f}',
            'Sigma (nm)':       f'{abs(sig):.3f} ± {sig_e:.3f}',
            'FWHM (nm)':        f'{fwhm:.3f}',
            'Area':             f'{area:.4g}',
        })

    # Total fit
    fig.add_trace(go.Scatter(
        x=wl_fine, y=fit_total, name='Total fit',
        line=dict(color='white', width=2.5, dash='dash')
    ), row=data_row, col=1)

    # Residuals
    if show_residuals:
        fig.add_trace(go.Scatter(
            x=wl, y=residuals, name='Residuals', mode='lines',
            line=dict(color='rgba(232,92,76,0.85)', width=1.2),
            fill='tozeroy', fillcolor='rgba(232,92,76,0.08)'
        ), row=res_row, col=1)
        fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.2)', dash='dash'), row=res_row, col=1)
        fig.update_xaxes(title_text="Wavelength (nm)", row=res_row, col=1)
        fig.update_yaxes(title_text="Residuals", row=res_row, col=1)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(18,18,28,0.95)',
        plot_bgcolor='rgba(18,18,28,0.95)',
        font=dict(family='monospace', size=13),
        height=600 if show_residuals else 460,
        title=f"{n_gauss}-Gaussian Fit — {file_label}  |  R² = {r2:.5f}  |  χ²_red = {chi2_red:.4f}"
    )
    fig.update_xaxes(title_text="Wavelength (nm)", row=data_row, col=1)
    fig.update_yaxes(title_text="Intensity (a.u.)", row=data_row, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # ── Metrics ───────────────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("R²", f"{r2:.6f}")
    chi_ok = chi2_red < 1.3
    m2.metric("χ²_red", f"{chi2_red:.4f}",
               delta="✓ Good" if chi_ok else "⚠ Check",
               delta_color="normal" if chi_ok else "inverse")
    m3.metric("DOF", f"{len(wl) - len(p0_list)}")

    # ── Parameter table ───────────────────────────────────────────────────────
    for i, row in enumerate(result_rows):
        row['Area fraction (%)'] = f'{comp_areas[i] / total_area * 100:.1f}' if total_area > 0 else '—'
    result_df = pd.DataFrame(result_rows)
    st.dataframe(result_df, use_container_width=True)

    # ── Area fraction bar ─────────────────────────────────────────────────────
    if n_gauss > 1:
        fig_bar = go.Figure(go.Bar(
            x=[f'G{k+1}  ({popt[3*k+1]:.1f} nm)' for k in range(n_gauss)],
            y=[comp_areas[k] / total_area * 100 for k in range(n_gauss)],
            marker=dict(color=comp_colors, line=dict(color='white', width=1)),
            text=[f'{comp_areas[k]/total_area*100:.1f}%' for k in range(n_gauss)],
            textposition='outside'
        ))
        fig_bar.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(18,18,28,0.95)',
            plot_bgcolor='rgba(18,18,28,0.95)',
            title='Area Fraction per Component (%)',
            yaxis_title='Area Fraction (%)', height=260,
            font=dict(family='monospace', size=12), margin=dict(t=40, b=40)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Excel export ──────────────────────────────────────────────────────────
    comp_df_dict = {'Fit Parameters': result_df}
    comp_df_dict['Total Fit'] = pd.DataFrame({'Wavelength (nm)': wl_fine, 'Total Fit': fit_total})
    comp_df_dict['Residuals'] = pd.DataFrame({'Wavelength (nm)': wl, 'Residuals': residuals})
    for k in range(n_gauss):
        amp, cen, sig = popt[3*k], popt[3*k+1], popt[3*k+2]
        comp_df_dict[f'G{k+1} ({cen:.1f}nm)'] = pd.DataFrame({
            'Wavelength (nm)': wl_fine,
            f'G{k+1}': gaussian(wl_fine, amp, cen, sig)
        })
    excel_bytes = to_excel_download(comp_df_dict)
    st.download_button(
        f"📥 {n_gauss}-Gaussian 피팅 결과 Excel", data=excel_bytes,
        file_name=f"pl_{n_gauss}gauss_fit.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f'dl_gauss_{n_gauss}'
    )


# ══════════════════════════════════════════════════════════════════════════════
# Page layout
# ══════════════════════════════════════════════════════════════════════════════

st.title("📊 Basics — Peak Analysis & Gaussian Fitting")
st.markdown("스펙트럼 평균화, 피크 감지, 1G / 2G / 3G 피팅")

# ── File upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader("스펙트럼 파일 업로드 (여러 파일 가능)",
                             type=['csv', 'txt', 'xlsx', 'xls'],
                             accept_multiple_files=True)
if not uploaded:
    st.info("CSV / TXT / Excel 파일을 업로드하세요.  첫 번째 열: 파장(nm)  ·  두 번째 열: 강도")
    st.stop()

spectra = load_multiple_files(uploaded)
if not spectra:
    st.error("파일 로딩 실패"); st.stop()

st.success(f"✅ {len(spectra)}개 파일 로드 완료")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 공통 설정")
    normalize        = st.checkbox("최대값 정규화", value=False)
    baseline_correct = st.checkbox("기준선 보정 (최솟값 차감)", value=False)
    show_residuals   = st.checkbox("잔차 패널 표시", value=True)

    st.markdown("---")
    st.subheader("피크 감지")
    prominence = st.slider("Prominence 임계값", 0.01, 0.5, 0.05, 0.01)
    min_dist   = st.slider("최소 피크 간격 (nm)", 1, 100, 15)

colors = rainbow_colors(len(spectra))

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_view, tab_avg, tab_peak, tab_1g, tab_2g, tab_3g = st.tabs([
    "📈 스펙트럼",
    "➕ 평균 & 합산",
    "🔍 피크 감지",
    "〰️ 1-Gaussian",
    "〰️〰️ 2-Gaussian",
    "〰️〰️〰️ 3-Gaussian",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Spectrum viewer
# ══════════════════════════════════════════════════════════════════════════════
with tab_view:
    fig = make_figure(title="Loaded Spectra")
    style_axes(fig, "Wavelength (nm)", "Intensity (a.u.)")
    for i, s in enumerate(spectra):
        wl, inten = process(s, normalize, baseline_correct)
        add_spectrum(fig, wl, inten, name=s['name'], color=colors[i])
    st.plotly_chart(fig, use_container_width=True)

    rows = []
    for s in spectra:
        wl, inten = process(s, normalize, baseline_correct)
        rows.append({
            'File':           s['name'],
            'λ range (nm)':   f"{wl.min():.1f} – {wl.max():.1f}",
            'Points':         len(wl),
            'Peak λ (nm)':    f"{wl[np.argmax(inten)]:.2f}",
            'Max intensity':  f"{inten.max():.4g}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Averaging
# ══════════════════════════════════════════════════════════════════════════════
with tab_avg:
    st.subheader("스펙트럼 평균 & 합산")
    file_names = [s['name'] for s in spectra]
    selected   = st.multiselect("평균할 파일 선택", file_names, default=file_names)
    sel_spectra = [s for s in spectra if s['name'] in selected]

    if not sel_spectra:
        st.warning("파일을 선택하세요")
    else:
        common_wl, interp_list = interpolate_spectra(sel_spectra)
        stack     = np.array(interp_list)
        avg_inten = np.mean(stack, axis=0)
        std_inten = np.std(stack,  axis=0)
        sum_inten = np.sum(stack,  axis=0)

        avg_plot = avg_inten.copy()
        if baseline_correct: avg_plot -= avg_plot.min()
        if normalize and avg_plot.max() > 0: avg_plot /= avg_plot.max()

        fig2 = make_figure(title="Averaged Spectrum")
        style_axes(fig2, "Wavelength (nm)", "Intensity (a.u.)")

        for i, (s, inten) in enumerate(zip(sel_spectra, interp_list)):
            wl_p, inten_p = process({'wavelength': common_wl, 'intensity': inten},
                                     normalize, baseline_correct)
            fig2.add_trace(go.Scatter(
                x=wl_p, y=inten_p, name=s['name'], mode='lines',
                line=dict(color=colors[i % len(colors)], width=1, dash='dot'), opacity=0.4
            ))
        fig2.add_trace(go.Scatter(
            x=common_wl, y=avg_plot, name='Average',
            mode='lines', line=dict(color='#FFFFFF', width=2.5)
        ))
        fig2.add_trace(go.Scatter(
            x=np.concatenate([common_wl, common_wl[::-1]]),
            y=np.concatenate([avg_plot + std_inten, (avg_plot - std_inten)[::-1]]),
            fill='toself', fillcolor='rgba(255,255,255,0.07)',
            line=dict(color='rgba(255,255,255,0)'), name='±1 SD'
        ))
        st.plotly_chart(fig2, use_container_width=True)

        df_avg = pd.DataFrame({'Wavelength (nm)': common_wl,
                                'Average': avg_inten, 'Std Dev': std_inten, 'Sum': sum_inten})
        st.download_button(
            "📥 평균 데이터 Excel", data=to_excel_download({'Average': df_avg}),
            file_name="pl_average.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Peak detection
# ══════════════════════════════════════════════════════════════════════════════
with tab_peak:
    st.subheader("피크 자동 감지")
    sel_file = st.selectbox("파일 선택", [s['name'] for s in spectra], key='peak_sel')
    s_sel    = next(s for s in spectra if s['name'] == sel_file)
    wl, inten = process(s_sel, normalize, baseline_correct)

    peaks_idx, peaks_wl, peaks_inten = detect_peaks(wl, inten, prominence, min_dist)

    show_deriv = st.checkbox("미분 스펙트럼 표시", value=False)
    order      = st.radio("미분 차수", [1, 2], horizontal=True) if show_deriv else 1

    fig3 = make_figure(title=f"Peak Detection — {sel_file}")
    style_axes(fig3, "Wavelength (nm)", "Intensity (a.u.)")

    if show_deriv:
        deriv = np.gradient(inten, wl)
        if order == 2: deriv = np.gradient(deriv, wl)
        deriv_norm = deriv / (np.max(np.abs(deriv)) + 1e-30)
        fig3.add_trace(go.Scatter(x=wl, y=deriv_norm, name=f'{order}차 미분',
                                   line=dict(color='rgba(232,184,76,0.7)', dash='dot', width=1.5)))

    add_spectrum(fig3, wl, inten, name=sel_file, color=COLORS[0])
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
            'Peak #':             range(1, len(peaks_idx)+1),
            'Wavelength (nm)':    peaks_wl.round(2),
            'Intensity':          peaks_inten.round(4),
            'Rel. Intensity (%)': (peaks_inten / peaks_inten.max() * 100).round(1)
        })
        st.dataframe(peak_df, use_container_width=True)
        st.download_button(
            "📥 피크 데이터 Excel", data=to_excel_download({'Peaks': peak_df}),
            file_name="pl_peaks.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ══════════════════════════════════════════════════════════════════════════════
# TABs 4-6: 1G / 2G / 3G Gaussian fitting
# ══════════════════════════════════════════════════════════════════════════════

def gaussian_fitting_tab(tab, n_gauss: int, tab_key: str):
    """1G, 2G, 3G 공통 UI 렌더링."""
    with tab:
        st.subheader(f"{n_gauss}-Gaussian 피팅")
        st.markdown(f"스펙트럼을 **{n_gauss}개의 Gaussian 성분**으로 분해합니다.")

        # File selector
        sel = st.selectbox("파일 선택", [s['name'] for s in spectra], key=f'gsel_{tab_key}')
        s_g = next(s for s in spectra if s['name'] == sel)
        wl_g, inten_g = process(s_g, normalize, baseline_correct)

        # Fitting range
        st.markdown("**📐 피팅 범위**")
        rc1, rc2 = st.columns(2)
        wl_lo = rc1.number_input("시작 λ (nm)", value=float(wl_g.min()),
                                  format="%.2f", key=f'wllo_{tab_key}')
        wl_hi = rc2.number_input("끝 λ (nm)",   value=float(wl_g.max()),
                                  format="%.2f", key=f'wlhi_{tab_key}')
        mask = (wl_g >= wl_lo) & (wl_g <= wl_hi)
        wl_fit, inten_fit = wl_g[mask], inten_g[mask]

        if len(wl_fit) < n_gauss * 3 + 2:
            st.warning("피팅 범위 내 데이터 포인트가 너무 적습니다."); return

        # ── Initial value table ───────────────────────────────────────────────
        st.markdown("**🎛 초기값 설정**")

        # Auto-detect peaks as defaults
        from utils.fitting_utils import detect_peaks as _dp
        auto_idx, auto_wl, auto_int = _dp(wl_fit, inten_fit, prominence=0.05)

        init_rows = []
        for k in range(n_gauss):
            if k < len(auto_wl):
                cen_def = float(auto_wl[k])
                amp_def = float(auto_int[k])
            else:
                cen_def = float(wl_fit.mean()) + k * 20
                amp_def = float(inten_fit.max()) * (0.5 ** k)
            init_rows.append({
                f'G{k+1} — 파라미터': f'G{k+1}',
                'Amplitude':          round(amp_def, 4),
                'Center (nm)':        round(cen_def, 2),
                'Sigma (nm)':         15.0,
            })

        init_df  = pd.DataFrame(init_rows)
        edited   = st.data_editor(init_df, use_container_width=True, key=f'init_{tab_key}')

        # Constraint options
        with st.expander("🔒 제약 조건 (선택)", expanded=False):
            fix_amp    = st.checkbox("Amplitude 하한 = 0 고정",   value=True,  key=f'famp_{tab_key}')
            center_tol = st.slider("Center 허용 범위 ±(nm)", 0.0, 200.0, 50.0, 1.0,
                                    key=f'ctol_{tab_key}')
            sig_min    = st.number_input("Sigma 최솟값 (nm)", value=0.5, min_value=0.01,
                                          format="%.2f", key=f'smin_{tab_key}')
            sig_max    = st.number_input("Sigma 최댓값 (nm)", value=200.0, min_value=1.0,
                                          format="%.1f", key=f'smax_{tab_key}')

        # Build p0 / bounds
        p0, lo, hi = [], [], []
        for k in range(n_gauss):
            row  = edited.iloc[k]
            amp0 = float(row['Amplitude'])
            cen0 = float(row['Center (nm)'])
            sig0 = float(row['Sigma (nm)'])
            p0  += [amp0, cen0, sig0]
            lo  += [0.0 if fix_amp else -abs(amp0)*10,
                    max(wl_lo, cen0 - center_tol),
                    sig_min]
            hi  += [abs(amp0) * 20,
                    min(wl_hi, cen0 + center_tol),
                    sig_max]

        # Preview plot before fitting
        fig_pre = make_figure(title=f"Data preview — {sel}")
        style_axes(fig_pre, "Wavelength (nm)", "Intensity (a.u.)")
        fig_pre.add_trace(go.Scatter(x=wl_fit, y=inten_fit, name='Data', mode='markers',
                                      marker=dict(color='rgba(160,170,255,0.5)', size=3)))
        # Show initial Gaussian guess
        wl_pre = np.linspace(wl_fit.min(), wl_fit.max(), 1000)
        g_colors_pre = rainbow_colors(n_gauss)
        for k in range(n_gauss):
            row  = edited.iloc[k]
            amp0 = float(row['Amplitude'])
            cen0 = float(row['Center (nm)'])
            sig0 = float(row['Sigma (nm)'])
            g_pre = gaussian(wl_pre, amp0, cen0, sig0)
            fig_pre.add_trace(go.Scatter(
                x=wl_pre, y=g_pre,
                name=f'G{k+1} initial ({cen0:.1f} nm)',
                line=dict(color=g_colors_pre[k], width=1.5, dash='dot'),
                fill='tozeroy', fillcolor=comp_fill_color(k, n_gauss, 0.08)
            ))
        total_pre = sum(
            gaussian(wl_pre,
                     float(edited.iloc[k]['Amplitude']),
                     float(edited.iloc[k]['Center (nm)']),
                     float(edited.iloc[k]['Sigma (nm)']))
            for k in range(n_gauss)
        )
        fig_pre.add_trace(go.Scatter(x=wl_pre, y=total_pre, name='Initial sum',
                                      line=dict(color='rgba(255,255,255,0.4)', width=1.5, dash='dash')))
        fig_pre.update_layout(height=320)
        st.plotly_chart(fig_pre, use_container_width=True)

        st.markdown("초기값이 맞으면 피팅을 실행하세요.")

        if st.button("🔄 피팅 실행", type="primary", key=f'fit_{tab_key}'):
            run_gaussian_fit(
                wl_fit, inten_fit,
                n_gauss, p0, lo, hi,
                show_residuals, sel
            )

# Render each fitting tab
gaussian_fitting_tab(tab_1g, 1, '1g')
gaussian_fitting_tab(tab_2g, 2, '2g')
gaussian_fitting_tab(tab_3g, 3, '3g')
