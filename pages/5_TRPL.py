"""
Module 5 — TRPL: Time-Resolved PL with IRF Deconvolution
- IRF 파일 업로드 + offset/shift/scale 조정
- Mono / Bi / Tri / Quad exponential reconvolution fitting
- Residual kinetics & reduced chi-squared
- Lifetime distribution (Tikhonov-regularized NNLS)
- Amplitude/Intensity-weighted lifetime
- Multi-file comparison
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit, nnls
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io_utils import load_spectrum_file, load_multiple_files, to_excel_download
from utils.fitting_utils import amplitude_weighted_lifetime, intensity_weighted_lifetime
from utils.plot_utils import make_figure, style_axes, COLORS, rainbow_colors

st.set_page_config(page_title="TRPL | PL Analyzer", layout="wide", page_icon="⏱")
st.title("⏱ TRPL — Time-Resolved PL & IRF Deconvolution")
st.markdown("IRF reconvolution 피팅 · 최대 4-지수 감쇠 · Lifetime distribution · χ² 실시간 표시")

# ══════════════════════════════════════════════════════════════════════════════
# Core functions
# ══════════════════════════════════════════════════════════════════════════════

def reconvolve_model(t, irf_t, irf_y, params_exp, n_exp, t_shift, scale):
    """
    Reconvolution: (IRF shifted) ⊗ (multi-exp decay) * scale + baseline
    params_exp: [A1,tau1, A2,tau2, ..., An,taun, y0]
    """
    # Shift IRF and interpolate onto data grid
    irf_fn = interp1d(irf_t - t_shift, irf_y, bounds_error=False, fill_value=0.0)
    irf_on_t = np.clip(irf_fn(t), 0.0, None)
    norm = irf_on_t.sum()
    if norm > 0:
        irf_on_t /= norm

    # Build decay kernel on [0, t_range]
    dt = (t[-1] - t[0]) / (len(t) - 1) if len(t) > 1 else 1.0
    t_kernel = np.arange(0, (t[-1] - t[0]) * 2 + dt, dt)
    decay_k = np.zeros(len(t_kernel))
    for k in range(n_exp):
        A, tau = params_exp[2*k], params_exp[2*k+1]
        decay_k += A * np.exp(-t_kernel / max(tau, 1e-10))

    # Convolve and trim
    conv = fftconvolve(irf_on_t, decay_k, mode='full')[:len(t)]
    peak = conv.max()
    if peak > 0:
        conv /= peak
    return scale * conv + params_exp[-1]  # + baseline y0


def make_reconv_func(irf_t, irf_y, n_exp, fit_shift, fit_scale, fixed_shift, fixed_scale):
    """Factory: returns callable for curve_fit with variable signature."""
    def model(t, *p):
        # p layout: [A1,tau1,...,An,taun, y0,  (t_shift,) (scale,)]
        n_core = 2*n_exp + 1
        core   = list(p[:n_core])
        idx    = n_core
        t_sh   = p[idx] if fit_shift else fixed_shift; idx += (1 if fit_shift else 0)
        sc     = p[idx] if fit_scale else fixed_scale
        return reconvolve_model(t, irf_t, irf_y, core, n_exp, t_sh, sc)
    return model


def tail_model(t, *p):
    """Pure multi-exp (no IRF). p = [A1,tau1,...,An,taun,y0]"""
    n = (len(p) - 1) // 2
    y = np.zeros_like(t, dtype=float)
    for k in range(n):
        y += p[2*k] * np.exp(-t / max(p[2*k+1], 1e-10))
    return y + p[-1]


def reduced_chi2(data, model_vals, n_params):
    sigma2 = np.maximum(np.abs(data), 1.0)
    chi2   = np.sum((data - model_vals)**2 / sigma2)
    dof    = max(len(data) - n_params, 1)
    return chi2 / dof


def lifetime_distribution(t, I, irf_t=None, irf_y=None,
                           n_tau=150, tau_min=0.01, tau_max=1000.0, alpha=1e-3):
    """
    Tikhonov-regularized NNLS lifetime distribution.
    I(t) ≈ K @ p,  K[i,j] = exp(-t[i]/tau[j])  (or IRF-convolved).
    Returns tau_grid, p (unnormalized amplitudes).
    """
    tau_grid = np.logspace(np.log10(tau_min), np.log10(tau_max), n_tau)

    # Build kernel
    K = np.exp(-np.outer(t, 1.0 / tau_grid))

    if irf_t is not None and irf_y is not None:
        irf_fn  = interp1d(irf_t, irf_y, bounds_error=False, fill_value=0.0)
        irf_col = np.clip(irf_fn(t), 0, None)
        irf_col /= (irf_col.sum() + 1e-30)
        for j in range(n_tau):
            col_conv = fftconvolve(irf_col, K[:, j], mode='full')[:len(t)]
            K[:, j]  = col_conv

    # Normalize data
    d = I / (I.max() + 1e-30)

    # Tikhonov: second-difference regularization
    L = np.zeros((n_tau - 2, n_tau))
    for i in range(n_tau - 2):
        L[i, i] = 1; L[i, i+1] = -2; L[i, i+2] = 1

    K_aug = np.vstack([K, np.sqrt(alpha) * L])
    d_aug = np.concatenate([d, np.zeros(n_tau - 2)])

    p, _ = nnls(K_aug, d_aug)
    return tau_grid, p

# ══════════════════════════════════════════════════════════════════════════════
# File upload
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("📁 데이터 업로드")
col_data, col_irf_up = st.columns(2)
with col_data:
    trpl_files = st.file_uploader(
        "TRPL 파일(들) — [time, intensity]",
        type=['csv','txt','xlsx','xls'], accept_multiple_files=True, key='trpl_data')
with col_irf_up:
    irf_file = st.file_uploader(
        "IRF 파일 (선택) — [time, intensity]",
        type=['csv','txt','xlsx','xls'], key='irf_file',
        help="IRF 없으면 tail-fit으로 진행됩니다")

if not trpl_files:
    st.info("TRPL 데이터를 업로드하세요. 형식: [time, intensity]")
    st.stop()

spectra = load_multiple_files(trpl_files)
if not spectra:
    st.error("TRPL 파일 로딩 실패"); st.stop()

irf_loaded = False
irf_t_raw = irf_y_raw = None
if irf_file:
    irf_t_raw, irf_y_raw = load_spectrum_file(irf_file)
    irf_loaded = irf_t_raw is not None
    if not irf_loaded:
        st.warning("IRF 로딩 실패 — tail-fit 모드로 진행합니다")

mode_str = "IRF Reconvolution 모드" if irf_loaded else "Tail-fit 모드 (IRF 없음)"
st.success(f"✅ {len(spectra)}개 TRPL 파일 | {mode_str}")

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ 전역 설정")
    time_unit  = st.selectbox("시간 단위", ["ns", "ps", "µs", "ms"])
    log_scale  = st.checkbox("Log y축", value=True)
    normalize_ = st.checkbox("미리보기 정규화", value=True)

    st.markdown("---")
    st.subheader("지수 성분 수")
    n_exp = st.radio("N-exponential", [1, 2, 3, 4], index=1, horizontal=True)

    st.markdown("---")
    st.subheader("IRF 조정")
    irf_t_shift  = st.number_input(f"IRF shift ({time_unit})", value=0.0,
                                    step=0.01, format="%.4f",
                                    help="IRF를 시간축에서 이동 (양수 = 오른쪽)")
    irf_scale_   = st.slider("IRF amplitude scale", 0.1, 5.0, 1.0, 0.05)
    irf_offset_  = st.number_input("IRF baseline offset", value=0.0,
                                    step=0.0001, format="%.6f")
    fit_t_shift  = st.checkbox("t_shift 피팅 파라미터 포함", value=True,
                                disabled=not irf_loaded)
    fit_scale_p  = st.checkbox("scale 피팅 파라미터 포함",   value=True,
                                disabled=not irf_loaded)

    st.markdown("---")
    st.subheader("피팅 시간 범위")
    t_start_fit = st.number_input(f"t_start ({time_unit})", value=0.0, format="%.4f")
    t_end_fit   = st.number_input(f"t_end ({time_unit})",   value=9999.0, format="%.4f")

    st.markdown("---")
    st.subheader("Lifetime Distribution")
    n_tau_pts   = st.slider("τ grid 포인트", 50, 500, 150, 50)
    alpha_reg   = st.select_slider(
        "Regularization α",
        options=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        value=1e-3, format_func=lambda x: f"{x:.0e}")

# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════

colors_all = rainbow_colors(max(len(spectra), 1))
tab_ov, tab_irf_tab, tab_fit, tab_dist, tab_cmp = st.tabs([
    "📈 미리보기", "🔍 IRF 확인", "🔄 피팅", "📊 Lifetime Distribution", "📋 다중 비교"
])

# ── TAB 1: Overview ───────────────────────────────────────────────────────────
with tab_ov:
    fig_ov = make_figure(title="TRPL Decay Curves")
    style_axes(fig_ov, f"Time ({time_unit})", "Intensity (a.u.)")
    if log_scale: fig_ov.update_yaxes(type='log')

    for i, s in enumerate(spectra):
        t = s['wavelength']; I = s['intensity'].copy()
        mask = (t >= t_start_fit) & (t <= t_end_fit)
        t_p, I_p = t[mask], I[mask]
        if normalize_ and I_p.max() > 0: I_p /= I_p.max()
        fig_ov.add_trace(go.Scatter(x=t_p, y=I_p, name=s['name'],
                                     line=dict(color=colors_all[i], width=2)))

    if irf_loaded:
        irf_yp = np.clip((irf_y_raw - irf_offset_) * irf_scale_, 0, None)
        if normalize_ and irf_yp.max() > 0: irf_yp /= irf_yp.max()
        fig_ov.add_trace(go.Scatter(x=irf_t_raw + irf_t_shift, y=irf_yp,
                                     name='IRF',
                                     line=dict(color='rgba(255,220,80,0.8)',
                                               width=1.5, dash='dot')))
    st.plotly_chart(fig_ov, use_container_width=True)

    rows = []
    for s in spectra:
        t, I = s['wavelength'], s['intensity']
        mask = (t >= t_start_fit) & (t <= t_end_fit)
        rows.append({'File': s['name'],
                     f't range ({time_unit})': f"{t[mask].min():.3f}–{t[mask].max():.3f}",
                     'Points': int(mask.sum()), 'Peak counts': f"{I[mask].max():.0f}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ── TAB 2: IRF inspect ────────────────────────────────────────────────────────
with tab_irf_tab:
    if not irf_loaded:
        st.info("IRF 파일을 업로드하면 여기서 시각적으로 확인하고 조정할 수 있습니다.\n\n"
                "IRF 없이도 tail-fit으로 lifetime을 추출할 수 있습니다.")
    else:
        irf_y_adj = np.clip((irf_y_raw - irf_offset_) * irf_scale_, 0, None)
        irf_t_adj = irf_t_raw + irf_t_shift

        fig_irf = make_figure(title="IRF Inspection & Timing Alignment")
        style_axes(fig_irf, f"Time ({time_unit})", "Intensity (normalized)")

        # Raw IRF
        fig_irf.add_trace(go.Scatter(
            x=irf_t_raw, y=irf_y_raw / (irf_y_raw.max()+1e-30),
            name='IRF raw', line=dict(color='rgba(200,200,150,0.35)', width=1.2, dash='dot')))
        # Adjusted IRF
        fig_irf.add_trace(go.Scatter(
            x=irf_t_adj, y=irf_y_adj / (irf_y_adj.max()+1e-30),
            name='IRF adjusted', line=dict(color='rgba(255,220,80,0.9)', width=2.5)))
        # First TRPL for alignment
        s0 = spectra[0]; I0 = s0['intensity'].copy()
        if I0.max() > 0: I0 /= I0.max()
        fig_irf.add_trace(go.Scatter(
            x=s0['wavelength'], y=I0,
            name=f'TRPL: {s0["name"]}',
            line=dict(color=colors_all[0], width=1.5, dash='dash')))

        st.plotly_chart(fig_irf, use_container_width=True)

        # IRF stats
        irf_peak_t = irf_t_adj[np.argmax(irf_y_adj)]
        above_half = irf_t_adj[irf_y_adj >= irf_y_adj.max()/2]
        irf_fwhm   = float(above_half[-1]-above_half[0]) if len(above_half)>1 else float('nan')
        c1,c2,c3 = st.columns(3)
        c1.metric(f"IRF peak time ({time_unit})", f"{irf_peak_t:.4f}")
        c2.metric(f"IRF FWHM ({time_unit})", f"{irf_fwhm:.4f}")
        c3.metric("IRF peak counts", f"{irf_y_adj.max():.0f}")

        st.markdown("**💡 조정 팁:**")
        st.markdown("- `IRF shift`: TRPL의 rise 부분과 IRF peak가 겹치도록 조정  \n"
                    "- `IRF baseline offset`: IRF의 background를 0으로 맞춤  \n"
                    "- `IRF scale`: 진폭은 자동 정규화되므로 보통 1.0 유지  \n"
                    "- `t_shift 피팅 포함`: 체크하면 피팅 중 미세 조정 자동으로 수행")

# ── TAB 3: Fitting ────────────────────────────────────────────────────────────
with tab_fit:
    st.subheader(f"{'IRF Reconvolution' if irf_loaded else 'Tail'} Fitting — {n_exp}-Exponential")

    sel_name = st.selectbox("피팅할 파일", [s['name'] for s in spectra], key='fit_sel')
    s_sel    = next(s for s in spectra if s['name'] == sel_name)
    t_full   = s_sel['wavelength']
    I_full   = s_sel['intensity'].copy()
    mask_fit = (t_full >= t_start_fit) & (t_full <= t_end_fit)
    t_data   = t_full[mask_fit]
    I_data   = I_full[mask_fit]

    if len(t_data) < n_exp*2 + 3:
        st.error("피팅 범위 내 데이터 포인트가 너무 적습니다. t_start/t_end를 조정하세요.")
        st.stop()

    # ── Initial values ────────────────────────────────────────────────────────
    st.markdown("**초기값 설정**")
    t_range = float(t_data[-1] - t_data[0])
    p0_exp, lo_exp, hi_exp = [], [], []
    cols_init = st.columns(n_exp)
    for k in range(n_exp):
        with cols_init[k]:
            st.markdown(f"**성분 {k+1}**")
            A0   = st.number_input(f"A{k+1}",     value=round(1.0/(k+1),4),
                                    key=f'A_init_{k}', format="%.5f", min_value=1e-5)
            tau0 = st.number_input(f"τ{k+1} ({time_unit})",
                                    value=round(t_range / (2**(n_exp-k)), 4),
                                    key=f'tau_init_{k}', format="%.4f", min_value=0.001)
            p0_exp  += [A0, tau0]
            lo_exp  += [0.0,     0.001]
            hi_exp  += [1e7, t_range*20]

    cr1, cr2, cr3 = st.columns(3)
    y0_init    = cr1.number_input("y₀ (baseline)",  value=0.0,  format="%.6f", key='y0init')
    sh_init    = cr2.number_input(f"t_shift init ({time_unit})",
                                   value=float(irf_t_shift), format="%.4f", key='shinit') if irf_loaded else 0.0
    sc_init    = cr3.number_input("scale init",     value=float(irf_scale_),
                                   format="%.4f", min_value=0.001, key='scinit') if irf_loaded else 1.0

    if st.button("🔄 피팅 실행", type="primary", key='run_fit'):
        with st.spinner("피팅 중..."):
            try:
                if irf_loaded:
                    irf_y_use = np.clip((irf_y_raw - irf_offset_) * 1.0, 0, None)
                    irf_t_use = irf_t_raw   # shift is handled inside model

                    model_fn = make_reconv_func(
                        irf_t_use, irf_y_use, n_exp,
                        fit_t_shift, fit_scale_p,
                        fixed_shift=irf_t_shift, fixed_scale=irf_scale_)

                    p0_full = p0_exp + [y0_init]
                    lo_full = lo_exp + [-I_data.max()*0.5]
                    hi_full = hi_exp + [I_data.max()]

                    if fit_t_shift:
                        p0_full += [sh_init];  lo_full += [-t_range*0.5]; hi_full += [t_range*0.5]
                    if fit_scale_p:
                        p0_full += [sc_init];  lo_full += [0.001]; hi_full += [I_data.max()*200]

                    n_params_total = len(p0_full)
                    popt, pcov = curve_fit(model_fn, t_data, I_data,
                                           p0=p0_full, bounds=(lo_full, hi_full),
                                           maxfev=300000, method='trf')
                    I_fitted = model_fn(t_data, *popt)

                else:
                    n_tail = n_exp
                    p0_tail = p0_exp + [y0_init]
                    lo_tail = lo_exp + [-I_data.max()*0.1]
                    hi_tail = hi_exp + [I_data.max()]
                    n_params_total = len(p0_tail)
                    popt, pcov = curve_fit(tail_model, t_data, I_data,
                                           p0=p0_tail, bounds=(lo_tail, hi_tail),
                                           maxfev=300000)
                    I_fitted = tail_model(t_data, *popt)

                perr = np.sqrt(np.diag(np.clip(pcov, 0, None)))

                # Statistics
                residuals  = I_data - I_fitted
                chi2_red   = reduced_chi2(I_data, I_fitted, n_params_total)
                r2         = 1 - np.sum(residuals**2) / (np.sum((I_data-I_data.mean())**2)+1e-30)
                sigma      = np.sqrt(np.maximum(I_data, 1.0))
                w_res      = residuals / sigma

                # ── 3-panel figure ──────────────────────────────────────────
                fig_fit = make_subplots(
                    rows=3, cols=1,
                    row_heights=[0.58, 0.21, 0.21],
                    shared_xaxes=True,
                    vertical_spacing=0.025,
                    subplot_titles=[
                        f"Data & Reconvolution Fit",
                        "Residuals  (I − F)",
                        "Weighted Residuals  (I − F) / √I   [±2σ dashed]"
                    ]
                )

                # Panel 1
                fig_fit.add_trace(go.Scatter(
                    x=t_data, y=I_data, name='Data', mode='markers',
                    marker=dict(color='rgba(140,160,255,0.45)', size=3)), row=1, col=1)
                fig_fit.add_trace(go.Scatter(
                    x=t_data, y=I_fitted, name='Fit',
                    line=dict(color='white', width=2.5)), row=1, col=1)
                if irf_loaded:
                    irf_plot = np.clip((irf_y_raw-irf_offset_)*irf_scale_, 0, None)
                    t_sh_used = popt[2*n_exp+1] if (irf_loaded and fit_t_shift) else irf_t_shift
                    irf_plot_norm = irf_plot / (irf_plot.max()+1e-30) * I_data.max()
                    fig_fit.add_trace(go.Scatter(
                        x=irf_t_raw + t_sh_used, y=irf_plot_norm, name='IRF (scaled)',
                        line=dict(color='rgba(255,220,80,0.55)', width=1.2, dash='dot')), row=1, col=1)

                # Panel 2: residuals
                fig_fit.add_trace(go.Scatter(
                    x=t_data, y=residuals, name='Residuals', mode='lines',
                    line=dict(color='rgba(232,92,76,0.85)', width=1.2),
                    fill='tozeroy', fillcolor='rgba(232,92,76,0.08)'), row=2, col=1)
                fig_fit.add_hline(y=0, line=dict(color='rgba(255,255,255,0.2)', dash='dash'), row=2, col=1)

                # Panel 3: weighted residuals
                fig_fit.add_trace(go.Scatter(
                    x=t_data, y=w_res, name='Weighted res.', mode='lines',
                    line=dict(color='rgba(76,210,185,0.85)', width=1.2),
                    fill='tozeroy', fillcolor='rgba(76,210,185,0.07)'), row=3, col=1)
                fig_fit.add_hline(y=0,  line=dict(color='rgba(255,255,255,0.2)', dash='dash'), row=3, col=1)
                fig_fit.add_hline(y=2,  line=dict(color='rgba(255,210,80,0.35)', dash='dot'),  row=3, col=1)
                fig_fit.add_hline(y=-2, line=dict(color='rgba(255,210,80,0.35)', dash='dot'),  row=3, col=1)

                fig_fit.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(18,18,28,0.95)',
                    plot_bgcolor='rgba(18,18,28,0.95)',
                    font=dict(family='monospace', size=12),
                    height=720,
                    title=(f"{'Reconvolution' if irf_loaded else 'Tail'} Fit — {sel_name}"
                           f"  |  χ²_red = {chi2_red:.4f}  |  R² = {r2:.5f}"),
                    showlegend=True
                )
                if log_scale:
                    fig_fit.update_yaxes(type='log', row=1, col=1)
                for row_i, ytitle in enumerate(["Intensity","Residuals","(I−F)/√I"], 1):
                    fig_fit.update_yaxes(title_text=ytitle, row=row_i, col=1)
                fig_fit.update_xaxes(title_text=f"Time ({time_unit})", row=3, col=1)

                st.plotly_chart(fig_fit, use_container_width=True)

                # ── Metrics row ───────────────────────────────────────────────
                mc1, mc2, mc3, mc4 = st.columns(4)
                chi2_ok = chi2_red < 1.3
                mc1.metric("χ²_red", f"{chi2_red:.4f}",
                            delta="✓ Good" if chi2_ok else "⚠ Check",
                            delta_color="normal" if chi2_ok else "inverse")
                mc2.metric("R²",  f"{r2:.6f}")
                mc3.metric("DOF", f"{len(I_data) - n_params_total}")
                mc4.metric("N data pts", f"{len(I_data)}")

                # ── Parameter table ────────────────────────────────────────────
                st.markdown("---")
                st.subheader("📋 피팅 파라미터")

                amps, taus, rows_param = [], [], []
                A_sum = sum(popt[2*k] for k in range(n_exp))
                At_sum = sum(popt[2*k]*popt[2*k+1] for k in range(n_exp))

                for k in range(n_exp):
                    Ak, tauk = popt[2*k], popt[2*k+1]
                    Ak_e = perr[2*k] if 2*k < len(perr) else float('nan')
                    tk_e = perr[2*k+1] if 2*k+1 < len(perr) else float('nan')
                    amps.append(Ak); taus.append(tauk)
                    rows_param.append({
                        'Component':           f'#{k+1}',
                        f'A{k+1}':             f'{Ak:.6f} ± {Ak_e:.6f}',
                        f'τ{k+1} ({time_unit})': f'{tauk:.5f} ± {tk_e:.5f}',
                        'Amp. fraction (%)':   f'{Ak/A_sum*100:.2f}' if A_sum>0 else '—',
                        'Int. fraction (%)':   f'{Ak*tauk/At_sum*100:.2f}' if At_sum>0 else '—',
                    })

                y0_p = popt[2*n_exp]
                y0_e = perr[2*n_exp] if 2*n_exp < len(perr) else float('nan')
                rows_param.append({'Component': 'y₀ (baseline)',
                                    f'A1': f'{y0_p:.6f} ± {y0_e:.6f}',
                                    f'τ1 ({time_unit})': '—',
                                    'Amp. fraction (%)': '—', 'Int. fraction (%)': '—'})
                if irf_loaded:
                    idx_sh = 2*n_exp+1; idx_sc = 2*n_exp+2
                    if fit_t_shift and idx_sh < len(popt):
                        rows_param.append({'Component': 't_shift (fitted)',
                                            f'A1': f'{popt[idx_sh]:.5f} ± {perr[idx_sh] if idx_sh<len(perr) else float("nan"):.5f}',
                                            f'τ1 ({time_unit})':'—','Amp. fraction (%)':'—','Int. fraction (%)':'—'})
                        idx_sc += 0  # already correct
                    if fit_scale_p and idx_sc < len(popt):
                        rows_param.append({'Component': 'scale (fitted)',
                                            f'A1': f'{popt[idx_sc]:.5f} ± {perr[idx_sc] if idx_sc<len(perr) else float("nan"):.5f}',
                                            f'τ1 ({time_unit})':'—','Amp. fraction (%)':'—','Int. fraction (%)':'—'})

                st.dataframe(pd.DataFrame(rows_param), use_container_width=True)

                # ── Weighted lifetimes ──────────────────────────────────────
                tau_amp_w = amplitude_weighted_lifetime(amps, taus)
                tau_int_w = intensity_weighted_lifetime(amps, taus)
                wc1, wc2 = st.columns(2)
                wc1.metric(f"⟨τ⟩_amp ({time_unit})", f"{tau_amp_w:.4f}",
                            help="Σ(Ai·τi) / Σ(Ai)")
                wc2.metric(f"⟨τ⟩_int ({time_unit})", f"{tau_int_w:.4f}",
                            help="Σ(Ai·τi²) / Σ(Ai·τi)")

                # ── Component bar ───────────────────────────────────────────
                comp_colors = rainbow_colors(n_exp)
                fig_bar = go.Figure(go.Bar(
                    x=[f'#{k+1}: τ={taus[k]:.3f} {time_unit}' for k in range(n_exp)],
                    y=[amps[k]/A_sum*100 for k in range(n_exp)],
                    marker=dict(color=comp_colors, line=dict(color='white', width=1)),
                    text=[f'{amps[k]/A_sum*100:.1f}%' for k in range(n_exp)],
                    textposition='outside'
                ))
                fig_bar.update_layout(
                    template='plotly_dark', paper_bgcolor='rgba(18,18,28,0.95)',
                    plot_bgcolor='rgba(18,18,28,0.95)', height=260,
                    title='Amplitude Fraction per Component',
                    yaxis_title='Amplitude Fraction (%)',
                    font=dict(family='monospace', size=12), margin=dict(t=40)
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # ── Save to session state ───────────────────────────────────
                st.session_state['trpl_fit'] = dict(
                    t=t_data, I=I_data, I_fit=I_fitted,
                    residuals=residuals, w_res=w_res,
                    chi2_red=chi2_red, r2=r2,
                    amps=amps, taus=taus,
                    tau_amp=tau_amp_w, tau_int=tau_int_w,
                    name=sel_name
                )

                # ── Excel export ────────────────────────────────────────────
                df_p  = pd.DataFrame(rows_param)
                df_cv = pd.DataFrame({
                    f'Time ({time_unit})':  t_data,
                    'Data':                 I_data,
                    'Fit':                  I_fitted,
                    'Residuals':            residuals,
                    'Weighted Residuals':   w_res,
                })
                excel_bytes = to_excel_download({'Parameters': df_p, 'Curves': df_cv})
                st.download_button("📥 피팅 결과 Excel", data=excel_bytes,
                                   file_name=f"trpl_fit_{sel_name}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            except Exception as e:
                import traceback
                st.error(f"피팅 실패: {e}")
                with st.expander("상세 오류 보기"):
                    st.code(traceback.format_exc())
                st.info("💡 초기값을 조정하거나 t_start/t_end를 좁혀보세요. "
                        "IRF shift가 크게 어긋난 경우 Tab 2에서 먼저 정렬하세요.")

# ── TAB 4: Lifetime Distribution ──────────────────────────────────────────────
with tab_dist:
    st.subheader("📊 Lifetime Distribution — Tikhonov Regularized NNLS")
    st.markdown(r"""
    $$I(t) \approx \int p(\tau)\, e^{-t/\tau}\, d\tau \quad (\text{or IRF} \otimes \int p(\tau)\,e^{-t/\tau}d\tau)$$
    Regularization: $\min_{p \geq 0} \|Kp - I\|^2 + \alpha \|L p\|^2$, where $L$ = 2nd-difference matrix.
    """)

    sel_dist = st.selectbox("파일 선택", [s['name'] for s in spectra], key='dist_sel')
    sd       = next(s for s in spectra if s['name'] == sel_dist)
    t_d      = sd['wavelength']
    I_d      = sd['intensity'].copy()
    mask_d   = (t_d >= t_start_fit) & (t_d <= t_end_fit)
    t_d, I_d = t_d[mask_d], I_d[mask_d]

    cd1, cd2 = st.columns(2)
    dt_data   = float(t_d[1]-t_d[0]) if len(t_d) > 1 else 0.01
    tau_lo    = cd1.number_input(f"τ_min ({time_unit})", value=round(dt_data*0.5,5),
                                  min_value=1e-5, format="%.5f", key='taumin')
    tau_hi    = cd2.number_input(f"τ_max ({time_unit})", value=round(float(t_d[-1]-t_d[0])*3, 3),
                                  min_value=0.01, format="%.3f", key='taumax')

    if st.button("🔄 Distribution 계산", type="primary", key='dist_run'):
        with st.spinner("Lifetime distribution 계산 중 (수 초 소요)..."):
            try:
                irf_t_use = (irf_t_raw + irf_t_shift) if irf_loaded else None
                irf_y_use = np.clip((irf_y_raw - irf_offset_)*irf_scale_, 0, None) if irf_loaded else None

                tau_grid, p_tau = lifetime_distribution(
                    t_d, I_d,
                    irf_t=irf_t_use, irf_y=irf_y_use,
                    n_tau=n_tau_pts, tau_min=tau_lo, tau_max=tau_hi,
                    alpha=alpha_reg)

                p_norm = p_tau / (p_tau.sum() + 1e-30)

                # Reconstruction quality
                K_rec   = np.exp(-np.outer(t_d, 1.0/tau_grid))
                I_recon = K_rec @ p_tau
                I_recon_n = I_recon / (I_recon.max()+1e-30)
                I_d_n     = I_d    / (I_d.max()+1e-30)
                resid_d   = I_d_n - I_recon_n
                chi2_d    = reduced_chi2(I_d_n*I_d.max(), I_recon_n*I_d.max(), n_tau_pts)

                # Distribution stats
                tau_mean = float(np.sum(tau_grid * p_norm))
                tau_mode = float(tau_grid[np.argmax(p_tau)])
                tau_med  = float(tau_grid[np.searchsorted(np.cumsum(p_norm), 0.5)
                                          if np.cumsum(p_norm)[-1] >= 0.5 else -1])

                # ── Plots ──────────────────────────────────────────────────
                fig_d = make_subplots(rows=2, cols=2,
                                       subplot_titles=[
                                           "Lifetime Distribution  p(τ)  [log τ]",
                                           "Reconstruction Check",
                                           "Residuals (reconstruction)",
                                           "Weighted Residuals"
                                       ],
                                       row_heights=[0.65, 0.35],
                                       vertical_spacing=0.10,
                                       horizontal_spacing=0.08)

                # (1,1) Distribution
                fig_d.add_trace(go.Scatter(
                    x=tau_grid, y=p_norm, name='p(τ)',
                    line=dict(color=COLORS[0], width=2.5),
                    fill='tozeroy', fillcolor='rgba(76,155,232,0.18)'), row=1, col=1)
                fig_d.add_vline(x=tau_mean, row=1, col=1,
                                 line=dict(color='rgba(255,220,80,0.7)', dash='dash', width=1.5),
                                 annotation_text=f"⟨τ⟩={tau_mean:.3f}",
                                 annotation_font_color='rgba(255,220,80,0.9)')
                fig_d.add_vline(x=tau_mode, row=1, col=1,
                                 line=dict(color='rgba(120,255,160,0.6)', dash='dot', width=1.5),
                                 annotation_text=f"τ_mode={tau_mode:.3f}",
                                 annotation_font_color='rgba(120,255,160,0.9)')

                # (1,2) Reconstruction
                fig_d.add_trace(go.Scatter(x=t_d, y=I_d_n, name='Data',
                                            mode='markers',
                                            marker=dict(color='rgba(140,160,255,0.4)', size=3)), row=1, col=2)
                fig_d.add_trace(go.Scatter(x=t_d, y=I_recon_n, name='Reconstruction',
                                            line=dict(color='white', width=2.5)), row=1, col=2)

                # (2,1) Residuals
                fig_d.add_trace(go.Scatter(x=t_d, y=resid_d, name='Residuals',
                                            line=dict(color='rgba(232,92,76,0.85)', width=1.2),
                                            fill='tozeroy', fillcolor='rgba(232,92,76,0.1)'), row=2, col=1)
                fig_d.add_hline(y=0, line=dict(color='rgba(255,255,255,0.2)', dash='dash'), row=2, col=1)

                # (2,2) Weighted residuals
                sig_d  = np.sqrt(np.maximum(I_d, 1.0))
                w_rd   = (I_d * (I_d_n - I_recon_n)) / sig_d
                fig_d.add_trace(go.Scatter(x=t_d, y=w_rd, name='Weighted res.',
                                            line=dict(color='rgba(76,210,185,0.85)', width=1.2),
                                            fill='tozeroy', fillcolor='rgba(76,210,185,0.08)'), row=2, col=2)
                fig_d.add_hline(y=0,  line=dict(color='rgba(255,255,255,0.2)', dash='dash'), row=2, col=2)
                fig_d.add_hline(y=2,  line=dict(color='rgba(255,210,80,0.3)', dash='dot'),   row=2, col=2)
                fig_d.add_hline(y=-2, line=dict(color='rgba(255,210,80,0.3)', dash='dot'),   row=2, col=2)

                fig_d.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(18,18,28,0.95)',
                    plot_bgcolor='rgba(18,18,28,0.95)',
                    font=dict(family='monospace', size=12),
                    height=600,
                    title=f"Lifetime Distribution — {sel_dist} | α={alpha_reg:.0e} | χ²_red={chi2_d:.3f}"
                )
                if log_scale:
                    fig_d.update_yaxes(type='log', row=1, col=2)
                fig_d.update_xaxes(title_text=f"τ ({time_unit})", type='log', row=1, col=1)
                fig_d.update_xaxes(title_text=f"Time ({time_unit})", row=1, col=2)
                fig_d.update_xaxes(title_text=f"Time ({time_unit})", row=2, col=1)
                fig_d.update_xaxes(title_text=f"Time ({time_unit})", row=2, col=2)
                fig_d.update_yaxes(title_text="p(τ) (norm.)", row=1, col=1)

                st.plotly_chart(fig_d, use_container_width=True)

                # Metrics
                dm1, dm2, dm3, dm4 = st.columns(4)
                dm1.metric(f"⟨τ⟩ ({time_unit})", f"{tau_mean:.4f}")
                dm2.metric(f"τ_mode ({time_unit})", f"{tau_mode:.4f}")
                dm3.metric(f"τ_median ({time_unit})", f"{tau_med:.4f}")
                dm4.metric("χ²_red (recon.)", f"{chi2_d:.4f}")

                # Download
                excel_bytes = to_excel_download({
                    'Distribution': pd.DataFrame({
                        f'τ ({time_unit})': tau_grid,
                        'p(τ)': p_tau, 'p(τ) norm': p_norm}),
                    'Reconstruction': pd.DataFrame({
                        f'Time ({time_unit})': t_d,
                        'Data norm': I_d_n, 'Reconstruction': I_recon_n,
                        'Residuals': resid_d}),
                })
                st.download_button("📥 Distribution 결과 Excel", data=excel_bytes,
                                   file_name=f"trpl_dist_{sel_dist}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            except Exception as e:
                import traceback
                st.error(f"Distribution 계산 실패: {e}")
                with st.expander("상세 오류"):
                    st.code(traceback.format_exc())

# ── TAB 5: Multi-file comparison ──────────────────────────────────────────────
with tab_cmp:
    st.subheader("다중 파일 TRPL 비교")
    fig_cmp = make_figure(title="TRPL Comparison (normalized)")
    style_axes(fig_cmp, f"Time ({time_unit})", "Normalized Intensity")
    if log_scale: fig_cmp.update_yaxes(type='log')

    for i, s in enumerate(spectra):
        tc = s['wavelength']; Ic = s['intensity'].copy()
        mask_c = (tc >= t_start_fit) & (tc <= t_end_fit)
        tc, Ic = tc[mask_c], Ic[mask_c]
        if Ic.max() > 0: Ic /= Ic.max()
        fig_cmp.add_trace(go.Scatter(x=tc, y=Ic, name=s['name'],
                                      line=dict(color=colors_all[i], width=2)))

    if irf_loaded:
        irf_yc = np.clip((irf_y_raw-irf_offset_)*irf_scale_, 0, None)
        if irf_yc.max() > 0: irf_yc /= irf_yc.max()
        fig_cmp.add_trace(go.Scatter(
            x=irf_t_raw+irf_t_shift, y=irf_yc, name='IRF',
            line=dict(color='rgba(255,220,80,0.7)', width=1.5, dash='dot')))

    st.plotly_chart(fig_cmp, use_container_width=True)
