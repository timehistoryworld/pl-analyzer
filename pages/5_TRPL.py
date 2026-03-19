"""
Module 5 — TRPL: Time-Resolved PL Lifetime Fitting
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io_utils import load_spectrum_file, load_multiple_files, to_excel_download
from utils.fitting_utils import (mono_exp, bi_exp, tri_exp,
                                  amplitude_weighted_lifetime, intensity_weighted_lifetime)
from utils.plot_utils import make_figure, style_axes, COLORS, rainbow_colors

st.set_page_config(page_title="TRPL | PL Analyzer", layout="wide", page_icon="⏱")
st.title("⏱ TRPL — Time-Resolved PL & Lifetime Fitting")
st.markdown("단일/이중/삼중 지수 감쇠 피팅, amplitude/intensity-weighted lifetime 계산")

# ── File upload ─────────────────────────────────────────────────────────────
col_up, col_set = st.columns([2, 1])
with col_up:
    trpl_files = st.file_uploader("TRPL 파일(들) — 첫 번째 열: 시간, 두 번째 열: 강도",
                                   type=['csv','txt','xlsx','xls'], accept_multiple_files=True)
with col_set:
    time_unit = st.selectbox("시간 단위", ["ns", "ps", "µs", "ms"])
    log_scale = st.checkbox("Log y축", value=True)
    normalize_trpl = st.checkbox("최대값 정규화", value=True)
    t_start = st.number_input("피팅 시작 시간 (t_start)", value=0.0, format="%.4f",
                               help="IRF 또는 초기 artifact 제외")

if not trpl_files:
    st.info("TRPL 데이터를 업로드하세요. 형식: [time, intensity]")
    st.stop()

spectra = load_multiple_files(trpl_files)
if not spectra:
    st.error("파일 로딩 실패"); st.stop()

st.success(f"{len(spectra)}개 파일 로드")

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("피팅 모델")
    decay_model = st.radio("모델 선택",
                            ["단일 지수 (Mono-exponential)",
                             "이중 지수 (Bi-exponential)",
                             "삼중 지수 (Tri-exponential)"])
    st.markdown("---")
    st.subheader("초기값 설정")
    n_comp = {"단일": 1, "이중": 2, "삼중": 3}[decay_model[:2]]

    p0_vals = []
    for k in range(n_comp):
        st.markdown(f"**성분 {k+1}**")
        a = st.number_input(f"A{k+1}", value=1.0/(k+1), key=f'A{k}', format="%.4f")
        tau = st.number_input(f"τ{k+1} ({time_unit})", value=float((k+1)*5), key=f'tau{k}', min_value=0.001, format="%.4f")
        p0_vals += [a, tau]
    y0_init = st.number_input("Baseline y₀", value=0.0, format="%.6f")
    p0_vals.append(y0_init)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_overview, tab_fit, tab_compare = st.tabs(["📈 스펙트럼 보기", "🔄 단일 파일 피팅", "📊 다중 파일 비교"])

colors_all = rainbow_colors(len(spectra))

with tab_overview:
    fig_ov = make_figure(title="TRPL Decay Curves")
    style_axes(fig_ov, f"Time ({time_unit})", "Intensity (a.u.)")
    if log_scale:
        fig_ov.update_yaxes(type='log')

    for i, s in enumerate(spectra):
        t = s['wavelength']   # first column = time
        inten = s['intensity']
        if normalize_trpl and inten.max() > 0: inten = inten / inten.max()
        mask = t >= t_start
        fig_ov.add_trace(go.Scatter(x=t[mask], y=inten[mask], name=s['name'],
                                     line=dict(color=colors_all[i], width=2)))
    st.plotly_chart(fig_ov, use_container_width=True)

with tab_fit:
    sel_name = st.selectbox("피팅할 파일", [s['name'] for s in spectra])
    s_sel = next(s for s in spectra if s['name'] == sel_name)
    t_raw = s_sel['wavelength']
    I_raw = s_sel['intensity']

    # Apply t_start mask
    mask = t_raw >= t_start
    t = t_raw[mask]
    I = I_raw[mask]

    if normalize_trpl and I.max() > 0:
        I = I / I.max()

    # Select model function
    model_map = {
        "단일 지수 (Mono-exponential)": (mono_exp, 3),
        "이중 지수 (Bi-exponential)":   (bi_exp, 5),
        "삼중 지수 (Tri-exponential)":  (tri_exp, 7),
    }
    model_func, n_params = model_map[decay_model]

    if st.button("🔄 피팅 실행", type="primary"):
        # Bounds: all positive except y0 which can be small negative
        lo = [0] * (n_params - 1) + [-0.1]
        hi = [np.inf] * (n_params - 1) + [0.5]

        try:
            popt, pcov = curve_fit(model_func, t, I, p0=p0_vals[:n_params],
                                    bounds=(lo, hi), maxfev=100000)
            perr = np.sqrt(np.diag(pcov))
            t_fine = np.linspace(t.min(), t.max(), 2000)
            I_fit = model_func(t_fine, *popt)
            residuals = I - model_func(t, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((I - I.mean())**2)
            r2 = 1 - ss_res/ss_tot

            # Weighted residuals (chi-squared proxy)
            chi2_red = np.sum((residuals**2) / (np.abs(I) + 1e-10)) / max(len(I) - n_params, 1)

            # Plot with residuals
            fig_fit = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25],
                                     shared_xaxes=True, vertical_spacing=0.04)
            fig_fit.add_trace(go.Scatter(x=t, y=I, mode='markers', name='Data',
                                          marker=dict(color='rgba(180,180,255,0.5)', size=4)), row=1, col=1)
            fig_fit.add_trace(go.Scatter(x=t_fine, y=I_fit, name='Fit',
                                          line=dict(color='white', width=2.5)), row=1, col=1)
            fig_fit.add_trace(go.Scatter(x=t, y=residuals, name='Residuals',
                                          mode='lines+markers', marker=dict(size=3),
                                          line=dict(color='rgba(232,92,76,0.8)', width=1.5),
                                          fill='tozeroy', fillcolor='rgba(232,92,76,0.1)'), row=2, col=1)
            fig_fit.add_hline(y=0, line=dict(color='rgba(255,255,255,0.3)', dash='dash'), row=2, col=1)
            fig_fit.update_layout(template='plotly_dark', paper_bgcolor='rgba(18,18,28,0.95)',
                                   plot_bgcolor='rgba(18,18,28,0.95)',
                                   title=f"TRPL Fit — {sel_name}  (R² = {r2:.5f})")
            if log_scale:
                fig_fit.update_yaxes(type='log', row=1, col=1)
            fig_fit.update_xaxes(title_text=f"Time ({time_unit})", row=2, col=1)
            fig_fit.update_yaxes(title_text="Intensity", row=1, col=1)
            fig_fit.update_yaxes(title_text="Residuals", row=2, col=1)
            st.plotly_chart(fig_fit, use_container_width=True)

            # ── Parameter results ────────────────────────────────────────────
            st.markdown(f"**R² = {r2:.6f} &nbsp;&nbsp; χ²_red = {chi2_red:.4f}**")

            amps, taus = [], []
            result_rows = []

            if "단일" in decay_model:
                A1, tau1, y0 = popt
                amps, taus = [A1], [tau1]
                result_rows = [
                    {'Parameter': 'A₁', 'Value': f'{A1:.6f}', 'Std Error': f'{perr[0]:.6f}'},
                    {'Parameter': f'τ₁ ({time_unit})', 'Value': f'{tau1:.4f}', 'Std Error': f'{perr[1]:.4f}'},
                    {'Parameter': 'y₀ (baseline)', 'Value': f'{y0:.6f}', 'Std Error': f'{perr[2]:.6f}'},
                ]
            elif "이중" in decay_model:
                A1, tau1, A2, tau2, y0 = popt
                amps, taus = [A1, A2], [tau1, tau2]
                total_A = A1 + A2
                result_rows = [
                    {'Parameter': 'A₁', 'Value': f'{A1:.6f}', 'Std Error': f'{perr[0]:.6f}', 'Fraction (%)': f'{A1/total_A*100:.1f}'},
                    {'Parameter': f'τ₁ ({time_unit})', 'Value': f'{tau1:.4f}', 'Std Error': f'{perr[1]:.4f}', 'Fraction (%)': ''},
                    {'Parameter': 'A₂', 'Value': f'{A2:.6f}', 'Std Error': f'{perr[2]:.6f}', 'Fraction (%)': f'{A2/total_A*100:.1f}'},
                    {'Parameter': f'τ₂ ({time_unit})', 'Value': f'{tau2:.4f}', 'Std Error': f'{perr[3]:.4f}', 'Fraction (%)': ''},
                    {'Parameter': 'y₀', 'Value': f'{y0:.6f}', 'Std Error': f'{perr[4]:.6f}', 'Fraction (%)': ''},
                ]
            else:
                A1, tau1, A2, tau2, A3, tau3, y0 = popt
                amps, taus = [A1, A2, A3], [tau1, tau2, tau3]
                total_A = A1 + A2 + A3
                result_rows = [
                    {'Parameter': 'A₁', 'Value': f'{A1:.6f}', 'Std Error': f'{perr[0]:.6f}', 'Fraction (%)': f'{A1/total_A*100:.1f}'},
                    {'Parameter': f'τ₁ ({time_unit})', 'Value': f'{tau1:.4f}', 'Std Error': f'{perr[1]:.4f}', 'Fraction (%)': ''},
                    {'Parameter': 'A₂', 'Value': f'{A2:.6f}', 'Std Error': f'{perr[2]:.6f}', 'Fraction (%)': f'{A2/total_A*100:.1f}'},
                    {'Parameter': f'τ₂ ({time_unit})', 'Value': f'{tau2:.4f}', 'Std Error': f'{perr[3]:.4f}', 'Fraction (%)': ''},
                    {'Parameter': 'A₃', 'Value': f'{A3:.6f}', 'Std Error': f'{perr[4]:.6f}', 'Fraction (%)': f'{A3/total_A*100:.1f}'},
                    {'Parameter': f'τ₃ ({time_unit})', 'Value': f'{tau3:.4f}', 'Std Error': f'{perr[5]:.4f}', 'Fraction (%)': ''},
                    {'Parameter': 'y₀', 'Value': f'{y0:.6f}', 'Std Error': f'{perr[6]:.6f}', 'Fraction (%)': ''},
                ]

            st.dataframe(pd.DataFrame(result_rows), use_container_width=True)

            # Weighted lifetimes
            if len(amps) > 0:
                tau_amp = amplitude_weighted_lifetime(amps, taus)
                tau_int = intensity_weighted_lifetime(amps, taus)
                col_m1, col_m2 = st.columns(2)
                col_m1.metric(f"<τ>_amp ({time_unit})", f"{tau_amp:.4f}",
                               help="Σ(Ai·τi) / Σ(Ai)")
                col_m2.metric(f"<τ>_int ({time_unit})", f"{tau_int:.4f}",
                               help="Σ(Ai·τi²) / Σ(Ai·τi)")

            # Export
            df_params = pd.DataFrame(result_rows)
            df_curve = pd.DataFrame({f'Time ({time_unit})': t_fine, 'Fit': I_fit})
            df_data  = pd.DataFrame({f'Time ({time_unit})': t, 'Data': I, 'Residuals': residuals})
            excel_bytes = to_excel_download({'Parameters': df_params, 'Fit Curve': df_curve, 'Data+Residuals': df_data})
            st.download_button("📥 피팅 결과 Excel", data=excel_bytes, file_name="trpl_fit.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.error(f"피팅 실패: {e}\n초기값을 조정하거나 t_start를 변경해보세요.")

with tab_compare:
    st.subheader("다중 파일 감쇠 곡선 비교")
    fig_cmp = make_figure(title="TRPL Comparison")
    style_axes(fig_cmp, f"Time ({time_unit})", "Normalized Intensity")
    if log_scale: fig_cmp.update_yaxes(type='log')

    for i, s in enumerate(spectra):
        t = s['wavelength']
        I = s['intensity']
        mask = t >= t_start
        t, I = t[mask], I[mask]
        if I.max() > 0: I = I / I.max()
        fig_cmp.add_trace(go.Scatter(x=t, y=I, name=s['name'],
                                      line=dict(color=colors_all[i], width=2)))
    st.plotly_chart(fig_cmp, use_container_width=True)
