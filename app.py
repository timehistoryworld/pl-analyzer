"""
PL Analyzer — Photoluminescence Analysis Suite
Main entry point / Home page
"""
import streamlit as st

st.set_page_config(
    page_title="PL Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4C9BE8 0%, #9B4CE8 50%, #E85C9B 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}

.subtitle {
    color: rgba(200,200,220,0.7);
    font-size: 1.0rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.08em;
    margin-bottom: 2rem;
}

.module-card {
    background: linear-gradient(135deg, rgba(30,30,50,0.9), rgba(20,20,40,0.95));
    border: 1px solid rgba(100,100,200,0.25);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}

.module-card:hover {
    border-color: rgba(100,150,255,0.5);
}

.module-number {
    font-family: 'JetBrains Mono', monospace;
    color: #4C9BE8;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.module-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #E8E8FF;
    margin: 0.2rem 0;
}

.module-desc {
    font-size: 0.85rem;
    color: rgba(180,180,210,0.7);
    line-height: 1.5;
}

.badge {
    display: inline-block;
    background: rgba(76,155,232,0.2);
    border: 1px solid rgba(76,155,232,0.4);
    border-radius: 4px;
    padding: 0.1rem 0.5rem;
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    color: #4C9BE8;
    margin-right: 0.3rem;
    margin-top: 0.4rem;
}

.info-box {
    background: rgba(76,155,232,0.08);
    border-left: 3px solid #4C9BE8;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    color: rgba(200,210,255,0.85);
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">PL Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Photoluminescence Analysis Suite · v1.0</div>', unsafe_allow_html=True)

st.markdown('<div class="info-box">📁 <b>지원 파일 형식:</b> CSV, TXT, Excel (.xlsx/.xls) &nbsp;|&nbsp; 구분자: 쉼표, 탭, 공백 자동 인식 &nbsp;|&nbsp; 첫 번째 열: 파장(nm), 두 번째 열: 강도</div>', unsafe_allow_html=True)
st.markdown('<br>', unsafe_allow_html=True)

# ── Module cards ──────────────────────────────────────────────────────────────
modules = [
    {
        "num": "Module 1",
        "title": "📊 Basics — Peak Analysis & Gaussian Fitting",
        "desc": "다중 스펙트럼 평균화, 피크 자동 감지 (1차/2차 미분 포함), 단일 및 다중 Gaussian 피팅, FWHM 추출",
        "tags": ["averaging", "peak detection", "gaussian fit", "FWHM", "derivative"],
        "page": "pages/1_Basics.py"
    },
    {
        "num": "Module 2",
        "title": "💡 PLQY — Photoluminescence Quantum Yield",
        "desc": "비교법(comparative method)으로 PLQY 계산. Sample + Reference의 Abs/PL 데이터 업로드, 여기 파장별 흡광도 보정, PL 적분 범위 설정",
        "tags": ["PLQY", "comparative method", "reference", "quantum yield"],
        "page": "pages/2_PLQY.py"
    },
    {
        "num": "Module 3",
        "title": "🔧 Spectral Tools — Raman Subtraction & eV Conversion",
        "desc": "여기 파장 기반 Raman peak 자동 제거, 파장→에너지 축 변환 (Jacobian transform 적용으로 스펙트럼 왜곡 보정)",
        "tags": ["Raman subtraction", "eV conversion", "Jacobian", "energy axis"],
        "page": "pages/3_Spectral_Tools.py"
    },
    {
        "num": "Module 4",
        "title": "📉 Stern-Volmer — Quenching Analysis",
        "desc": "다중 농도 파일 업로드, 농도 입력 테이블, Linear/Modified/Combined SV 피팅, Ksv·kq·fa 파라미터 추출, 선형/비선형 구분 자동 판별",
        "tags": ["Stern-Volmer", "quenching", "Ksv", "kq", "modified SV"],
        "page": "pages/4_Stern_Volmer.py"
    },
    {
        "num": "Module 5",
        "title": "⏱ TRPL — Time-Resolved PL & Lifetime Fitting",
        "desc": "단일/이중/삼중 지수 감쇠 피팅, amplitude/intensity-weighted lifetime 계산, IRF deconvolution (선택), 잔차 플롯",
        "tags": ["TRPL", "lifetime", "decay", "mono/bi/tri-exp", "IRF"],
        "page": "pages/5_TRPL.py"
    },
    {
        "num": "Module 6",
        "title": "🗺 EEM — Excitation-Emission Matrix",
        "desc": "여러 여기 파장에서 측정한 PL 스펙트럼을 2D contour/heatmap으로 시각화, Rayleigh/Raman 산란선 마스킹",
        "tags": ["EEM", "excitation-emission matrix", "contour", "2D heatmap"],
        "page": "pages/6_EEM.py"
    },
    {
        "num": "Module 7",
        "title": "🌡 Temperature-Dependent PL",
        "desc": "온도별 PL 스펙트럼 비교, Peak position vs T 피팅 (Varshni + Bose-Einstein), FWHM vs T 분석 (phonon coupling), 열적 소광 (Ea 추출)",
        "tags": ["temperature PL", "Varshni", "Bose-Einstein", "phonon", "thermal quenching"],
        "page": "pages/7_TempPL.py"
    },
    {
        "num": "Module 8",
        "title": "🔬 Deconvolution — Multi-component Spectral Fitting",
        "desc": "다중 Gaussian/Voigt 피팅, 피크 수 자유 설정, 초기값 대화형 조정, 면적비·중심·FWHM 결과 테이블, 잔차 분석",
        "tags": ["deconvolution", "multi-Gaussian", "Voigt", "spectral fitting", "components"],
        "page": "pages/8_Deconvolution.py"
    },
]

col1, col2 = st.columns(2)
for i, m in enumerate(modules):
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        badges_html = ''.join([f'<span class="badge">{t}</span>' for t in m["tags"]])
        st.markdown(f"""
        <div class="module-card">
            <div class="module-number">{m['num']}</div>
            <div class="module-title">{m['title']}</div>
            <div class="module-desc">{m['desc']}</div>
            <div style="margin-top:0.5rem">{badges_html}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:rgba(150,150,180,0.5); font-size:0.8rem; font-family:monospace">
PL Analyzer · Built with Streamlit · 
<a href="https://github.com" style="color:rgba(76,155,232,0.6)">GitHub</a>
</div>
""", unsafe_allow_html=True)
