from __future__ import annotations

from typing import Dict, List

import streamlit as st

from medrag_ui.retrieval_pipeline import run_retrieval


st.set_page_config(
    page_title="MedRAG",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-deep: #07111f;
            --bg-mid: #0c2238;
            --bg-soft: #11314c;
            --panel: rgba(202, 231, 255, 0.10);
            --panel-strong: rgba(229, 243, 255, 0.16);
            --panel-heavy: rgba(11, 29, 49, 0.58);
            --text: #e8f4ff;
            --muted: #99b8d3;
            --line: rgba(182, 221, 255, 0.18);
            --line-strong: rgba(182, 221, 255, 0.28);
            --accent: #8fd6ff;
            --accent-2: #c3ebff;
            --accent-3: #58b6f8;
            --shadow: 0 24px 70px rgba(2, 10, 22, 0.42);
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 18%, rgba(84, 182, 245, 0.30), transparent 24%),
                radial-gradient(circle at 82% 14%, rgba(157, 216, 255, 0.18), transparent 22%),
                radial-gradient(circle at 50% 78%, rgba(69, 145, 204, 0.18), transparent 28%),
                linear-gradient(180deg, var(--bg-deep) 0%, var(--bg-mid) 52%, #08131f 100%);
            color: var(--text);
            font-family: "Avenir Next", "Helvetica Neue", "Segoe UI", sans-serif;
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        [data-testid="stHeader"] {
            background: linear-gradient(180deg, rgba(196, 231, 255, 0.42), rgba(160, 211, 246, 0.26)) !important;
            border-bottom: 1px solid rgba(183, 223, 255, 0.22);
            backdrop-filter: blur(16px);
            height: 3.05rem;
        }

        [data-testid="stHeader"] > div {
            height: 3.05rem;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(8, 24, 40, 0.94), rgba(9, 30, 48, 0.88));
            border-right: 1px solid rgba(170, 218, 255, 0.12);
        }

        [data-testid="stSidebar"] > div:first-child {
            backdrop-filter: blur(18px);
        }

        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] div {
            color: var(--text);
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--accent-2) !important;
        }

        [data-testid="stSidebar"] [data-baseweb="radio"] label span,
        [data-testid="stSidebar"] [data-baseweb="checkbox"] label span {
            color: var(--text) !important;
        }

        [data-testid="stSidebar"] input[type="radio"],
        [data-testid="stSidebar"] input[type="checkbox"] {
            accent-color: #8fd6ff !important;
        }

        [data-testid="stSidebar"] input[type="range"] {
            accent-color: #8fd6ff !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: rgba(190, 228, 255, 0.40) !important;
            border: 1px solid rgba(170, 218, 255, 0.22) !important;
        }

        [data-testid="stSidebar"] [data-baseweb="radio"] label div[aria-checked="true"] {
            background-color: rgba(143, 214, 255, 0.18) !important;
            border-color: #8fd6ff !important;
        }

        [data-testid="stSidebar"] [data-baseweb="radio"] label div[aria-checked="true"] * {
            color: #8fd6ff !important;
        }

        h1, h2, h3 {
            font-family: "Optima", "Avenir Next", "Segoe UI", sans-serif;
            color: var(--text);
            letter-spacing: -0.02em;
        }

        .hero {
            background:
                linear-gradient(135deg, rgba(191, 229, 255, 0.15), rgba(34, 93, 138, 0.16)),
                linear-gradient(180deg, rgba(9, 29, 47, 0.72), rgba(8, 24, 40, 0.52));
            border: 1px solid var(--line-strong);
            border-radius: 30px;
            padding: 2.2rem 2.1rem 1.7rem 2.1rem;
            box-shadow: var(--shadow);
            overflow: hidden;
            position: relative;
            backdrop-filter: blur(20px);
        }

        .hero:before {
            content: "";
            position: absolute;
            inset: auto -34px -54px auto;
            width: 260px;
            height: 260px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(143, 214, 255, 0.32), rgba(88, 182, 248, 0));
        }

        .hero:after {
            content: "";
            position: absolute;
            top: -60px;
            right: 140px;
            width: 180px;
            height: 180px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(195, 235, 255, 0.16), rgba(195, 235, 255, 0));
        }

        .hero-kicker {
            color: var(--accent);
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            margin-bottom: 0.6rem;
        }

        .hero-text {
            color: var(--muted);
            font-size: 1.03rem;
            line-height: 1.75;
            max-width: 55rem;
        }

        .hero-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 0.7rem;
            margin-top: 1.25rem;
        }

        .hero-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: rgba(196, 231, 255, 0.10);
            border: 1px solid rgba(182, 221, 255, 0.18);
            color: var(--accent-2);
            border-radius: 999px;
            padding: 0.45rem 0.8rem;
            font-size: 0.84rem;
            font-weight: 600;
            backdrop-filter: blur(12px);
        }

        .stat-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 1.3rem;
        }

        .stat-card {
            background: linear-gradient(180deg, rgba(217, 240, 255, 0.12), rgba(157, 216, 255, 0.05));
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            backdrop-filter: blur(14px);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
        }

        .stat-label {
            color: var(--muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .stat-value {
            color: var(--text);
            font-size: 1.35rem;
            font-weight: 700;
            margin-top: 0.2rem;
        }

        .section-card {
            background:
                linear-gradient(180deg, rgba(192, 229, 255, 0.11), rgba(162, 211, 255, 0.07)),
                linear-gradient(180deg, rgba(8, 25, 42, 0.74), rgba(7, 20, 34, 0.56));
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1.25rem;
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
        }

        .product-card-title {
            color: var(--accent-2);
            font-size: 0.88rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.55rem;
            font-weight: 700;
        }

        .input-shell {
            background:
                linear-gradient(180deg, rgba(205, 235, 255, 0.12), rgba(157, 216, 255, 0.06)),
                linear-gradient(180deg, rgba(8, 25, 42, 0.74), rgba(7, 20, 34, 0.56));
            border: 1px solid var(--line);
            border-radius: 26px;
            padding: 1.25rem;
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
        }

        .mini-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 0.9rem;
        }

        .query-hero {
            background:
                linear-gradient(135deg, rgba(214, 239, 255, 0.18), rgba(94, 165, 216, 0.12)),
                linear-gradient(180deg, rgba(9, 28, 45, 0.88), rgba(7, 20, 34, 0.72));
            border: 1px solid rgba(182, 221, 255, 0.22);
            border-radius: 28px;
            padding: 1.35rem 1.4rem 1.2rem 1.4rem;
            box-shadow: 0 24px 60px rgba(4, 14, 28, 0.34);
            backdrop-filter: blur(18px);
            margin-top: 1.2rem;
            margin-bottom: 1.35rem;
        }

        .query-kicker {
            color: var(--accent);
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            margin-bottom: 0.45rem;
        }

        .query-copy {
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.6;
            margin-bottom: 0.85rem;
            max-width: 48rem;
        }

        .mini-card {
            background: linear-gradient(180deg, rgba(220, 241, 255, 0.10), rgba(158, 211, 249, 0.05));
            border: 1px solid rgba(182, 221, 255, 0.16);
            border-radius: 18px;
            padding: 0.85rem 0.95rem;
            backdrop-filter: blur(12px);
        }

        .mini-label {
            color: var(--muted);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .mini-value {
            color: var(--text);
            font-size: 1rem;
            font-weight: 700;
            margin-top: 0.2rem;
        }

        .answer-card {
            background:
                linear-gradient(180deg, rgba(163, 221, 255, 0.16), rgba(124, 188, 234, 0.08)),
                linear-gradient(180deg, rgba(8, 29, 48, 0.84), rgba(7, 21, 35, 0.76));
            border: 1px solid rgba(143, 214, 255, 0.22);
            border-radius: 24px;
            padding: 1.5rem;
            box-shadow: var(--shadow);
            backdrop-filter: blur(20px);
        }

        .caption {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.6;
        }

        .citation-card {
            background:
                linear-gradient(180deg, rgba(224, 243, 255, 0.12), rgba(154, 206, 247, 0.08)),
                linear-gradient(180deg, rgba(8, 24, 40, 0.82), rgba(8, 22, 35, 0.72));
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1rem 1rem 0.95rem 1rem;
            margin-bottom: 0.95rem;
            box-shadow: 0 16px 40px rgba(4, 12, 24, 0.26);
            backdrop-filter: blur(16px);
        }

        .retrieval-tag {
            display: inline-block;
            background: rgba(143, 214, 255, 0.12);
            color: var(--accent-2);
            border: 1px solid rgba(143, 214, 255, 0.18);
            border-radius: 999px;
            padding: 0.3rem 0.72rem;
            font-size: 0.8rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
        }

        [data-testid="stTextArea"] textarea,
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        [data-testid="stNumberInput"] input {
            background: rgba(232, 245, 255, 0.96) !important;
            color: #06131f !important;
            border: 1px solid rgba(170, 218, 255, 0.20) !important;
            border-radius: 18px !important;
            backdrop-filter: blur(12px);
        }

        [data-testid="stTextArea"] textarea {
            min-height: 146px !important;
            font-size: 1rem !important;
            line-height: 1.65 !important;
            padding: 1rem 1rem 1rem 1rem !important;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
            caret-color: #06131f !important;
        }

        [data-testid="stTextArea"] label {
            color: var(--accent-2) !important;
            font-weight: 700 !important;
            letter-spacing: 0.01em;
        }

        [data-testid="stTextArea"] textarea::placeholder {
            color: rgba(20, 48, 74, 0.45) !important;
        }

        .stButton > button {
            background: linear-gradient(135deg, #7bc3f4, #4ba7e8) !important;
            color: #041321 !important;
            border: 0 !important;
            border-radius: 16px !important;
            height: 3rem;
            font-weight: 700;
            letter-spacing: 0.01em;
            box-shadow: 0 14px 34px rgba(57, 151, 218, 0.35);
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #98d7ff, #60b4ee) !important;
        }

        [data-testid="metric-container"] {
            background: linear-gradient(180deg, rgba(188, 228, 255, 0.11), rgba(156, 206, 246, 0.07));
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            backdrop-filter: blur(16px);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
        }

        [data-testid="metric-container"] label,
        [data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: var(--text) !important;
        }

        .stAlert {
            background: rgba(132, 187, 226, 0.12) !important;
            color: var(--text) !important;
            border: 1px solid rgba(155, 210, 248, 0.24) !important;
            border-radius: 16px !important;
            backdrop-filter: blur(16px);
        }

        a {
            color: var(--accent) !important;
        }

        code {
            color: var(--accent-2) !important;
            background: rgba(190, 228, 255, 0.08) !important;
            padding: 0.15rem 0.35rem;
            border-radius: 8px;
        }

        @media (max-width: 900px) {
            .stat-row {
                grid-template-columns: 1fr;
            }

            .mini-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Trusted Medical Evidence Workspace</div>
            <h1>MedRAG</h1>
            <p class="hero-text">
                MedRAG helps users explore medical questions through retrieval-first, citation-aware
                local AI. Every answer is grounded in surfaced source context so the system feels
                closer to a clinical evidence console than a generic chatbot.
            </p>
            <div class="hero-actions">
                <div class="hero-pill">Local LLM inference</div>
                <div class="hero-pill">Traceable citations</div>
                <div class="hero-pill">Sparse, dense, hybrid search</div>
            </div>
            <div class="stat-row">
                <div class="stat-card">
                    <div class="stat-label">Knowledge Base</div>
                    <div class="stat-value">MedQuAD</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Inference Layer</div>
                    <div class="stat-value">Ollama + Llama 3</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Response Format</div>
                    <div class="stat-value">Answer + Evidence</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> Dict[str, int | str]:
    st.sidebar.markdown("## Search Settings")
    strategy = st.sidebar.radio(
        "Retrieval strategy",
        options=["Hybrid", "Sparse (BM25)", "Dense"],
        help="Switch between your retrieval baselines and comparison modes.",
    )
    top_k = st.sidebar.slider("Top-k passages", min_value=2, max_value=10, value=5)
    rerank_depth = st.sidebar.slider("Dense re-rank depth", min_value=5, max_value=25, value=10)
    temperature = st.sidebar.slider("Generation temperature", min_value=0.0, max_value=1.0, value=0.2)
    model_name = st.sidebar.selectbox(
        "Local LLM",
        options=["llama3"],
        help="This first integration targets Ollama with the llama3 model.",
    )

    st.sidebar.markdown("## Display Options")
    st.sidebar.checkbox("Show retrieval diagnostics", value=True, key="show_diagnostics")
    st.sidebar.checkbox("Show evidence panel", value=True, key="show_evidence")
    st.sidebar.checkbox("Show answer confidence", value=True, key="show_confidence")

    return {
        "strategy": strategy,
        "top_k": top_k,
        "rerank_depth": rerank_depth,
        "temperature": temperature,
        "model_name": model_name,
    }


def render_answer(result: Dict[str, object]) -> None:
    st.markdown('<div class="retrieval-tag">Grounded response</div>', unsafe_allow_html=True)
    st.markdown("### Clinical Answer")
    st.markdown('<div class="answer-card">', unsafe_allow_html=True)
    st.write(result["answer"])
    st.markdown("</div>", unsafe_allow_html=True)
    if result.get("llm_error"):
        st.warning(f"Local LLM fallback active: {result['llm_error']}")

    if st.session_state.get("show_confidence", True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Retriever", result["retrieval_strategy"])
        col2.metric("Grounding score", result["grounding_score"])
        col3.metric("Citations", len(result["citations"]))


def render_citations(citations: List[Dict[str, str]]) -> None:
    st.markdown("### Evidence Sources")
    for idx, citation in enumerate(citations, start=1):
        st.markdown('<div class="citation-card">', unsafe_allow_html=True)
        st.markdown(f"**[{idx}] {citation['title']}**")
        st.markdown(
            f"<div class='caption'><strong>Source:</strong> {citation['source']} | "
            f"<strong>Focus:</strong> {citation['focus_area']}</div>",
            unsafe_allow_html=True,
        )
        st.write(citation["snippet"])
        # st.markdown(f"[Open source]({citation['url']})")
        st.markdown("</div>", unsafe_allow_html=True)


def render_diagnostics(result: Dict[str, object]) -> None:
    st.markdown("### System Diagnostics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Retrieval Metrics")
        for label, value in result["metrics"].items():
            st.metric(label, value)
    with col2:
        st.markdown("#### Run Summary")
        st.markdown(
            f"""
            <div class="section-card">
                <p class="caption">
                    This answer was generated with <strong>{result['retrieval_strategy']}</strong>
                    retrieval over <code>{result['chunks_path']}</code> and a local
                    <code>{result['llm_model']}</code> model. The current panel shows live
                    operational metrics and is ready to expand into evaluation-grade benchmark
                    reporting.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    load_styles()
    controls = render_sidebar()

    st.markdown(
        """
        <div class="query-hero">
            <div class="query-kicker">Primary Input</div>
            <h3 style="margin-top:0; margin-bottom:0.4rem;">Ask MedRAG a medical question</h3>
            <div class="query-copy">
                Type in plain language. The system retrieves supporting evidence first, then uses
                the local model to generate a grounded response with visible citations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    query = st.text_area(
        "Question",
        value="What are the common symptoms of iron deficiency anemia?",
        height=150,
        label_visibility="visible",
        placeholder="Enter a patient-friendly medical question...",
    )
    st.markdown(
        """
        <div class="mini-grid">
            <div class="mini-card">
                <div class="mini-label">Retrieval Mode</div>
                <div class="mini-value">Compare ranking behavior</div>
            </div>
            <div class="mini-card">
                <div class="mini-label">Answer Policy</div>
                <div class="mini-value">Context-grounded only</div>
            </div>
            <div class="mini-card">
                <div class="mini-label">Output</div>
                <div class="mini-value">Answer with citations</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    run = st.button("Run MedRAG", type="primary", use_container_width=True)

    render_hero()

    st.markdown("")
    info_col, spacer_col = st.columns([1.1, 0.9])

    with info_col:
        st.markdown(
            """
            <div class="section-card">
                <div class="product-card-title">How It Works</div>
                <h3 style="margin-top:0;">Evidence-First QA</h3>
                <p class="caption">
                    Retrieval runs first, generation runs second, and citations stay visible
                    throughout the experience. This keeps the page closer to a professional
                    medical decision-support surface than a free-form assistant.
                </p>
                <p class="caption" style="margin-top:0.8rem;">
                    Use the sidebar to adjust search mode, top-k evidence, rerank depth, and
                    generation temperature. The answer panel shows the grounded response, while
                    the evidence panel exposes the exact source chunks used for support.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if run:
        with st.spinner("Retrieving evidence and drafting a grounded answer..."):
            result = run_retrieval(
                query=query,
                strategy=controls["strategy"],
                top_k=controls["top_k"],
                candidate_k=controls["rerank_depth"],
                temperature=controls["temperature"],
                model_name=controls["model_name"],
            )

        left, right = st.columns([1.25, 1])
        with left:
            render_answer(result)
            if st.session_state.get("show_diagnostics", True):
                st.markdown("")
                render_diagnostics(result)
        with right:
            if st.session_state.get("show_evidence", True):
                render_citations(result["citations"])


if __name__ == "__main__":
    main()
