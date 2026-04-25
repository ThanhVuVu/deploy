"""
demo_scripting.py  –  Streamlit UI for Scripting Agent + Multimodal RAG
Run:  streamlit run demo_scripting.py
"""

import streamlit as st
import time

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Virtual Lab – Scripting Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Force dark theme colors for all text to fix Streamlit's default light text */
.stApp {
    background: #0f1117; /* Streamlit native dark background */
}

/* Override default text colors that might be black */
p, h1, h2, h3, h4, h5, h6, span, label, li, div {
    color: #f1f5f9 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1a1c23 !important;
    border-right: 1px solid #2d3748 !important;
}

/* Hero header */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid #2d3748;
    margin-bottom: 2rem;
}
.hero-header h1 {
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #a855f7, #3b82f6, #10b981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    color: transparent !important; /* to let gradient show */
}
.hero-header p {
    color: #94a3b8 !important;
    font-size: 1.1rem !important;
    font-weight: 400 !important;
}

/* Text area */
.stTextArea textarea {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
    font-size: 1.05rem !important;
    padding: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 1px #3b82f6 !important;
}
.stTextArea textarea::placeholder {
    color: #64748b !important;
}

/* Primary Button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
}

/* Suggestion Buttons (Secondary) */
div[data-testid="stHorizontalBlock"] .stButton > button {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #cbd5e1 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    width: 100%;
    border-radius: 8px !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    background: #334155 !important;
    border-color: #475569 !important;
    color: #f8fafc !important;
}

/* Source Chips */
.source-chip {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    margin-right: 6px;
    letter-spacing: 0.5px;
}
.chip-text  { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
.chip-image { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }
.chip-table { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }

.score-badge {
    font-size: 0.75rem;
    padding: 3px 8px;
    border-radius: 6px;
    background: #334155;
    color: #cbd5e1;
    font-family: monospace;
}

/* Source Card Container */
.source-card-container {
    background: #1e293b;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    border: 1px solid #334155;
    transition: all 0.2s ease;
    height: 100%;
}
.source-card-container:hover {
    border-color: #475569;
    background: #233044;
}

/* Result card */
.result-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 2.5rem;
    margin-top: 1.5rem;
    box-shadow: 0 10px 20px -5px rgba(0,0,0,0.3);
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 16px;
    margin: 1.5rem 0;
}
.metric-card {
    flex: 1;
    background: #1e293b;
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    border: 1px solid #334155;
}
.metric-card .val {
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #a855f7, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent !important;
}
.metric-card .lbl {
    font-size: 0.8rem !important;
    color: #94a3b8 !important;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600 !important;
}

/* Spinner override */
div[data-testid="stSpinner"] > div {
    border-color: #8b5cf6 transparent transparent transparent !important;
}

hr { border-color: #334155 !important; margin: 2rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Lazy init (cache agent) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_agent():
    from src.agents.scripting.scripting_agent import ScriptingAgent
    return ScriptingAgent(use_multimodal=True)

@st.cache_resource(show_spinner=False)
def load_retriever():
    from src.agents.scripting.rag import MultimodalRetriever
    return MultimodalRetriever()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Cài đặt")
    top_k = st.slider("Số chunk RAG (top_k)", 2, 8, 4)
    show_raw_rag = st.checkbox("Hiển thị RAG context thô", value=False)

    st.markdown("---")
    st.markdown("### 📚 Nguồn dữ liệu")
    st.markdown("""
    <div style='color: rgba(255,255,255,0.6); font-size:0.82rem; line-height:1.7'>
    📄 <b>science_db_6.pdf</b><br>
    207 trang – Scan KHTN lớp 6<br>
    Bóc tách bởi <span style='color:#a78bfa'>GPT-4o Vision</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🗃️ DB Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Text", "389")
    col2.metric("Image", "205")
    col3.metric("Table", "56")

    st.markdown("---")
    st.markdown("""
    <div style='color:rgba(255,255,255,0.3); font-size:0.72rem; text-align:center'>
    A20 Virtual Lab · Multimodal RAG
    </div>
    """, unsafe_allow_html=True)

# ── Main layout ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🔬 Virtual Lab Scripting</h1>
    <p>Nhập yêu cầu · AI tìm kiếm tài liệu · Xuất kịch bản thí nghiệm chuẩn</p>
</div>
""", unsafe_allow_html=True)

# Suggested queries
st.markdown("**💡 Gợi ý nhanh:**")
SUGGESTIONS = [
    "Lập kịch bản thí nghiệm xác định thành phần % oxygen trong không khí",
    "Kịch bản thí nghiệm về vai trò của oxygen đối với sự cháy",
    "Hướng dẫn quan sát vi khuẩn lactic trong nước dưa muối",
    "Kịch bản thí nghiệm tìm hiểu tính chất đàn hồi của cao su",
    "Thí nghiệm xác định khả năng dẫn nhiệt của các vật liệu",
]
cols = st.columns(3)
for i, s in enumerate(SUGGESTIONS[:3]):
    if cols[i].button(s[:45] + "…", key=f"sug_{i}"):
        st.session_state["query_input"] = s
cols2 = st.columns(2)
for i, s in enumerate(SUGGESTIONS[3:]):
    if cols2[i].button(s[:55] + "…", key=f"sug2_{i}"):
        st.session_state["query_input"] = s

st.markdown("---")

# Main input
query = st.text_area(
    "✏️ Nhập yêu cầu kịch bản thí nghiệm",
    value=st.session_state.get("query_input", ""),
    height=90,
    placeholder="Ví dụ: Lập kịch bản thí nghiệm xác định thành phần % oxygen trong không khí...",
    key="query_input",
)

run_btn = st.button("🚀 Tạo kịch bản", type="primary")

# ── Run pipeline ─────────────────────────────────────────────────────────────
if run_btn and query.strip():

    # Step 1: RAG retrieval
    with st.spinner("🔍 Đang tìm kiếm tài liệu liên quan..."):
        t0 = time.time()
        try:
            retriever = load_retriever()
            mm_result = retriever.hybrid_retrieve(query, top_k=top_k)
            rag_time = time.time() - t0
            rag_ok = True
        except Exception as e:
            st.error(f"❌ Lỗi RAG: {e}")
            rag_ok = False

    if rag_ok:
        # Source trace display
        st.markdown("#### 📎 Nguồn tài liệu tìm được")
        all_chunks = mm_result.text_chunks + mm_result.image_chunks + mm_result.table_chunks
        all_chunks.sort(key=lambda x: x.score, reverse=True)
        
        if all_chunks:
            cols_src = st.columns(min(len(all_chunks), 4))
            for i, chunk in enumerate(all_chunks[:4]):
                src_type = chunk.modality.upper()
                page = chunk.page_number
                score = chunk.score
                chip_cls = {"TEXT": "chip-text", "IMAGE": "chip-image", "TABLE": "chip-table"}.get(src_type, "chip-text")
                with cols_src[i % 4]:
                    st.markdown(f"""
                    <div class="source-card-container">
                        <span class="source-chip {chip_cls}">{src_type}</span>
                        <span class="score-badge">{score:.3f}</span>
                        <div style='color:#94a3b8; font-size:0.85rem; margin-top:10px'>
                            📄 Trang <b style="color:#f1f5f9">{page}</b><br>
                            <span style='color:#64748b; font-size:0.75rem'>science_db_6.pdf</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        if show_raw_rag:
            with st.expander("📋 RAG context thô"):
                st.text(mm_result.grounded_context[:3000] + ("..." if len(mm_result.grounded_context) > 3000 else ""))

        # Step 2: Generate script
        with st.spinner("✍️ AI đang lập kịch bản thí nghiệm..."):
            t1 = time.time()
            try:
                agent = load_agent()
                result = agent.run(query, context={"rag_top_k": top_k})
                gen_time = time.time() - t1
                gen_ok = True
            except Exception as e:
                st.error(f"❌ Lỗi tạo kịch bản: {e}")
                gen_ok = False

        if gen_ok:
            # Metrics row
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card">
                    <div class="val">{len(all_chunks)}</div>
                    <div class="lbl">Chunks retrieved</div>
                </div>
                <div class="metric-card">
                    <div class="val">{rag_time:.1f}s</div>
                    <div class="lbl">RAG time</div>
                </div>
                <div class="metric-card">
                    <div class="val">{gen_time:.1f}s</div>
                    <div class="lbl">Gen time</div>
                </div>
                <div class="metric-card">
                    <div class="val">{len(result.split())} từ</div>
                    <div class="lbl">Output length</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Result card
            st.markdown("#### 📝 Kịch bản thí nghiệm")
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(result)
            st.markdown('</div>', unsafe_allow_html=True)

            # Download button
            st.download_button(
                "⬇️ Tải kịch bản (.md)",
                data=result,
                file_name=f"kich_ban_{query[:30].replace(' ', '_')}.md",
                mime="text/markdown",
            )

elif run_btn and not query.strip():
    st.warning("⚠️ Vui lòng nhập yêu cầu trước khi tạo kịch bản.")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:rgba(255,255,255,0.2); font-size:0.75rem'>
    A20 Virtual Lab · Powered by GPT-4o Vision RAG + GPT-4o-mini Scripting
</div>
""", unsafe_allow_html=True)
