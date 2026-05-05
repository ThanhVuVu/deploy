"""
Streamlit UI for the full Virtual Lab pipeline.

Run:
    streamlit run demo_virtual_lab.py
"""

from __future__ import annotations

import html
import re
import time
import unicodedata
import zipfile
from io import BytesIO
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from src.agents.scripting.rag import MultimodalIndexer, MultimodalRetriever, ScriptingScienceRetriever
from src.agents.scripting.scripting_agent import ScriptingAgent
from src.agents.simulator import SimulatorAgent, SimulatorArtifacts, SimulatorOutputError
from src.pipeline.experiment_pipeline import ExperimentGenerationPipeline


load_dotenv()

APP_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = APP_ROOT / "generated_labs"


st.set_page_config(
    page_title="Virtual Lab Generator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #0b1020;
  --panel: #111827;
  --panel-soft: #172033;
  --line: #273449;
  --text: #e5eefb;
  --muted: #8ea3bd;
  --blue: #38bdf8;
  --green: #34d399;
  --amber: #fbbf24;
  --red: #fb7185;
}

html, body, [class*="css"] {
  font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
  background:
    radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 32rem),
    linear-gradient(180deg, #0b1020 0%, #0c1222 45%, #0b1020 100%);
}

[data-testid="stSidebar"] {
  background: #0f172a;
  border-right: 1px solid var(--line);
}

.hero {
  padding: 1.5rem 0 1rem;
  border-bottom: 1px solid var(--line);
  margin-bottom: 1.25rem;
}

.hero h1 {
  margin: 0;
  color: var(--text);
  font-size: 2.25rem;
  line-height: 1.05;
  font-weight: 800;
}

.hero p {
  margin: .6rem 0 0;
  color: var(--muted);
  font-size: 1rem;
}

.status-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: .75rem;
  margin: 1rem 0;
}

.status-card {
  background: rgba(17, 24, 39, .86);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: .9rem 1rem;
}

.status-card .label {
  color: var(--muted);
  font-size: .75rem;
  text-transform: uppercase;
  letter-spacing: .06em;
  font-weight: 700;
}

.status-card .value {
  color: var(--text);
  font-size: 1.2rem;
  font-weight: 800;
  margin-top: .3rem;
}

.file-box {
  background: rgba(17, 24, 39, .86);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 1rem;
  color: var(--text);
}

.file-path {
  color: #bae6fd;
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  word-break: break-all;
}

.source-box {
  background: #111827;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: .8rem;
  color: var(--muted);
  min-height: 9rem;
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  font-size: .82rem;
  white-space: pre-wrap;
}

.library-panel {
  background: rgba(17, 24, 39, .72);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 1rem;
  margin: .75rem 0 1rem;
}

.library-panel h2 {
  color: var(--text);
  font-size: 1.05rem;
  margin: 0 0 .35rem;
  font-weight: 800;
}

.library-panel p {
  color: #b6c7df;
  margin: 0;
  font-size: .9rem;
}

.stMarkdown,
.stMarkdown p,
.stCaption,
[data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
  color: #d9e6f7 !important;
}

[data-testid="stWidgetLabel"] p {
  font-weight: 700;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
  color: #a9bbd3 !important;
}

.stButton > button {
  background: #162033;
  border: 1px solid #334155;
  color: #e8f1ff;
  font-weight: 700;
  border-radius: 9px;
  min-height: 2.55rem;
}

.stButton > button:hover {
  background: #1e293b;
  border-color: #38bdf8;
  color: white;
}

.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #0284c7, #059669);
  border: 0;
  color: white;
  font-weight: 800;
  border-radius: 9px;
  min-height: 3rem;
}

.stTextArea textarea,
.stTextInput input,
.stSelectbox div[data-baseweb="select"] > div {
  background: #111a2d !important;
  border-color: #334155 !important;
  color: var(--text) !important;
}

.stTextArea textarea:focus,
.stTextInput input:focus {
  border-color: #38bdf8 !important;
  box-shadow: 0 0 0 1px rgba(56, 189, 248, .35) !important;
}

[data-baseweb="select"] span,
[data-baseweb="select"] div {
  color: #e5eefb !important;
}

[data-testid="stAlert"] {
  border-radius: 10px;
  border: 1px solid rgba(52, 211, 153, .28);
}

.stTabs [data-baseweb="tab-list"] {
  gap: .35rem;
}

.stTabs [data-baseweb="tab"] {
  background: #121c30;
  border: 1px solid #263449;
  border-radius: 8px 8px 0 0;
  color: #cbdaf0;
  font-weight: 700;
}

.stTabs [aria-selected="true"] {
  background: #1c2a44 !important;
  color: white !important;
}
</style>
""",
    unsafe_allow_html=True,
)


def slugify(text: str, max_len: int = 42) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text.lower()).strip("-")
    return (slug or "virtual-lab")[:max_len].strip("-") or "virtual-lab"


def build_standalone_html(artifacts: SimulatorArtifacts) -> str:
    """Return the single runnable HTML file for preview/download."""
    return artifacts.single_file_html


def build_zip_bytes(files: dict[str, str]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buffer.getvalue()


def resolve_output_base(output_root_value: str) -> Path:
    output_base = Path(output_root_value)
    if not output_base.is_absolute():
        output_base = APP_ROOT / output_base
    return output_base


def discover_existing_labs(output_root_value: str) -> list[Path]:
    output_base = resolve_output_base(output_root_value)
    if not output_base.exists():
        return []

    html_files = [
        path
        for path in output_base.rglob("*.html")
        if path.is_file() and not path.name.startswith(".")
    ]
    return sorted(html_files, key=lambda path: path.stat().st_mtime, reverse=True)


def format_lab_label(path: Path, output_root_value: str) -> str:
    output_base = resolve_output_base(output_root_value)
    try:
        relative = path.relative_to(output_base)
    except ValueError:
        relative = path
    modified = time.strftime("%H:%M %d/%m/%Y", time.localtime(path.stat().st_mtime))
    return f"{relative.as_posix()}  |  {modified}"


def load_existing_lab_html(path: Path) -> str:
    content = path.read_text(encoding="utf-8")
    sketch_path = path.with_name("sketch.js")
    if "sketch.js" in content and sketch_path.exists():
        return SimulatorArtifacts._inline_sketch(
            content,
            sketch_path.read_text(encoding="utf-8"),
        )
    return content


@st.cache_resource(show_spinner=False)
def load_retriever(use_multimodal: bool):
    if use_multimodal:
        return MultimodalRetriever()
    return ScriptingScienceRetriever()


@st.cache_resource(show_spinner=False)
def load_scripting_agent(use_multimodal: bool):
    retriever = load_retriever(use_multimodal)
    return ScriptingAgent(
        use_multimodal=use_multimodal,
        multimodal_retriever=retriever if use_multimodal else None,
        retriever=retriever if not use_multimodal else None,
    )


def load_simulator_agent(model_name: str | None):
    return SimulatorAgent(llm_model=model_name or None)


def build_pipeline(use_multimodal: bool, simulator_model: str | None):
    retriever = load_retriever(use_multimodal)
    scripting_agent = load_scripting_agent(use_multimodal)
    simulator_agent = load_simulator_agent(simulator_model)
    return ExperimentGenerationPipeline(
        retriever=retriever,
        scripting_agent=scripting_agent,
        simulator_agent=simulator_agent,
        use_multimodal=use_multimodal,
    )


def render_metrics(rag_seconds: float, total_seconds: float, files_count: int, output_dir: Path):
    st.markdown(
        f"""
<div class="status-grid">
  <div class="status-card"><div class="label">RAG time</div><div class="value">{rag_seconds:.1f}s</div></div>
  <div class="status-card"><div class="label">Total time</div><div class="value">{total_seconds:.1f}s</div></div>
  <div class="status-card"><div class="label">Files</div><div class="value">{files_count}</div></div>
  <div class="status-card"><div class="label">Output</div><div class="value">{html.escape(output_dir.name)}</div></div>
</div>
""",
        unsafe_allow_html=True,
    )


with st.sidebar:
    if st.button("Reload config / clear cache", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
    st.markdown("### Cấu hình pipeline")
    use_multimodal = st.toggle("Multimodal RAG", value=True)
    rag_top_k = st.slider("Số chunk RAG", min_value=2, max_value=8, value=6)
    simulator_model = st.text_input(
        "Simulator model",
        value="",
        placeholder="Để trống để dùng SIMULATOR_LLM_MODEL / LLM_MODEL",
    )
    output_root = st.text_input(
        "Thư mục output",
        value=str(DEFAULT_OUTPUT_ROOT.relative_to(APP_ROOT)),
    )
    show_script = st.checkbox("Hiển thị kịch bản trung gian", value=True)
    show_rag = st.checkbox("Hiển thị RAG trace", value=True)

    st.markdown("---")
    st.markdown("### Điều kiện chạy")
    st.caption("Cần `.env` có khóa cho scripting agent và simulator model.")
    st.caption("RAG index cần được ingest trước bằng `python -m src.agents.scripting.ingest_cli --all`.")

    st.markdown("---")
    st.markdown("### Quản lý chỉ mục RAG")
    try:
        # Check if index exists or is empty
        indexer = MultimodalIndexer()
        counts = indexer.collection_counts()
        total_chunks = sum(counts.values())
        st.write(f"Tổng số chunks hiện có: **{total_chunks}**")
        
        if total_chunks == 0:
            st.warning("⚠️ Chỉ mục RAG đang trống (thường do deploy lần đầu).")
        
        if st.button("🚀 Ingest dữ liệu JSON (science_db/)", use_container_width=True):
            with st.spinner("Đang ingest dữ liệu JSON... Quá trình này gọi OpenAI embedding nên có thể mất vài phút."):
                from src.agents.scripting.ingest_cli import cmd_ingest_all_json
                cmd_ingest_all_json(indexer, APP_ROOT / "science_db")
                st.success("Đã ingest xong! Vui lòng tải lại trang.")
                st.rerun()
    except Exception as e:
        st.error(f"Không thể khởi tạo Indexer: {e}")


st.markdown(
    """
<div class="hero">
  <h1>Virtual Lab Generator</h1>
  <p>Nhập prompt của giáo viên, hệ thống tự chạy RAG, tạo kịch bản, sinh simulator p5.js và xuất file HTML thí nghiệm.</p>
</div>
""",
    unsafe_allow_html=True,
)


suggestions = [
    "So sánh nhiệt độ nóng chảy, nhiệt độ sôi và trạng thái phân tử của oxygen, ethanol, nước, thủy ngân và sắt bằng dữ liệu thực nghiệm.",
    "Mô phỏng thí nghiệm xác định thành phần phần trăm oxygen trong không khí.",
    "Tạo thí nghiệm tương tác về ảnh hưởng của áp suất đến nhiệt độ sôi của nước.",
    "Mô phỏng lực hút, đẩy giữa các cực nam châm và độ lệch của la bàn.",
]

st.markdown(
    """
<div class="library-panel">
  <h2>Kho thí nghiệm đã sinh</h2>
  <p>Chọn một file HTML trong generated_labs để chạy lại ngay trong giao diện.</p>
</div>
""",
    unsafe_allow_html=True,
)

existing_labs = discover_existing_labs(output_root)
if existing_labs:
    lab_options = {
        format_lab_label(path, output_root): path
        for path in existing_labs
    }
    selected_label = st.selectbox(
        "Thí nghiệm có sẵn",
        options=list(lab_options.keys()),
        index=0,
    )
    selected_lab_path = lab_options[selected_label]
    selected_lab_html = load_existing_lab_html(selected_lab_path)

    col_preview, col_download, col_path = st.columns([2, 1, 3])
    with col_preview:
        preview_existing = st.button(
            "Chạy thí nghiệm đã chọn",
            use_container_width=True,
        )
    with col_download:
        st.download_button(
            "Tải HTML",
            data=selected_lab_html,
            file_name=selected_lab_path.name,
            mime="text/html",
            use_container_width=True,
        )
    with col_path:
        st.markdown(
            f'<div class="file-path">{html.escape(str(selected_lab_path))}</div>',
            unsafe_allow_html=True,
        )

    if preview_existing:
        st.session_state["preview_existing_lab"] = str(selected_lab_path)

    if st.session_state.get("preview_existing_lab") == str(selected_lab_path):
        st.caption("Đang chạy thí nghiệm từ kho generated_labs.")
        components.html(selected_lab_html, height=820, scrolling=True)
else:
    st.info("Chưa tìm thấy file HTML nào trong `generated_labs/`.")

cols = st.columns(4)
for idx, suggestion in enumerate(suggestions):
    if cols[idx].button(f"Gợi ý {idx + 1}", use_container_width=True):
        st.session_state["teacher_prompt"] = suggestion

teacher_prompt = st.text_area(
    "Prompt giáo viên",
    key="teacher_prompt",
    height=130,
    placeholder="Ví dụ: So sánh nhiệt độ sôi và trạng thái phân tử của oxygen, ethanol, nước, thủy ngân và sắt...",
)

run_btn = st.button("Chạy toàn bộ pipeline", type="primary", use_container_width=True)

if run_btn and not teacher_prompt.strip():
    st.warning("Nhập prompt giáo viên trước khi chạy pipeline.")

if run_btn and teacher_prompt.strip():
    output_base = resolve_output_base(output_root)

    experiment_slug = slugify(teacher_prompt)
    file_stamp = time.strftime("%H%M%S-%Y%m%d")
    output_filename = f"{experiment_slug}-{file_stamp}.html"
    output_dir = output_base

    status = st.status("Đang khởi tạo pipeline...", expanded=True)
    started_at = time.time()

    try:
        status.write("Đang nạp retriever và agents...")
        pipeline = build_pipeline(use_multimodal, simulator_model.strip() or None)

        status.write("Đang truy xuất RAG để kiểm tra nguồn dữ liệu...")
        rag_t0 = time.time()
        rag_snapshot = pipeline.retrieve(teacher_prompt, top_k=rag_top_k)
        rag_seconds = time.time() - rag_t0
        if not rag_snapshot.has_enough_data:
            raise SimulatorOutputError("RAG không tìm thấy đủ dữ liệu phù hợp cho prompt này.")

        status.write("Đang tạo kịch bản thí nghiệm từ RAG...")
        status.write("Đang gọi simulator agent để sinh một file HTML duy nhất...")
        result = pipeline.run(
            teacher_prompt,
            output_dir=output_dir,
            output_filename=output_filename,
            rag_top_k=rag_top_k,
            context={"ui": "streamlit"},
            write_files=True,
        )

        standalone_html = build_standalone_html(result.artifacts)
        standalone_path = result.written_files[0] if result.written_files else output_dir / output_filename

        total_seconds = time.time() - started_at
        status.update(label="Pipeline hoàn tất", state="complete", expanded=False)

        st.success("Đã sinh thí nghiệm HTML.")
        render_metrics(
            rag_seconds=rag_seconds,
            total_seconds=total_seconds,
            files_count=len(result.written_files),
            output_dir=output_dir,
        )

        st.markdown(
            f"""
<div class="file-box">
  <strong>Thư mục đã ghi:</strong><br>
  <span class="file-path">{html.escape(str(output_dir))}</span><br><br>
  <strong>File chính:</strong><br>
  <span class="file-path">{html.escape(str(standalone_path))}</span>
</div>
""",
            unsafe_allow_html=True,
        )

        tab_preview, tab_files, tab_script, tab_sources = st.tabs(
            ["Preview", "Files", "Kịch bản", "RAG"]
        )

        with tab_preview:
            st.caption("Preview dùng file HTML duy nhất đã inline toàn bộ mã mô phỏng.")
            components.html(standalone_html, height=820, scrolling=True)

        with tab_files:
            zip_bytes = build_zip_bytes({standalone_path.name: standalone_html})
            st.download_button(
                "Tải toàn bộ thí nghiệm (.zip)",
                data=zip_bytes,
                file_name=f"{standalone_path.stem}.zip",
                mime="application/zip",
                use_container_width=True,
            )
            st.download_button(
                "Tải file HTML thí nghiệm",
                data=standalone_html,
                file_name=standalone_path.name,
                mime="text/html",
                use_container_width=True,
            )
            st.markdown(f"#### {standalone_path.name}")
            st.code(standalone_html, language="html")

        with tab_script:
            if show_script:
                st.markdown(result.experiment_script)
            else:
                st.info("Bật tuỳ chọn hiển thị kịch bản trung gian ở sidebar.")

        with tab_sources:
            if show_rag:
                st.markdown("#### Source trace")
                st.markdown(
                    f'<div class="source-box">{html.escape(result.rag_snapshot.source_report)}</div>',
                    unsafe_allow_html=True,
                )
                with st.expander("Grounded context"):
                    st.text(result.rag_snapshot.grounded_context)
            else:
                st.info("Bật tuỳ chọn hiển thị RAG trace ở sidebar.")

    except Exception as exc:
        status.update(label="Pipeline thất bại", state="error", expanded=True)
        st.error(f"Lỗi: {exc}")
        st.exception(exc)

else:
    st.info("Nhập prompt rồi bấm chạy. Kết quả sẽ được ghi vào `generated_labs/` và hiển thị preview ở đây.")
