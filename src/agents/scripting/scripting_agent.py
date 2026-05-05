"""
src/agents/scripting/scripting_agent.py

Scripting agent responsible for writing comprehensive guides and step-by-step procedures for laboratory experiments.
"""

import logging
import os
import sys
from typing import Any, Optional

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.llm.provider import get_provider
from src.agents.base.base_agent import BaseAgent
from src.agents.scripting.rag import ScriptingScienceRetriever, MultimodalRetriever

logger = logging.getLogger(__name__)


class ScriptingAgent(BaseAgent):
    """
    Agent responsible for writing and formatting step-by-step educational guides,
    safety procedures, and theoretical background for science experiments.
    """

    _NO_DATA_MESSAGE = "I don't have enough scientific data"
    _SYSTEM_PROMPT = (
        "You are scripting_agent in a multi-agent virtual lab system for middle school science. "
        "Your goal is to write a prompt for a p5.js developer agent and a scenario for the student.\n\n"
        "STRICT GROUNDING RULES:\n"
        "1) Use ONLY facts present in the section 'Retrieved Scientific Context' for educational content.\n"
        "2) If the retrieved context lacks explicit numeric parameters (e.g. temperatures, thermal conductivity coefficients, velocities) needed to build a working p5.js physics simulation, you MUST suggest standard real-world physics values (e.g., Copper heat conductivity = 401 W/mK) and clearly mark them as [Standard Simulation Values]. Never invent educational facts.\n"
        "3) EXHAUSTIVE EXTRACTION: You MUST extract EVERY SINGLE scientific fact, property, numeric data, percentage, and constraint from the context.\n"
        "4) FORCE SIMULATOR COMPLIANCE: You MUST explicitly instruct the p5.js developer to visually and functionally demonstrate ALL extracted facts. Nothing can be left out.\n"
        "5) Always include page numbers in your citations (e.g., [Text 1, Trang 50]) directly next to the fact or step they support.\n"
        "6) If context is missing or insufficient for the teacher request, output exactly this template in Vietnamese:\n"
        "Rất tiếc, chủ đề này hiện không có trong dữ liệu sách giáo khoa Khoa học Tự nhiên 6. \n\n"
        "Hệ thống Virtual Lab hiện hỗ trợ tốt nhất các chủ đề:\n"
        "- Các phép đo (Chiều dài, Khối lượng, Thời gian, Nhiệt độ)\n"
        "- Oxygen và Không khí\n"
        "- Vật liệu, nhiên liệu, nguyên liệu, lương thực - thực phẩm\n"
        "- Chất tinh khiết - Hỗn hợp và Tách chất\n"
        "- Tế bào và Đa dạng thế giới sống\n"
        "- Lực, Năng lượng, Trái Đất và Bầu trời\n"
        "Vui lòng thử lại với một yêu cầu liên quan đến các chủ đề trên.\n\n"
        "OUTPUT REQUIREMENTS (Strict Markdown format):\n"
        "## Experiment: [Tên thí nghiệm dựa trên RAG]\n\n"
        "**Topic:** [Chủ đề]\n"
        "**File:** `sketch.js`\n"
        "**Keywords:** [Keywords]\n\n"
        "### Prompt\n\n"
        "**Role:**  \n"
        "Bạn là một lập trình viên chuyên về p5.js và mô phỏng vật lý 2D.\n\n"
        "**Task:**  \n"
        "Viết code p5.js để mô phỏng [tóm tắt nội dung]. BẮT BUỘC phải mô phỏng trực quan và logic TẤT CẢ các đặc tính khoa học sau đây: [Liệt kê TOÀN BỘ các sự thật/số liệu đã trích xuất].\n\n"
        "**Yêu cầu chi tiết:**\n\n"
        "1. **Canvas & Môi trường:** [Mô tả]\n"
        "2. **Hệ thống & Đối tượng:** [Thiết kế các đối tượng phải phản ánh chính xác kích thước, tính chất từ dữ liệu SGK]\n"
        "3. **Tương tác & Vật lý:** [Ép code p5.js phải có logic thể hiện MỌI thông số, tỉ lệ, hiện tượng khoa học (vd: tốc độ dẫn nhiệt khác nhau, tỉ lệ 21% oxygen...)]\n"
        "4. **Data Visualization (HUD):** [Yêu cầu p5.js vẽ các biểu đồ, in text, bảng hiển thị thông số khoa học trực tiếp lên màn hình canvas để học sinh thấy rõ sự thay đổi định lượng]\n"
        "5. **Điều khiển:** [Phím/Chuột]\n\n"
        "**Output format:**\n"
        "- Trả về duy nhất code JavaScript hoàn chỉnh chạy được trong p5.js\n"
        "- Không giải thích, không thêm text ngoài code\n\n"
        "**Mục tiêu:**  \n"
        "[Mục tiêu kèm số liệu cụ thể nếu có]\n\n"
        "### Scenario\n"
        "Trong thí nghiệm này, bạn sẽ thấy [Mô tả].\n\n"
        "**Cách thực hiện:**\n"
        "[Các bước có trích dẫn số trang và số liệu]\n\n"
        "**Điều bạn sẽ học:**\n"
        "[Kiến thức thu được, BẮT BUỘC liệt kê lại toàn bộ các facts/số liệu từ SGK]\n\n"
        "## Grounding notes (map claims → source chunks + page numbers)\n"
        "- [Liệt kê trích dẫn]\n\n"
        "When [Image N] blocks appear in context: describe the apparatus, "
        "phenomena, or diagram they depict to define the p5.js visual requirements."
    )

    def __init__(
        self,
        llm_model: Optional[str] = None,
        retriever: Optional[ScriptingScienceRetriever] = None,
        use_multimodal: bool = False,
        multimodal_retriever: Optional[MultimodalRetriever] = None,
    ) -> None:
        self.chain = None
        self.retriever = None
        self.use_multimodal = use_multimodal
        self.mm_retriever = None

        try:
            # Resolve provider/model from per-agent env configuration.
            resolved_model = llm_model or os.getenv("SCRIPTING_LLM_MODEL") or "gpt-4o"
            provider = get_provider(
                backend=os.getenv("SCRIPTING_PROVIDER_BACKEND"),
                model=resolved_model,
                temperature=float(os.getenv("SCRIPTING_TEMPERATURE", "0.2")),
                max_tokens=int(os.getenv("SCRIPTING_MAX_TOKENS", "4096")),
                timeout=float(os.getenv("SCRIPTING_TIMEOUT", "300")),
            )
            llm = ChatOpenAI(
                model=provider.model,
                api_key=provider.api_key,
                base_url=provider.base_url,
                temperature=provider.temperature,
                max_tokens=provider.max_tokens,
                max_retries=3,
                timeout=provider.timeout,
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self._SYSTEM_PROMPT),
                    (
                        "human",
                        "Teacher Request:\n{query}\n\n"
                        "Retrieved Scientific Context (authoritative source):\n{retrieved_context}\n\n"
                        "Retrieved Source Trace:\n{retrieved_sources}\n\n"
                        "Additional Orchestration Context:\n{context_notes}\n\n"
                        "If context is insufficient, output exactly: I don't have enough scientific data",
                    ),
                ]
            )
            self.chain = prompt | llm | StrOutputParser()
        except Exception as exc:
            logger.warning(f"ScriptingAgent: failed to initialize chain: {exc}")

        if use_multimodal:
            try:
                self.mm_retriever = multimodal_retriever or MultimodalRetriever()
            except Exception as exc:
                logger.warning(
                    f"ScriptingAgent: failed to initialize MultimodalRetriever: {exc}"
                )
        else:
            try:
                self.retriever = retriever or ScriptingScienceRetriever()
            except Exception as exc:
                logger.warning(f"ScriptingAgent: failed to initialize retriever: {exc}")

    def run(
        self,
        query: str,
        context: Optional[dict[str, Any]] = None,
        messages: Optional[list[BaseMessage]] = None,
    ) -> str:
        if not self.chain:
            return "ScriptingAgent is not properly configured (chain missing)."

        context = context or {}
        context_notes = "\n".join(f"{k}: {v}" for k, v in context.items()) if context else "None"
        top_k = self._resolve_top_k(context)

        use_multimodal = getattr(self, "use_multimodal", False)
        mm_retriever = getattr(self, "mm_retriever", None)
        retriever = getattr(self, "retriever", None)

        # ── Multimodal path ────────────────────────────────────────────────
        if use_multimodal:
            if not mm_retriever:
                return self._NO_DATA_MESSAGE
            try:
                mm_result = mm_retriever.hybrid_retrieve(query, top_k=top_k)
                if not mm_result.has_enough_data:
                    return self._NO_DATA_MESSAGE
                return self.chain.invoke(
                    {
                        "query": query,
                        "context_notes": context_notes,
                        "retrieved_context": mm_result.grounded_context,
                        "retrieved_sources": mm_result.source_report,
                    }
                )
            except Exception as exc:
                logger.error(f"ScriptingAgent multimodal error: {exc}")
                return f"Guide generation failed: {exc}"

        # ── Legacy text-only path ──────────────────────────────────────────
        if not retriever:
            return self._NO_DATA_MESSAGE

        try:
            retrieval = retriever.retrieve(query, top_k=top_k)
            if not retrieval.has_enough_data:
                return self._NO_DATA_MESSAGE

            return self.chain.invoke(
                {
                    "query": query,
                    "context_notes": context_notes,
                    "retrieved_context": retrieval.grounded_context,
                    "retrieved_sources": retrieval.source_report,
                }
            )
        except Exception as exc:
            logger.error(f"ScriptingAgent error: {exc}")
            return f"Guide generation failed: {exc}"

    @staticmethod
    def _resolve_top_k(context: dict[str, Any]) -> int:
        raw_value = context.get("rag_top_k", 4)
        try:
            top_k = int(raw_value)
        except (TypeError, ValueError):
            return 4
        return max(1, min(top_k, 8))


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    if len(sys.argv) < 2:
        print("Sử dụng: python -m src.agents.scripting.scripting_agent \"Yêu cầu của bạn\"")
        sys.exit(1)
        
    user_query = sys.argv[1]
    
    # Khởi tạo Agent với chế độ Multimodal (có RAG cho scan)
    agent = ScriptingAgent(use_multimodal=True)
    
    print(f"\n🔍 Đang truy vấn RAG và lập kịch bản cho: '{user_query}'\n")
    result = agent.run(user_query)
    
    print("-" * 50)
    print(result)
    print("-" * 50)
