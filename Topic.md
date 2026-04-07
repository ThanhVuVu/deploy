# 🧪 Thí Nghiệm Ảo — Multi-Agent Virtual Science Lab

[![Lĩnh vực: K-12](https://img.shields.io/badge/Lĩnh%20vực-K--12-blue.svg)](https://en.wikipedia.org/wiki/K%E2%80%9312)
[![Kỹ thuật: Multi-agent](https://img.shields.io/badge/Kỹ%20thuật-Multi--agent-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Trạng thái: MVP](https://img.shields.io/badge/Trạng%20thái-MVP%20Ready-green.svg)]()

---

## 📌 Thông tin dự án
* **Mã đề:** `AI20K-049`
* **Lĩnh vực:** Giáo dục phổ thông (K-12)
* **Kỹ thuật trọng tâm:** Hệ thống đa tác nhân (Multi-agent System)
* **Mục tiêu:** Giải quyết tình trạng thiếu hụt cơ sở vật chất thí nghiệm tại các trường THCS Việt Nam.

---

## 🎯 Bối cảnh & Bài toán
Dựa trên thực trạng giáo dục hiện nay:
* **Thiếu hụt:** ~60% trường THCS thiếu phòng Lab đạt chuẩn.
* **Hạn chế:** Học sinh chỉ thực hiện 5-8 thí nghiệm/năm (trong khi chương trình yêu cầu 20+).
* **Hệ quả:** Điểm thi thực hành trung bình thấp hơn 30% so với lý thuyết.

---

## 🤖 Kiến trúc Multi-Agent
Hệ thống được vận hành bởi 3 Agent chuyên biệt phối hợp qua **LangGraph**:

| Agent | Vai trò chính | Chức năng chi tiết |
| :--- | :--- | :--- |
| **🧪 Simulator** | Mô phỏng | Thực hiện 30+ thí nghiệm, thay đổi biến số, quan sát kết quả và thu thập dữ liệu thời gian thực. |
| **📘 Guide** | Hướng dẫn | Chỉ dẫn từng bước (step-by-step), cảnh báo an toàn và cung cấp kiến thức nền tảng (theory background). |
| **📝 Evaluator** | Đánh giá | Chấm điểm báo cáo theo Rubric, cung cấp phản hồi chi tiết và trực quan hóa (visualization) kết quả. |

---

## 🛠 Tech Stack
Dự án sử dụng các công nghệ hiện đại nhất để đảm bảo tính ổn định và khả năng mở rộng:

* **LLM & Orchestration:** OpenAI/Anthropic API, **LangGraph** (Multi-agent workflow).
* **Simulation & Math:** Python (SciPy, Matplotlib), NumPy.
* **Frontend & Visualization:** **Streamlit**, Plotly, Three.js (cho mô phỏng 3D).
* **Backend & Database:** Python, **PostgreSQL**.
* **Deployment:** **Railway** (Deployed Online với URL công khai).

---

## 🏗️ Yêu cầu MVP (Minimum Viable Product)
Để đạt chuẩn nghiệm thu, sản phẩm phải đáp ứng:

### ✅ Yêu cầu bắt buộc:
- [x] **Deployed Online:** Có URL truy cập chính thức (không chạy localhost).
- [x] **Quản lý người dùng:** Đăng nhập, đăng ký và phân quyền cơ bản.
- [x] **UI/UX Hoàn chỉnh:** Giao diện web thân thiện với học sinh và giáo viên.
- [x] **Hệ thống Agent:** Luồng tương tác mượt mà giữa Simulator, Guide và Evaluator.

### ⛔ Không chấp nhận:
- Demo dưới dạng Notebook (.ipynb).
- Script chạy qua dòng lệnh (CLI).
- Prototype chỉ hoạt động trên môi trường local.

---

## 📅 Roadmap phát triển
1.  **Giai đoạn 1:** Xây dựng core logic cho 10 thí nghiệm Vật lý lớp 6 (Chương trình GDPT 2018).
2.  **Giai đoạn 2:** Hoàn thiện luồng Multi-agent với LangGraph.
3.  **Giai đoạn 3:** Triển khai Cloud và kiểm thử bảo mật.

---
© 2026 - Dự án AI20K-049 | Virtual Science Lab Team
