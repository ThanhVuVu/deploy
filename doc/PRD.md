# PRD - Hệ thống Multi-Agent Virtual Science Lab

## 1. Tổng quan sản phẩm (Overview)

**Tên sản phẩm:** AI Virtual Science Lab for Secondary Education v1.0

**Mô tả ngắn:** Hệ thống sử dụng multi-agent AI để hỗ trợ giáo viên THCS tạo, chuẩn hóa, triển khai và đánh giá thí nghiệm ảo, giúp học sinh thực hành nhiều hơn trong môi trường mô phỏng an toàn và có kiểm soát.

## 2. Đặt vấn đề (Problem Statement)

### 2.1 Pain points phía học sinh

- ~60% trường thiếu phòng thí nghiệm đạt chuẩn
- Học sinh chỉ thực hiện 5–8 thí nghiệm/năm (so với mục tiêu 20+)
- ~30% học sinh dưới trung bình phần thực hành
- Thiếu trải nghiệm trực quan, kỹ năng quan sát và phản hồi kịp thời

### 2.2 Pain points phía giáo viên

- Khó tổ chức thí nghiệm do thiếu thiết bị và hạn chế thời gian trên lớp
- Khó theo dõi chi tiết quá trình thực hành của từng học sinh
- Khó cá nhân hóa hướng dẫn
- Khó phát hiện học sinh sai ở thao tác hay sai ở nhận thức

## 3. Mục tiêu (Goals)

### 3.1. Mục tiêu sản phẩm

- Số hóa 30+ thí nghiệm trọng tâm THCS
- Giúp mỗi học sinh thực hiện ít nhất 15 thí nghiệm/năm

### 3.2. Mục tiêu giáo dục

- Tăng khả năng quan sát
- Tăng tư duy thực nghiệm
- Tăng mức độ hiểu bản chất khoa học

### 3.3. Mục tiêu hệ thống AI

- Hỗ trợ giáo viên tạo thí nghiệm nhanh, đúng chuẩn
- Kiểm soát nội dung trước khi giao cho học sinh
- Phân tích hành vi học sinh từ log tương tác
- Cung cấp feedback cá nhân hóa và insight cho giáo viên

## 4. User Personas

### 4.1 Primary User - Giáo viên

- Tạo, chỉnh sửa, duyệt thí nghiệm
- Giao bài cho học sinh
- Xem báo cáo và đánh giá

### 4.2. Secondary User - Học sinh

- Thực hiện thí nghiệm
- Tương tác trong môi trường mô phỏng
- Trả lời câu hỏi và nhận feedback

## 5. Workflow

### 5.1. Teacher Flow

- Giáo viên nhập đề bài thí nghiệm hoặc mục tiêu bài học
- Hệ thống multi-agent phân tích yêu cầu và sinh ra kịch bản thí nghiệm
- Giáo viên duyệt hoặc chỉnh sửa thí nghiệm
- Hệ thống triển khai thí nghiệm cho học sinh

### 5.2. Student Flow

- Học sinh mở thí nghiệm được giao
- Thực hiện các bước theo hướng dẫn
- Tương tác bằng cách thay đổi biến hoặc thao tác trong phạm vi cho phép
- Quan sát hiện tượng và trả lời câu hỏi

### 5.3. Feedback Flow

- Hệ thống ghi nhận log thao tác, thời gian và kết quả
- Multi-agent phân tích lỗi thao tác và lỗi nhận thức
- Hệ thống sinh báo cáo cho giáo viên theo cá nhân và theo lớp

## 6. Core Capabilities & Multi-Agent Support

Hệ thống tập trung vào 4 năng lực chính:

### 6.1. Hỗ trợ giáo viên tạo thí nghiệm

Giáo viên nhập đề bài hoặc mục tiêu bài học, hệ thống phân tích yêu cầu và sinh ra một kịch bản thí nghiệm ảo có cấu trúc.

### 6.2. Kiểm tra và chuẩn hóa nội dung

Hệ thống kiểm tra tính phù hợp của thí nghiệm theo chương trình THCS, tính đúng đắn khoa học và mức độ khả thi trước khi đưa cho giáo viên duyệt

### 6.3. Triển khai mô phỏng cho học sinh

Sau khi giáo viên duyệt, học sinh có thể thực hiện thí nghiệm trong môi trường mô phỏng với các thao tác và biến số được kiểm soát

### 6.4. Phân tích log và hỗ trợ giáo viên đánh giá

Hệ thống ghi nhận quá trình tương tác của học sinh, từ đó tổng hợp lỗi phổ biến, mức độ hoàn thành và tạo feedback hỗ trợ giáo viên.

### 6.5. Vai trò của Multi-agent

Multi-agent được sử dụng để chia hệ thống thành các bước chuyên biệt như:

- phân tích yêu cầu của giáo viên,
- tạo kịch bản thí nghiệm,
- kiểm tra tính đúng đắn,
- phân tích hành vi học sinh,
- tổng hợp phản hồi cho giáo viên.

## 7. Metrics

### 7.1. Learning Metrics

- ≥70% học sinh cải thiện điểm thực hành
- ≥50% giảm lỗi hiểu sai khái niệm

### 7.2. Usage Metrics

- ≥15 thí nghiệm/học sinh/năm
- Thời gian tương tác trung bình mỗi session > 10 phút

### 7.3. System Metrics

- Latency phản hồi < 3 giây
- Tỉ lệ nội dung thí nghiệm được giáo viên chấp nhận ở mức cao
- Độ chính xác ngữ nghĩa và logic khoa học > 85%

## 8. Constraints & Risks

### 8.1. AI Risks

- Hallucination
- Sai logic khoa học
- Sinh feedback chưa đủ chính xác

### 8.2. Product Risks

- UX quá phức tạp với giáo viên hoặc học sinh
- Giáo viên không tin tưởng nội dung do AI sinh ra

### 8.3. Technical Constraints

- Trade-off giữa độ chi tiết mô phỏng và tốc độ phản hồi
- Chi phí inference nếu multi-agent quá nặng
