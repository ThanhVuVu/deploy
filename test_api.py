import sys
import os
import base64
import json
from pathlib import Path
from openai import OpenAI
sys.path.insert(0, os.getcwd())
from src.agents.scripting.rag.pdf_parser import PDFParser

parser = PDFParser()
parsed = parser.parse(Path('science_db/science_db_7.pdf'))
img = parsed.images[1]  # index 1 is the 2nd image (page 2)

b64 = base64.b64encode(img.image_bytes).decode('ascii')
client = OpenAI()

prompt = """Bạn là chuyên gia số hóa sách giáo khoa Khoa học Tự nhiên THCS Việt Nam.
Phân tích trang sách scan này, trả về JSON đúng cấu trúc sau:
{
  "texts":  [{"content": "toàn bộ văn bản lý thuyết, định nghĩa, bài tập..."}],
  "tables": [{"markdown": "| cột1 | cột2 |\\n|---|---|\\n| gt | gt |"}],
  "images": [{"description": "mô tả chi tiết hình vẽ/sơ đồ thí nghiệm..."}]
}
Quy tắc BẮT BUỘC:
1. OCR chính xác tiếng Việt có đầy đủ dấu (ưu tiên số 1).
2. Mỗi bảng biểu → chuyển thành Markdown table chuẩn.
3. Mỗi hình vẽ/sơ đồ → mô tả 3-5 câu nêu rõ dụng cụ, hiện tượng, cấu tạo.
4. Không có bảng → tables: [].  Không có hình → images: [].
CHỈ trả về JSON, không thêm bất kỳ text nào khác."""

response = client.chat.completions.create(
    model='gpt-4o', 
    messages=[
        {'role': 'user', 'content': [
            {'type': 'text', 'text': prompt}, 
            {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}}
        ]}
    ], 
    response_format={'type': 'json_object'}, 
    temperature=0.0
)
print('Finish Reason:', response.choices[0].finish_reason)
print('Content:', response.choices[0].message.content)
print('Refusal:', response.choices[0].message.refusal)
