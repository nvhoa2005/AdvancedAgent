import os
import ast
import re
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_experimental.utilities import PythonREPL

load_dotenv()

db_uri = os.getenv("DATABASE_URL")
if not db_uri:
    raise ValueError("Chưa cấu hình DATABASE_URL trong file .env")

db = SQLDatabase.from_uri(db_uri)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="company_policies",
    connection=db_uri,
    use_jsonb=True,
)

@tool
def query_sql_db(query: str) -> str:
    """
    Công cụ thực thi lệnh SQL truy vấn Database bán hàng.
    Chỉ sử dụng công cụ này khi cần lấy số liệu chính xác từ các bảng: 
    customers, products, orders, order_items, inventory.
    Input: Câu lệnh SQL hợp lệ (PostgreSQL).
    Output: Kết quả truy vấn dạng text.
    """
    try:
        
        query = query.strip().rstrip(";")
        
        if "limit" not in query.lower():
            query += " LIMIT 20"
            
        print(f"[SQL Tool] Running: {query}")
        return db.run(query)
    except Exception as e:
        return f"Lỗi SQL: {str(e)}. Hãy kiểm tra lại cú pháp hoặc tên bảng."

def get_db_schema() -> str:
    """Hàm phụ trợ giúp AI biết cấu trúc bảng"""
    return db.get_table_info()

@tool
def search_policy_docs(query: str) -> str:
    """
    Công cụ tìm kiếm thông tin trong tài liệu chính sách công ty (PDF).
    Sử dụng khi người dùng hỏi về: lương, thưởng, nghỉ phép, quy định, phúc lợi...
    Input: Từ khóa hoặc câu hỏi tìm kiếm.
    Output: Các đoạn văn bản liên quan nhất.
    """
    print(f"[RAG Tool] Searching: {query}")
    docs = vector_store.similarity_search(query, k=4)
    
    context = "\n\n".join([d.page_content for d in docs])
    return context if context else "Không tìm thấy thông tin trong tài liệu."

@tool
def python_chart_maker(code: str) -> str:
    """
    Công cụ chạy code Python để phân tích dữ liệu hoặc vẽ biểu đồ.
    Sử dụng thư viện: matplotlib, pandas.
    Input: Đoạn code Python hợp lệ.
    Output: Kết quả chạy code hoặc thông báo đã lưu ảnh.
    
    Lưu ý quan trọng cho AI: 
    - Luôn lưu biểu đồ vào đường dẫn 'static/chart_output.png'.
    - Không dùng plt.show().
    """
    print("[Chart Tool] Executing Python code...")
    
    os.makedirs('static', exist_ok=True)
    
    wrapped_code = f"""
import matplotlib.pyplot as plt
import pandas as pd
import os

# Code của AI bắt đầu
{code}
# Code của AI kết thúc

# Force save file
if plt.get_fignums():
    save_path = 'static/chart_output.png'
    if os.path.exists(save_path):
        os.remove(save_path)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"SUCCESSS_CHART_SAVED: {{save_path}}")
else:
    print("NO_CHART_CREATED")
"""
    repl = PythonREPL()
    try:
        result = repl.run(wrapped_code)
        
        if "SUCCESSS_CHART_SAVED" in result:
            return "Đã vẽ biểu đồ thành công và lưu tại 'static/chart_output.png'. Hãy hiển thị nó cho người dùng."
        elif "NO_CHART_CREATED" in result:
            return f"Code đã chạy nhưng không tạo ra biểu đồ. Output: {result}"
        else:
            return f"Kết quả chạy code: {result}"
            
    except Exception as e:
        return f"Lỗi Python: {str(e)}"