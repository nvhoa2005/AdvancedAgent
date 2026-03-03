from langchain_community.utilities import SQLDatabase
from langchain_core.tools import StructuredTool
from config.settings import settings

class SQLDatabaseService:
    def __init__(self):
        self.db = SQLDatabase.from_uri(settings.DATABASE_URL)

    def query_sql_db(self, query: str) -> str:
        """Thực thi lệnh SQL truy vấn Database."""
        try:
            query = query.strip().rstrip(";")
            if "limit" not in query.lower():
                query += " LIMIT 20"
                
            print(f"[SQL Tool] Running: {query}")
            return self.db.run(query)
        except Exception as e:
            return f"Lỗi SQL: {str(e)}. Hãy kiểm tra lại cú pháp hoặc tên bảng."

    def get_db_schema(self) -> str:
        """Lấy schema để nhúng vào System Prompt."""
        return self.db.get_table_info()

    def get_tool(self) -> StructuredTool:
        """Trả về LangChain Tool để bind vào LLM."""
        return StructuredTool.from_function(
            func=self.query_sql_db,
            name="query_sql_db",
            description=(
                "Công cụ thực thi lệnh SQL truy vấn Database bán hàng. "
                "Chỉ sử dụng công cụ này khi cần lấy số liệu chính xác từ các bảng: "
                "customers, products, orders, order_items, inventory. "
                "Input: Câu lệnh SQL hợp lệ (PostgreSQL). "
                "Output: Kết quả truy vấn dạng text."
            )
        )