from .sql_tool import SQLDatabaseService
from .rag_tool import PolicyRAGService
from .python_tool import PythonChartService

sql_service = SQLDatabaseService()
rag_service = PolicyRAGService()
python_service = PythonChartService()

insight_tools = [
    sql_service.get_tool(),
    rag_service.get_tool(),
    python_service.get_tool()
]