from abc import ABC, abstractmethod
from langchain_core.tools import StructuredTool

class BaseToolService(ABC):
    """
    Lớp base cho tất cả các công cụ của Insight Agent.
    """
    
    @abstractmethod
    def get_tool(self) -> StructuredTool:
        """
        Các class con override hàm này và trả về một LangChain StructuredTool.
        """
        pass