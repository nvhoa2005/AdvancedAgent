import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Class quản lý toàn bộ cấu hình của hệ thống."""
    
    # Database & API Keys
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY")
    
    # Models Configuration
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0
    WRITER_TEMPERATURE: float = 0.2
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    RERANK_MODEL: str = "rerank-v3.5"
    
    # Vector Store
    COLLECTION_NAME: str = "company_policies"

    def __init__(self):
        self._validate_settings()

    def _validate_settings(self):
        if not self.DATABASE_URL:
            raise ValueError("Lỗi: Chưa cấu hình DATABASE_URL trong file .env")
        if not self.COHERE_API_KEY:
            raise ValueError("Lỗi: Chưa cấu hình COHERE_API_KEY trong file .env")

settings = Settings()