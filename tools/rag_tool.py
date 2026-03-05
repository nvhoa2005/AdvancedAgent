import cohere
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.tools import StructuredTool
from config.settings import settings
from .base_tool import BaseToolService

class PolicyRAGService(BaseToolService):
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=settings.COLLECTION_NAME,
            connection=settings.DATABASE_URL,
            use_jsonb=True,
        )
        self.co = cohere.Client(settings.COHERE_API_KEY)

    def search_policy_docs(self, query: str) -> str:
        """Tìm kiếm và rerank tài liệu."""
        print(f"[RAG Tool] Searching: {query}")
        initial_docs = self.vector_store.similarity_search(query, k=5)
        
        doc_contents = [d.page_content for d in initial_docs]
        rerank_results = self.co.rerank(
            query=query, 
            documents=doc_contents, 
            top_n=3, 
            model=settings.RERANK_MODEL
        )
        
        formatted_results = []
        for res in rerank_results.results:
            original_doc = initial_docs[res.index]
            page_num = original_doc.metadata.get("page", 0) + 1
            chunk_text = f"[NGUỒN: TRANG {page_num}]\n{original_doc.page_content}\n"
            formatted_results.append(chunk_text)
            
        return "\n".join(formatted_results)

    def get_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self.search_policy_docs,
            name="search_policy_docs",
            description=(
                "Công cụ tìm kiếm thông tin trong tài liệu chính sách công ty (PDF). "
                "Sử dụng khi người dùng hỏi về: lương, thưởng, nghỉ phép, quy định, phúc lợi... "
                "Input: Từ khóa hoặc câu hỏi tìm kiếm. "
                "Output: Các đoạn văn bản liên quan nhất và kèm số trang."
            )
        )