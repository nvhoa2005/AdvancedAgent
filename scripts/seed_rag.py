import os
import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from config.settings import settings

class PolicyDocumentIngestor:
    """Class quản lý việc đọc, làm sạch và nhúng (embed) tài liệu PDF vào Vector DB."""
    
    def __init__(self, pdf_path: str = "./data/policy.pdf"):
        self.pdf_path = pdf_path
        self.embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=settings.COLLECTION_NAME,
            connection=settings.DATABASE_URL,
            use_jsonb=True,
        )

    @staticmethod
    def advanced_clean_text(text: str) -> str:
        """Hàm làm sạch cho tiếng Việt và PDF."""
        text = re.sub(r'(?<=[A-ZÀ-Ỹ])\s+(?=[A-ZÀ-Ỹ])', '', text) 
        text = text.replace('\x00', '').replace('…', '...')
        text = re.sub(r'(?<=[a-zà-ỹ])\n(?=[a-zà-ỹ])', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def load_and_split(self):
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"Không tìm thấy file: {self.pdf_path}")

        print(f"Đang đọc file: {self.pdf_path}...")
        loader = PyMuPDFLoader(self.pdf_path)
        documents = loader.load()
        
        for doc in documents:
            doc.page_content = self.advanced_clean_text(doc.page_content)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "; ", " - ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Đã cắt thành {len(chunks)} đoạn (chunks) chất lượng cao.")
        return chunks

    def ingest_to_db(self, chunks):
        print("Đang làm sạch Database cũ...")
        self.vector_store.drop_tables()
        
        print("Đang khởi tạo bảng Vector...")
        self.vector_store.create_tables_if_not_exists()
        self.vector_store.create_collection()
        
        print("Đang Embed và lưu vào Neon DB...")
        self.vector_store.add_documents(chunks)
        print("Hoàn tất nhúng dữ liệu!")

    def run(self):
        """Hàm kích hoạt toàn bộ quy trình."""
        try:
            chunks = self.load_and_split()
            self.ingest_to_db(chunks)
        except Exception as e:
            print(f"Lỗi trong quá trình ingest: {e}")

if __name__ == "__main__":
    ingestor = PolicyDocumentIngestor()
    ingestor.run()