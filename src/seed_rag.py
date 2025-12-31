import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

connection = os.getenv("DATABASE_URL")
if not connection:
    raise ValueError("Chưa tìm thấy DATABASE_URL trong file .env")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
collection_name = "company_policies"

def advanced_clean_text(text):
    """
    Hàm làm sạch cho tiếng Việt và PDF
    """
    text = re.sub(r'(?<=[A-ZÀ-Ỹ])\s+(?=[A-ZÀ-Ỹ])', '', text) 
    text = text.replace('\x00', '').replace('…', '...')
    text = re.sub(r'(?<=[a-zà-ỹ])\n(?=[a-zà-ỹ])', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def ingest_docs():
    print("Bắt đầu...")
    pdf_path = "./data/policy.pdf"
    if not os.path.exists(pdf_path):
        print(f"Không tìm thấy file: {pdf_path}")
        return

    print(f"Đang đọc file: {pdf_path}...")
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    
    full_text = ""
    for doc in documents:
        cleaned_page = advanced_clean_text(doc.page_content)
        full_text += cleaned_page + "\n\n" 
    print(f"   -> Tổng dung lượng văn bản sạch: {len(full_text)} ký tự.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=[
            "\n\n",     # Ưu tiên 1: Ngắt đoạn lớn
            "\n",       # Ưu tiên 2: Xuống dòng
            ". ",       # Ưu tiên 3: Hết câu
            "; ",       # Ưu tiên 4: Hết ý trong list
            " - ",      # Ưu tiên 5: Gạch đầu dòng
            " ",        # Cuối cùng: Cắt theo từ
            ""
        ]
    )
    
    chunks = text_splitter.create_documents([full_text])
    print(f"Đã cắt thành {len(chunks)} đoạn (chunks) chất lượng cao.")
    
    with open("debug_chunks_clean.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- CHUNK {i+1} ({len(chunk.page_content)} chars) ---\n")
            f.write(chunk.page_content)
            f.write("\n\n" + "="*50 + "\n\n")
    print("   -> Đã ghi file 'debug_chunks_clean.txt' để review.")

    # Khởi tạo Vector Store
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )

    # Reset & Nạp dữ liệu
    print("Đang làm sạch Database cũ...")
    vector_store.drop_tables()
    
    print("Đang khởi tạo bảng Vector...")
    vector_store.create_tables_if_not_exists()
    vector_store.create_collection()
    
    print("Đang Embed và lưu vào Neon DB...")
    vector_store.add_documents(chunks)
    
    print("Hoàn tất!.")

if __name__ == "__main__":
    ingest_docs()