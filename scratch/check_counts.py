
import os
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

load_dotenv()

def check_counts():
    persist_dir = ".rag/multimodal_chroma"
    print(f"--- ChromaDB Audit: {persist_dir} ---")
    
    if not os.path.exists(persist_dir):
        print(f"Directory {persist_dir} does not exist.")
        return

    client = chromadb.PersistentClient(path=persist_dir)
    
    collections = ["mm_text_chunks", "mm_image_chunks", "mm_table_chunks"]
    
    for col_name in collections:
        try:
            col = client.get_collection(col_name)
            count = col.count()
            print(f"Collection '{col_name}': {count} documents")
        except Exception as e:
            print(f"Collection '{col_name}': NOT FOUND or error ({e})")

if __name__ == "__main__":
    check_counts()
