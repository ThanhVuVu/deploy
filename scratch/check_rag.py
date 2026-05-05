
import os
import logging
from dotenv import load_dotenv
from src.agents.scripting.rag.multimodal_retriever import MultimodalRetriever

# Setup logging
logging.basicConfig(level=logging.INFO)
load_dotenv()

def test_rag():
    print("--- RAG Health Check ---")
    print(f"Backend: {os.getenv('PROVIDER_BACKEND')}")
    print(f"Embedding Model: {os.getenv('EMBEDDING_MODEL')}")
    
    try:
        retriever = MultimodalRetriever(max_distance=2.0) # Set very high to see everything
        query = "hat dau nay mam"
        print(f"\nQuerying RAG for: '{query}'...")
        
        result = retriever.hybrid_retrieve(query, top_k=5)
        
        print(f"\nResults Breakdown:")
        print(f"- Text chunks: {len(result.text_chunks)}")
        print(f"- Table chunks: {len(result.table_chunks)}")
        print(f"- Image chunks: {len(result.image_chunks)}")
        
        if result.text_chunks or result.table_chunks or result.image_chunks:
            print("\n--- Top Results Found ---")
            for i, chunk in enumerate(result.text_chunks[:3]):
                print(f"[TEXT {i+1}] Score: {chunk.score:.4f} | Source: {chunk.source_file} | Content: {chunk.content[:100]}...")
            for i, chunk in enumerate(result.image_chunks[:3]):
                print(f"[IMAGE {i+1}] Score: {chunk.score:.4f} | Source: {chunk.source_file} | Caption: {chunk.caption[:100]}...")
            
            if not (result.text_chunks or result.table_chunks):
                print("\n[WARN] Only images were found. The pipeline currently requires Text or Tables to proceed.")
        else:
            print("\n[ERROR] Absolutely nothing found even with max_distance=2.0. This suggests the index is empty.")
            
    except Exception as e:
        print(f"[ERROR] RAG Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag()
