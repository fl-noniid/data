"""
Updated RAG Components with latest LangChain structure
"""
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Dict, Any, Tuple
import time


class DenseRetriever:
    """Dense retriever using FAISS and HuggingFace Embeddings"""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 3
    ):
        # Using latest langchain-huggingface package
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        self.top_k = top_k
        self.vector_store = None
        
    def build_global_index(self, corpus: List[str]):
        """Build a global index from a list of strings"""
        if not corpus:
            print("Warning: Empty corpus provided. Skipping index build.")
            return
            
        print(f"Building global index with {len(corpus)} documents...")
        start_time = time.time()
        
        docs = [Document(page_content=text) for text in corpus]
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        
        end_time = time.time()
        print(f"Index built in {end_time - start_time:.2f} seconds")
        
    def retrieve(self, query: str, k: int = None) -> Tuple[List[Document], float]:
        """Retrieve documents and return latency"""
        if self.vector_store is None:
            return [], 0.0
            
        k = k or self.top_k
        start_time = time.time()
        docs = self.vector_store.similarity_search(query, k=k)
        latency = time.time() - start_time
        
        return docs, latency


def format_context(documents: List[Document]) -> str:
    """Format retrieved documents for the prompt"""
    return "\n\n".join([doc.page_content for doc in documents])
