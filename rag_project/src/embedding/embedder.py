from langchain_openai import OpenAIEmbeddings
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv()

class EmbeddingProcessor:
    def __init__(self, model_name="text-embedding-ada-002"):
        self.embeddings = OpenAIEmbeddings(model=model_name)
    
    def create_vector_store(self, documents, store_path="./data/vector_store"):
        """Create a FAISS vector store from documents"""
        vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Save the vector store locally
        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        vector_store.save_local(store_path)
        
        print(f"Vector store created and saved to {store_path}")
        return vector_store


    def load_vector_store(self, store_path="./data/vector_store", allow_dangerous_deserialization=True):
        """Load a vector store from disk"""
        if os.path.exists(store_path):
            vector_store = FAISS.load_local(
                store_path, 
                self.embeddings,
                allow_dangerous_deserialization=allow_dangerous_deserialization
            )
            print(f"Loaded vector store from {store_path}")
            return vector_store
        else:
            print(f"No vector store found at {store_path}")
            return None