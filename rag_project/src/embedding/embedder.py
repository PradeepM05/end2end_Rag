from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
import subprocess
import logging
import time
from dotenv import load_dotenv

load_dotenv()

class EmbeddingProcessor:
    def __init__(self, model_name="text-embedding-ada-002"):
        self.embeddings = OpenAIEmbeddings(model=model_name)
        self.logger = logging.getLogger("rag-system")
        
    def create_vector_store(self, documents, store_path="./data/vector_store"):
        """Create a FAISS vector store from documents"""
        vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Save the vector store locally
        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        vector_store.save_local(store_path)
        
        # If running in Cloud Run, also sync to Cloud Storage
        if os.environ.get("STORAGE_BUCKET"):
            self._sync_to_cloud_storage(store_path)
            
        self.logger.info(f"Vector store created and saved to {store_path}")
        return vector_store

    def load_vector_store(self, store_path="./data/vector_store", allow_dangerous_deserialization=True):
        """Load a vector store from disk or cloud storage if available"""
        
        # If running in Cloud Run, check if we need to sync from Cloud Storage
        if os.environ.get("STORAGE_BUCKET") and not os.path.exists(os.path.join(store_path, "index.faiss")):
            self.logger.info(f"Vector store not found locally. Attempting to download from Cloud Storage.")
            success = self._sync_from_cloud_storage(store_path)
            if not success:
                self.logger.error(f"Failed to download vector store from Cloud Storage.")
                return None
            
            # Wait a moment to ensure files are fully written
            time.sleep(2)
        
        if os.path.exists(os.path.join(store_path, "index.faiss")):
            try:
                self.logger.info(f"Loading vector store from {store_path}")
                vector_store = FAISS.load_local(
                    store_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=allow_dangerous_deserialization
                )
                self.logger.info(f"Successfully loaded vector store from {store_path}")
                return vector_store
            except Exception as e:
                self.logger.error(f"Error loading vector store: {str(e)}")
                return None
        else:
            self.logger.warning(f"No vector store found at {store_path}")
            return None
            
    def _sync_to_cloud_storage(self, local_path):
        """Sync local vector store to Cloud Storage"""
        bucket = os.environ.get("STORAGE_BUCKET")
        if not bucket:
            return False
            
        try:
            self.logger.info(f"Syncing vector store to gs://{bucket}/")
            cmd = f"gsutil -m cp -r {local_path}/* gs://{bucket}/"
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            self.logger.info(f"Sync to Cloud Storage complete: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to sync to Cloud Storage: {str(e)}, {e.stderr}")
            return False
            
    def _sync_from_cloud_storage(self, local_path):
        """Sync vector store from Cloud Storage to local disk"""
        bucket = os.environ.get("STORAGE_BUCKET")
        if not bucket:
            return False
            
        try:
            self.logger.info(f"Downloading vector store from gs://{bucket}/")
            os.makedirs(local_path, exist_ok=True)
            cmd = f"gsutil -m cp -r gs://{bucket}/* {local_path}/"
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            
            # List files to verify
            files = os.listdir(local_path)
            self.logger.info(f"Files in {local_path} after download: {files}")
            
            return "index.faiss" in files and "index.pkl" in files
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to sync from Cloud Storage: {str(e)}, {e.stderr}")
            return False