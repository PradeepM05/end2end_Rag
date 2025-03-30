import os
import sys
import time
import json
from datetime import datetime

# Add the project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from src.embedding.embedder import EmbeddingProcessor
from src.retrieval.retriever import EnhancedRetriever
from src.generation.rag_generator import RAGGenerator
from src.document_processing.loader import DocumentProcessor
from src.utils.logging_utils import setup_logger, MetricsTracker
from src.utils.config import get_settings

# Get settings and initialize logger and metrics
settings = get_settings()
logger = setup_logger()
metrics = MetricsTracker()

# Initialize FastAPI app
app = FastAPI(title="RAG API", description="Retrieval-Augmented Generation API")

# Define global variables
embedder = None
vector_store = None
retriever = None
generator = None
data_dir = "./data"
registry_path = os.path.join(data_dir, "document_registry.json")

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

def load_document_registry():
    """Load the document registry from file or create a new one"""
    if os.path.exists(registry_path):
        try:
            with open(registry_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading document registry: {str(e)}")
    
    # Create default registry
    registry = {
        "files": {},
        "last_update": datetime.now().isoformat(),
        "vector_store_status": "unknown"
    }
    return registry

def save_document_registry(registry):
    """Save the document registry to file"""
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    try:
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving document registry: {str(e)}")

def scan_document_directory():
    """Scan the document directory and update the registry"""
    registry = load_document_registry()
    registry_files = registry.get("files", {})
    
    # Track current files
    current_files = set()
    
    # Scan directory for PDF and TXT files
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.pdf', '.txt')):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, data_dir)
                current_files.add(rel_path)
                
                # Get file stats
                last_modified = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                file_size = os.path.getsize(file_path)
                
                # Check if file is new or modified
                if rel_path not in registry_files:
                    registry_files[rel_path] = {
                        "last_modified": last_modified,
                        "file_size": file_size,
                        "status": "new",
                        "chunks": 0,
                        "last_processed": None,
                        "deleted": False
                    }
                elif (registry_files[rel_path]["last_modified"] != last_modified or 
                      registry_files[rel_path]["file_size"] != file_size):
                    registry_files[rel_path]["last_modified"] = last_modified
                    registry_files[rel_path]["file_size"] = file_size
                    registry_files[rel_path]["status"] = "modified"
    
    # Mark missing files as deleted
    for file_path in list(registry_files.keys()):
        if file_path not in current_files:
            registry_files[file_path]["deleted"] = True
    
    # Update registry
    registry["files"] = registry_files
    registry["last_update"] = datetime.now().isoformat()
    save_document_registry(registry)
    
    return registry

def initialize_rag_components():
    """Initialize the RAG components"""
    global embedder, vector_store, retriever, generator
    
    try:
        # Scan document directory and update registry
        registry = scan_document_directory()
        
        # Initialize embedder
        if embedder is None:
            embedder = EmbeddingProcessor()
        
        # Try to load existing vector store
        vector_store = embedder.load_vector_store(allow_dangerous_deserialization=True)
        
        # If no vector store exists, create one
        if not vector_store:
            logger.info("No vector store found. Creating one from documents...")
            doc_processor = DocumentProcessor()
            chunks = doc_processor.process()
            
            if chunks and len(chunks) > 0:
                vector_store = embedder.create_vector_store(chunks)
                
                # Update registry with processed files
                for file_path, file_info in registry["files"].items():
                    if not file_info["deleted"]:
                        file_info["status"] = "processed"
                        file_info["last_processed"] = datetime.now().isoformat()
                
                registry["vector_store_status"] = "up_to_date"
                save_document_registry(registry)
                
                logger.info(f"Vector store created successfully with {len(chunks)} chunks")
        
        # Initialize retriever and generator if we have a vector store
        if vector_store:
            # Check if we need to filter deleted documents
            deleted_docs = [path for path, info in registry.get("files", {}).items() 
                           if info.get("deleted", False)]
            
            # Create retriever with metadata filter to exclude deleted documents
            retriever = EnhancedRetriever(vector_store, use_compression=settings.USE_COMPRESSION)
            generator = RAGGenerator(retriever, model_name=settings.MODEL_NAME)
            logger.info("RAG components initialized successfully")
            return True
        return False

    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        return False

def rebuild_vector_store():
    """Rebuild the vector store from non-deleted documents"""
    global embedder, vector_store, retriever, generator
    
    try:
        registry = load_document_registry()
        
        # Process only non-deleted documents
        doc_processor = DocumentProcessor()
        chunks = doc_processor.process()
        
        if not chunks or len(chunks) == 0:
            logger.warning("No documents found to process")
            return {"status": "error", "message": "No documents found to process"}
        
        # Create a new embedder if needed
        if embedder is None:
            embedder = EmbeddingProcessor()
        
        # Create vector store
        vector_store = embedder.create_vector_store(chunks)
        
        # Reinitialize retriever and generator
        retriever = EnhancedRetriever(vector_store, use_compression=settings.USE_COMPRESSION)
        generator = RAGGenerator(retriever, model_name=settings.MODEL_NAME)
        
        # Update registry
        for file_path, file_info in registry["files"].items():
            if not file_info["deleted"]:
                file_info["status"] = "processed"
                file_info["last_processed"] = datetime.now().isoformat()
        
        registry["vector_store_status"] = "up_to_date"
        registry["last_update"] = datetime.now().isoformat()
        save_document_registry(registry)
        
        logger.info(f"Vector store rebuilt successfully with {len(chunks)} chunks")
        return {"status": "success", "message": f"Vector store rebuilt with {len(chunks)} chunks"}
    
    except Exception as e:
        logger.error(f"Error rebuilding vector store: {str(e)}")
        return {"status": "error", "message": f"Error rebuilding vector store: {str(e)}"}

# Initialize RAG components on startup
initialize_rag_components()

# Define API models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    processing_time: float = None

class DocumentStatus(BaseModel):
    path: str
    status: str = "active"

# Define API endpoints
@app.get("/")
def read_root():
    if os.path.exists(os.path.join(static_dir, "index.html")):
        return FileResponse(os.path.join(static_dir, "index.html"))
    return {"message": "Welcome to the RAG API"}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    global retriever, generator
    
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if retriever is None or generator is None:
        raise HTTPException(status_code=503, detail="RAG system is not initialized. Check logs for details.")
    
    try:
        logger.info(f"Received query: {request.query}")
        overall_start_time = time.time()
        
        # Time the retrieval phase
        retrieval_start = time.time()
        docs = retriever.retrieve(request.query, top_k=settings.RETRIEVAL_TOP_K)
        retrieval_end = time.time()
        retrieval_time = retrieval_end - retrieval_start
        logger.info(f"Retrieval took {retrieval_time:.2f} seconds, found {len(docs)} documents")
        
        # Time the generation phase
        generation_start = time.time()
        answer = generator.generate_answer(request.query)
        generation_end = time.time()
        generation_time = generation_end - generation_start
        logger.info(f"Generation took {generation_time:.2f} seconds")
        
        # Overall processing time
        overall_end_time = time.time()
        processing_time = overall_end_time - overall_start_time
        
        # Log detailed timing information
        logger.info(f"Total processing time: {processing_time:.2f} seconds")
        logger.info(f"Retrieval: {retrieval_time:.2f}s ({(retrieval_time/processing_time)*100:.1f}% of total)")
        logger.info(f"Generation: {generation_time:.2f}s ({(generation_time/processing_time)*100:.1f}% of total)")
        logger.info(f"Overhead: {(processing_time - retrieval_time - generation_time):.2f}s ({((processing_time - retrieval_time - generation_time)/processing_time)*100:.1f}% of total)")
        
        # Track metrics
        metrics.track_query(request.query, processing_time, len(docs))
        
        return QueryResponse(answer=answer, processing_time=processing_time)
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.post("/refresh")
def manual_refresh(background_tasks: BackgroundTasks):
    """Manually refresh the vector store"""
    try:
        background_tasks.add_task(rebuild_vector_store)
        return {"status": "initiated", "message": "Vector store refresh has been initiated in the background"}
    except Exception as e:
        logger.error(f"Error initiating refresh: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error initiating refresh: {str(e)}")

@app.post("/documents/status")
def update_document_status(doc_status: DocumentStatus):
    """Mark a document as active or deleted"""
    try:
        registry = load_document_registry()
        
        if doc_status.path not in registry["files"]:
            raise HTTPException(status_code=404, detail=f"Document {doc_status.path} not found in registry")
        
        # Update status
        registry["files"][doc_status.path]["deleted"] = (doc_status.status == "deleted")
        registry["files"][doc_status.path]["status"] = "modified"
        registry["vector_store_status"] = "needs_rebuild"
        save_document_registry(registry)
        
        return {
            "status": "success", 
            "message": f"Document {doc_status.path} marked as {doc_status.status}",
            "note": "You may need to refresh the vector store for changes to take effect"
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error updating document status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating document status: {str(e)}")

@app.get("/documents")
def list_documents():
    """List all documents in the registry"""
    try:
        registry = load_document_registry()
        document_list = []
        
        for path, info in registry["files"].items():
            document_list.append({
                "path": path,
                "status": "deleted" if info.get("deleted", False) else "active",
                "last_modified": info.get("last_modified"),
                "last_processed": info.get("last_processed"),
                "size": info.get("file_size")
            })
        
        return {
            "documents": document_list,
            "count": len(document_list),
            "active_count": sum(1 for doc in document_list if doc["status"] == "active"),
            "last_update": registry.get("last_update")
        }
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.post("/scan")
def scan_documents():
    """Scan for new or modified documents"""
    try:
        registry = scan_document_directory()
        new_files = [path for path, info in registry["files"].items() 
                    if info["status"] == "new" and not info.get("deleted", False)]
        modified_files = [path for path, info in registry["files"].items() 
                         if info["status"] == "modified" and not info.get("deleted", False)]
        deleted_files = [path for path, info in registry["files"].items() 
                        if info.get("deleted", False)]
        
        return {
            "status": "success",
            "new_files": new_files,
            "modified_files": modified_files,
            "deleted_files": deleted_files,
            "total_files": len(registry["files"]),
            "active_files": len(registry["files"]) - len(deleted_files)
        }
    except Exception as e:
        logger.error(f"Error scanning documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error scanning documents: {str(e)}")

@app.get("/metrics")
def get_metrics():
    """Return usage metrics"""
    return metrics.get_metrics()

@app.get("/health")
def health_check():
    """Health check endpoint for Docker"""
    return {"status": "healthy", "components": {
        "vector_store": vector_store is not None,
        "retriever": retriever is not None,
        "generator": generator is not None
    }}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)