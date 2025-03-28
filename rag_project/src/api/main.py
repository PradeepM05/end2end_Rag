import os
import sys
import time
from datetime import datetime

# Add the project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from src.embedding.embedder import EmbeddingProcessor
from src.retrieval.retriever import EnhancedRetriever
from src.generation.rag_generator import RAGGenerator
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

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize RAG components
try:
    embedder = EmbeddingProcessor()
    vector_store = embedder.load_vector_store(allow_dangerous_deserialization=True)

    if not vector_store:
        logger.error("Vector store not found. Please run the document processing and embedding steps first.")
    else:
        retriever = EnhancedRetriever(vector_store, use_compression=settings.USE_COMPRESSION)
        generator = RAGGenerator(retriever, model_name=settings.MODEL_NAME)
        logger.info("RAG components initialized successfully")
except Exception as e:
    logger.error(f"Error initializing RAG components: {str(e)}")
    # The API will start but query endpoint will fail

# Define API models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    processing_time: float = None

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
        raise HTTPException(status_code=503, detail="RAG system not initialized. Check logs for details.")
    
    try:
        logger.info(f"Received query: {request.query}")
        start_time = time.time()
        
        # Get retrieved documents
        docs = retriever.retrieve(request.query, top_k=settings.RETRIEVAL_TOP_K)
        
        # Generate answer
        answer = generator.generate_answer(request.query)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Track metrics
        metrics.track_query(request.query, processing_time, len(docs))
        
        logger.info(f"Generated answer in {processing_time:.2f} seconds")
        return QueryResponse(answer=answer, processing_time=processing_time)
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.get("/metrics")
def get_metrics():
    """Return usage metrics"""
    return metrics.get_metrics()

@app.get("/health")
def health_check():
    """Health check endpoint for Docker"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)