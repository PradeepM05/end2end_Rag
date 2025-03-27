from embedding.embedder import EmbeddingProcessor
from retrieval.retriever import EnhancedRetriever
from generation.rag_generator import RAGGenerator

import time
from datetime import datetime
from src.utils.logging_utils import setup_logger, MetricsTracker
from src.utils.config import get_settings

settings = get_settings()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import sys

#for static files
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def read_root():
    return FileResponse(os.path.join(static_dir, "index.html"))


# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = FastAPI(title="RAG API", description="Retrieval-Augmented Generation API")

# Initialize components
embedder = EmbeddingProcessor()
vector_store = embedder.load_vector_store(allow_dangerous_deserialization=True)

if not vector_store:
    raise Exception("Vector store not found. Please run the document processing and embedding steps first.")

retriever = EnhancedRetriever(vector_store, use_compression=False)
generator = RAGGenerator(retriever)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG API"}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        answer = generator.generate_answer(request.query)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")
    

# Add these imports
import time
from datetime import datetime
from src.utils.logging_utils import setup_logger, MetricsTracker

# Setup logger and metrics
logger = setup_logger()
metrics = MetricsTracker()

# Add a metrics endpoint
@app.get("/metrics")
def get_metrics():
    return metrics.get_metrics()

# Update the query endpoint
@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        logger.info(f"Received query: {request.query}")
        start_time = time.time()
        
        # Get retrieved documents
        docs = retriever.retrieve(request.query)
        
        # Generate answer
        answer = generator.generate_answer(request.query)
        
        end_time = time.time()
        latency = end_time - start_time
        
        # Track metrics
        metrics.track_query(request.query, latency, len(docs))
        
        logger.info(f"Generated answer in {latency:.2f} seconds")
        return QueryResponse(answer=answer)
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")
    

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)