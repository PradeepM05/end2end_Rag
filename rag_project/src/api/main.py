from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import sys

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from embedding.embedder import EmbeddingProcessor
from retrieval.retriever import EnhancedRetriever
from generation.rag_generator import RAGGenerator

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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)