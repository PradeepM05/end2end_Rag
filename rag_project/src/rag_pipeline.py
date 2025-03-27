from document_processing.loader import DocumentProcessor
from embedding.embedder import EmbeddingProcessor
from retrieval.retriever import EnhancedRetriever
from generation.rag_generator import RAGGenerator
import argparse

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline")
    parser.add_argument("--process_docs", action="store_true", help="Process documents and create vector store")
    parser.add_argument("--query", type=str, help="Query to answer")
    args = parser.parse_args()
    
    if args.process_docs:
        print("Processing documents...")
        processor = DocumentProcessor()
        chunks = processor.process()
        
        print("Creating vector store...")
        embedder = EmbeddingProcessor()
        vector_store = embedder.create_vector_store(chunks)
        print("Vector store created successfully!")
    
    if args.query:
        embedder = EmbeddingProcessor()
        vector_store = embedder.load_vector_store(allow_dangerous_deserialization=True)
        
        if not vector_store:
            print("Error: Vector store not found. Please run with --process_docs first.")
            return
        
        retriever = EnhancedRetriever(vector_store)
        generator = RAGGenerator(retriever)
        
        print(f"\nQuestion: {args.query}")
        answer = generator.generate_answer(args.query)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()