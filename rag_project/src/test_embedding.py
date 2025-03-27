from document_processing.loader import DocumentProcessor
from embedding.embedder import EmbeddingProcessor

if __name__ == "__main__":
    # Process documents
    processor = DocumentProcessor()
    chunks = processor.process()
    
    # Create embeddings and vector store
    embedder = EmbeddingProcessor()
    vector_store = embedder.create_vector_store(chunks)
    
    # Test a simple query
    query = "What are the main topics covered in these documents?"
    results = vector_store.similarity_search(query, k=3)
    
    print("\nTest Query Results:")
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Content: {doc.page_content[:150]}...")
        print(f"Metadata: {doc.metadata}")