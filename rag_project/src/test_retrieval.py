from embedding.embedder import EmbeddingProcessor
from retrieval.retriever import EnhancedRetriever

if __name__ == "__main__":
    # Load vector store
    embedder = EmbeddingProcessor()
    vector_store = embedder.load_vector_store()
    
    if vector_store:
        # Initialize retriever
        retriever = EnhancedRetriever(vector_store, use_compression=False)
        
        # Test retrieval
        query = "What are the key concepts in these documents?"
        results = retriever.retrieve(query)
        
        print(f"\nRetrieved {len(results)} documents for query: '{query}'")
        for i, doc in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Content: {doc.page_content[:150]}...")
            print(f"Metadata: {doc.metadata}")
    else:
        print("Please run test_embedding.py first to create a vector store.")