from embedding.embedder import EmbeddingProcessor
from retrieval.retriever import EnhancedRetriever
from generation.rag_generator import RAGGenerator

if __name__ == "__main__":
    # Load vector store
    embedder = EmbeddingProcessor()
    vector_store = embedder.load_vector_store()
    
    if vector_store:
        # Initialize retriever
        retriever = EnhancedRetriever(vector_store, use_compression=False)
        
        # Initialize RAG generator
        generator = RAGGenerator(retriever)
        
        # Test with sample questions
        questions = [
            "What are the main topics covered in these documents?",
            "Give me a summary of the key points.",
            "What specific information is mentioned about [specific topic in your documents]?"
        ]
        
        for question in questions:
            print(f"\n\nQuestion: {question}")
            answer = generator.generate_answer(question)
            print(f"Answer: {answer}")
    else:
        print("Please run test_embedding.py first to create a vector store.")