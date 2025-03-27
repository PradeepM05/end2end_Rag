from document_processing.loader import DocumentProcessor

if __name__ == "__main__":
    processor = DocumentProcessor()
    chunks = processor.process()
    
    # Print some sample chunks
    print("\nSample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"Content: {chunk.page_content[:150]}...")
        print(f"Metadata: {chunk.metadata}")