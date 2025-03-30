from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
import time
import logging

logger = logging.getLogger("rag-system")

class EnhancedRetriever:
    def __init__(self, vector_store, use_compression=False):
        self.vector_store = vector_store
        self.base_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        self.use_compression = use_compression
        if use_compression:
            # Initialize compression for more relevant results
            llm = ChatOpenAI(temperature=0)
            compressor = LLMChainExtractor.from_llm(llm)
            self.retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.base_retriever
            )
        else:
            self.retriever = self.base_retriever
    
    def retrieve(self, query, top_k=5):
        """Retrieve relevant documents for a query with detailed timing"""
        logger.info(f"Starting retrieval for query: '{query}' with top_k={top_k}")
        
        # Modify search kwargs for this specific call
        if hasattr(self.base_retriever, "search_kwargs"):
            original_k = self.base_retriever.search_kwargs.get("k", 5)
            self.base_retriever.search_kwargs["k"] = top_k
        
        # Time the embedding of the query
        embedding_start = time.time()
        # We don't have direct access to the embedding step, so this is an approximation
        
        # Time the actual retrieval
        retrieval_start = time.time()
        if self.use_compression:
            # Using enhanced retrieval with compression
            documents = self.retriever.invoke(query)
        else:
            # Using basic vector similarity
            documents = self.base_retriever.invoke(query)
        retrieval_end = time.time()
        
        # Reset search_kwargs if we modified it
        if hasattr(self.base_retriever, "search_kwargs"):
            self.base_retriever.search_kwargs["k"] = original_k
        
        # Log timing information
        embedding_time = retrieval_start - embedding_start
        retrieval_time = retrieval_end - retrieval_start
        total_time = retrieval_end - embedding_start
        
        logger.info(f"Retrieval timing details:")
        logger.info(f"  Preparation: {embedding_time:.2f}s ({(embedding_time/total_time)*100:.1f}% of retrieval)")
        logger.info(f"  Vector search: {retrieval_time:.2f}s ({(retrieval_time/total_time)*100:.1f}% of retrieval)")
        logger.info(f"  Retrieved {len(documents)} documents in {total_time:.2f}s total")
        
        return documents