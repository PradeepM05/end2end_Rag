from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

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
        """Retrieve relevant documents for a query"""
        if self.use_compression:
            # Using enhanced retrieval with compression
            documents = self.retriever.invoke(query)
        else:
            # Using basic vector similarity
            documents = self.base_retriever.invoke(query)
            
        return documents