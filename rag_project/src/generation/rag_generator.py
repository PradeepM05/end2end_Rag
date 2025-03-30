from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains.combine_documents import create_stuff_documents_chain
import time
import logging
import os

logger = logging.getLogger("rag-system")

class RAGGenerator:
    def __init__(self, retriever, model_name="mistral-7b", temperature=0.1):
        self.retriever = retriever
        
        # Configure the model - Check if we're using Mistral or OpenAI
        if "mistral" in model_name.lower():
            # Using Mistral model
            # Note: You'll need a HuggingFace API token for this to work
            huggingface_api_token = os.environ.get("HUGGINGFACE_API_TOKEN")
            if not huggingface_api_token:
                logger.warning("HUGGINGFACE_API_TOKEN not found in environment. Using default OpenAI model.")
                self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature)
            else:
                logger.info(f"Using Mistral-7B model with temperature {temperature}")
                # Configure Mistral model through Hugging Face API
                self.llm = HuggingFaceEndpoint(
                    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                    huggingfacehub_api_token=huggingface_api_token,
                    temperature=temperature,
                    max_length=512  # You can adjust this parameter
                )
        else:
            # Default to OpenAI
            logger.info(f"Using OpenAI model {model_name} with temperature {temperature}")
            self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        
        # Create a prompt template
        self.prompt_template = PromptTemplate.from_template(
            """You are a helpful AI assistant that provides accurate information based on the given context.
            
            Context:
            {context}
            
            Question: {question}
            
            Provide a comprehensive answer based only on the given context. If the context doesn't contain 
            the information needed to answer the question, simply state "I don't have enough information to answer this question."
            
            Answer:"""
        )
        
        # Create the document chain
        self.doc_chain = create_stuff_documents_chain(self.llm, self.prompt_template)
    
    def generate_answer(self, query):
        """Generate an answer for the query using RAG with timing logs"""
        logger.info(f"Starting answer generation for query: '{query}'")
        
        # Time the document retrieval (if not already done)
        docs_start = time.time()
        retrieved_docs = self.retriever.retrieve(query)
        docs_end = time.time()
        docs_time = docs_end - docs_start
        
        if not retrieved_docs:
            logger.warning("No relevant documents found for the query")
            return "No relevant information found to answer your question."
        
        # Log document metadata if available
        try:
            logger.info(f"Retrieved {len(retrieved_docs)} documents:")
            for i, doc in enumerate(retrieved_docs[:3]):  # Log details of first 3 docs
                metadata = getattr(doc, "metadata", {})
                logger.info(f"  Doc {i+1}: {metadata.get('source', 'Unknown')} ({len(doc.page_content)} chars)")
        except Exception as e:
            logger.info(f"Could not log document details: {str(e)}")
        
        # Time the LLM generation
        llm_start = time.time()
        try:
            # Generate an answer using the documents
            answer = self.doc_chain.invoke({
                "context": retrieved_docs,
                "question": query
            })
            
            llm_end = time.time()
            llm_time = llm_end - llm_start
            
            # Log timing information
            total_time = llm_end - docs_start
            logger.info(f"Answer generation timing details:")
            if docs_time > 0.01:  # Only log if significant time was spent on retrieval
                logger.info(f"  Document retrieval: {docs_time:.2f}s ({(docs_time/total_time)*100:.1f}% of total)")
            logger.info(f"  LLM generation: {llm_time:.2f}s ({(llm_time/total_time)*100:.1f}% of total)")
            logger.info(f"  Total generation time: {total_time:.2f}s")
            logger.info(f"  Answer length: {len(str(answer))} characters")
            
            return answer
        except Exception as e:
            llm_end = time.time()
            logger.error(f"Error in LLM generation: {str(e)}")
            logger.error(f"LLM generation took {llm_end - llm_start:.2f}s before failing")
            raise e