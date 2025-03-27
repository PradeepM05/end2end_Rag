from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain

class RAGGenerator:
    def __init__(self, retriever, model_name="gpt-3.5-turbo"):
        self.retriever = retriever
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        
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
        """Generate an answer for the query using RAG"""
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(query)
        
        if not retrieved_docs:
            return "No relevant information found to answer your question."
        
        # Generate an answer using the documents
        answer = self.doc_chain.invoke({
            "context": retrieved_docs,
            "question": query
        })
        
        return answer