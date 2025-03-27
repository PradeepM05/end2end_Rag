from langchain.document_loaders import PyPDFLoader #, DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

class DocumentProcessor:
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def load_documents(self):
        """Load documents from the data directory"""
        documents = []
        
        # Load PDFs
        pdf_loader = DirectoryLoader(
            self.data_dir, 
            glob="**/*.pdf", 
            loader_cls=PyPDFLoader
        )
        documents.extend(pdf_loader.load())
        
        # Load text files
        text_loader = DirectoryLoader(
            self.data_dir, 
            glob="**/*.txt", 
            loader_cls=TextLoader
        )
        documents.extend(text_loader.load())
        
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def split_documents(self, documents):
        """Split documents into chunks"""
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        return chunks
    
    def process(self):
        """Load and split documents"""
        documents = self.load_documents()
        chunks = self.split_documents(documents)
        return chunks