from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from Embedder import Embedder

class TextEmbedder(Embedder):
    def __init__(self,api_key):
        super().__init__(api_key)
    
    def load_and_split(self,pdf_path):
        """
        Load a PDF file and split it into smaller chunks for processing.
        
        Args:
            pdf (str): Path to the PDF file
        
        Returns:
            list: List of document chunks
        """
        # Load the PDF document
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        # Split the document into smaller chunks with overlap
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        return chunks