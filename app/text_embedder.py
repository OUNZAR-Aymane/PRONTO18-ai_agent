from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()
# Path to the FAISS vector index
faiss_index_path = Path("..") / "faiss_index"

def load_and_split(pdf):
    """
    Load a PDF file and split it into smaller chunks for processing.
    
    Args:
        pdf (str): Path to the PDF file
    
    Returns:
        list: List of document chunks
    """
    # Load the PDF document
    loader = PyPDFLoader(pdf)
    documents = loader.load()
    # Split the document into smaller chunks with overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

def embedd(pdf_path):
    """
    Create embeddings from PDF content and store them in a FAISS vector index.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        None: Saves the vector store to disk
    """
    # Load and split the PDF into chunks
    chunks = load_and_split(pdf_path)
    # Define the embedding model using MistralAI
    embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key="MISTRAL_API_KEY")
    
    if not os.path.exists(faiss_index_path):
        # Create a new vector store if none exists
        vector = FAISS.from_documents(chunks, embeddings)
    else:
        # Load existing vector store and add new documents
        vector = FAISS.load_local(faiss_index_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        vector.add_documents(chunks)
    
    # Save the updated vector store to disk
    vector.save_local("faiss_index")



if __name__ == '__main__' :
    for filename in os.listdir("docs"):
        if filename.endswith(".pdf"):  # Check if it's a PDF file
            pdf_path = os.path.join("docs", filename)  # Full path to PDF
            print(f"Processing: {filename}...")
            chunks = load_and_split(pdf_path)
            try:
                embedd(chunks)
                print("Done!")
            except Exception as e:
                print("An error occured :",e)