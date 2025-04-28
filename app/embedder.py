from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
workspace_path = os.getcwd()  # Get current working directory
faiss_index_path = os.path.join(workspace_path, "faiss_index")

def load_and_split(pdf):
    loader = PyPDFLoader(pdf)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

def embedd(chunks):
    # Define the embedding model
    embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key="MISTRAL_API_KEY")
    if not os.path.exists(faiss_index_path):
        # Create the vector store 
        vector = FAISS.from_documents(chunks, embeddings)
    else:
        vector = FAISS.load_local(faiss_index_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        vector.add_documents(chunks)
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