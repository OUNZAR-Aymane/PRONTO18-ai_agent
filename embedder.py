from langchain_community.document_loaders import TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.document_loaders import UnstructuredPDFLoader
import time
from PyPDF2 import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_and_split(pdf):
    loader = UnstructuredPDFLoader(pdf, strategy="hi-res")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

def embedd(chunks):
    # Define the embedding model
    embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key="wNVyBAARBAah94Jwl9WtLFpGT7sM9xFj")
    # Create the vector store 
    vector = FAISS.from_documents(chunks, embeddings)
    vector.save_local("faiss_index")


if __name__ == '__main__' :
    chunks = load_and_split(r"C:\Users\ounza\IMT\Semestre 6\PRONTO\ai-agent\docs\astro-procedures-resume-anon.pdf")
    print(chunks)
    embedd(chunks)
