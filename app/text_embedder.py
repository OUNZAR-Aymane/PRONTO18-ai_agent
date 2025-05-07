from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
from pathlib import Path
import json
import uuid

# Load environment variables from .env file
load_dotenv()
api_key = os.environ["MISTRAL_API_KEY"]
# Path to the FAISS vector index
faiss_index_path = Path("..") / "faiss_index"
document_index_path = Path("..") / "document_index.json"  # pour traquer les IDs des documents

# Load ou init index externe

if document_index_path.exists():
    with open(document_index_path, "r") as f:
        document_index = json.load(f)
else:
    document_index = {}


def load_and_split(pdf_path,doc_id):
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
    # Ajouter doc_id dans les métadonnées pour chaque chunk
    for chunk in chunks:
        chunk.metadata["doc_id"] = doc_id
    return chunks

def text_embedder(pdf_path):
    """
    Create embeddings from PDF content and store them in a FAISS vector index.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        None: Saves the vector store to disk
    """
    # Générer un identifiant unique pour ce document
    doc_id = str(uuid.uuid4())

    # Load and split the PDF into chunks
    chunks = load_and_split(pdf_path,doc_id)
    # Define the embedding model using MistralAI
    embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)
    
    if not os.path.exists(faiss_index_path):
        # Create a new vector store if none exists
        vector = FAISS.from_documents(chunks, embeddings)
    else:
        # Load existing vector store and add new documents
        vector = FAISS.load_local(faiss_index_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        vector.add_documents(chunks)
    
    # Save the updated vector store to disk
    vector.save_local(str(faiss_index_path))
    # Enregistrer la correspondance document ↔ doc_id
    document_index[os.path.basename(pdf_path)] = doc_id
    with open(document_index_path, "w") as f:
        json.dump(document_index, f, indent=2)

def delete_document(doc_name: str) -> None:
    """
    Remove every vector that belongs to the PDF *doc_name*
    without re-embedding anything.
    """
    if doc_name not in document_index:
        print(f"{doc_name} not present.")
        return

    target_doc_id = document_index[doc_name]

    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=api_key,
    )
    store = FAISS.load_local(
        faiss_index_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    # -------- collect the doc-store IDs to delete ----------
    ids_to_delete = [
        ds_id
        for row, ds_id in store.index_to_docstore_id.items()
        if store.docstore.search(ds_id).metadata.get("doc_id") == target_doc_id
    ]
    if not ids_to_delete:
        print(f"No vectors found for {doc_name}")
        return

    # -------- constant-time in-place deletion --------------
    store.delete(ids=ids_to_delete)

    # -------- persist & housekeeping -----------------------
    store.save_local(str(faiss_index_path))
    del document_index[doc_name]
    with open(document_index_path, "w") as f:
        json.dump(document_index, f, indent=2)

    print(f"{doc_name}: {len(ids_to_delete)} vectors deleted.")



if __name__ == '__main__' :
        filename = "docs/ALLEZ.pdf"
        print(f"Processing: {filename}...")
        try:
            text_embedder(filename)
            delete_document()
            print("Done!")
        except Exception as e:
            print("An error occured :",e)