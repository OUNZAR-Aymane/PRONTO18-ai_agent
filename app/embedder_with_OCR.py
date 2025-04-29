from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from pathlib import Path
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import uuid
from pdf2image import convert_from_path
import io 

# Load environment variables from .env file
load_dotenv()
# Define path to store FAISS vector index
faiss_index_path = Path("..") / "faiss_index"


def extract_text_and_images(pdf_path):
    """
    Extract text and images from a PDF file, performing OCR on images when needed.
    
    Args:
        pdf_path (str): Path to the PDF file
    
    Returns:
        list: List of Document objects containing text and metadata
    """
    doc = fitz.open(pdf_path)
    documents = []
    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    processed_xrefs = set()  # Track image references to avoid duplicates
    
    for i, page in enumerate(doc):
        print(f"Processing page {i+1}/{len(doc)}")
        # Get text directly from PDF
        page_text = page.get_text() or ""
        image_texts = []
        # Extract images from the page
        page_images = page.get_images(full=True)
        print(f"  Found {len(page_images)} images on page {i+1}")

        for img_index, img_info in enumerate(page_images):
            xref = img_info[0]  # Image reference in the PDF

            # Skip if we've already processed this image
            if xref in processed_xrefs:
                continue
            processed_xrefs.add(xref)

            try:
                # Extract image bytes and convert to PIL Image
                img = doc.extract_image(xref)
                img_bytes = img["image"]
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                # Perform OCR on the image to extract text
                extracted_text = pytesseract.image_to_string(pil_img)
                if extracted_text.strip():
                    image_texts.append(extracted_text)

            except Exception as e:
                print(f"  [WARNING] Could not process image xref={xref}: {e}")
                continue

        # Combine native PDF text with OCR results
        combined_text = (page_text + "\n" + "\n".join(image_texts)).strip()

        # Fallback: If no text was extracted, convert entire page to image and OCR
        if not combined_text:
            try:
                # Convert page to image
                fallback_image = convert_from_path(pdf_path, first_page=i+1, last_page=i+1, dpi=150)[0]

                # OCR the entire page
                fallback_text = pytesseract.image_to_string(fallback_image)
                if fallback_text.strip():
                    combined_text = fallback_text.strip()
            except Exception as e:
                print(f"  [Fallback Failed] Page {i+1}: {e}")

        # Create Document object if we have extracted text
        if combined_text:
            documents.append(
                Document(
                    page_content=combined_text,
                    metadata={
                        "source": pdf_path,
                        "page": i + 1,
                    }
                )
            )

    return documents


def load_and_split(pdf_path):
    """
    Load PDF, extract text/images, and split into chunks for embedding.
    
    Args:
        pdf_path (str): Path to the PDF file
    
    Returns:
        list: List of Document chunks suitable for embedding
    """
    raw_docs = extract_text_and_images(pdf_path)
    # Split documents into smaller chunks for better embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(raw_docs)
    return chunks


def embedd(pdf_path):
    """
    Create embeddings for PDF content and store in FAISS index.
    
    Args:
        pdf_path (str): Path to the PDF file
    """
    chunks = load_and_split(pdf_path)
    # Initialize Mistral AI embedding model
    embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key="wNVyBAARBAah94Jwl9WtLFpGT7sM9xFj")
    
    if not os.path.exists(faiss_index_path):
        # Create new vector store if none exists
        vector = FAISS.from_documents(chunks, embeddings)
    else:
        # Load existing vector store and add new documents
        vector = FAISS.load_local(faiss_index_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        vector.add_documents(chunks)
    
    # Save the updated vector store
    vector.save_local("faiss_index")


if __name__ == '__main__' :
    for filename in os.listdir("docs"):
        if filename.endswith(".pdf"):  # Check if it's a PDF file
            pdf_path = os.path.join("docs", filename)  # Full path to PDF
            print("Processing:",f'{filename} ...')
            chunks = load_and_split(pdf_path)
            embedd(chunks)
            print("Done!")