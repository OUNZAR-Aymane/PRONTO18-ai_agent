from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv


import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import uuid
from pdf2image import convert_from_path
import io 
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ounza\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Load environment variables from .env file
load_dotenv()
workspace_path = os.getcwd()  # Get current working directory
images_folder = os.path.join(workspace_path, "images")
faiss_index_path = os.path.join(workspace_path, "faiss_index")



def extract_text_and_images(pdf_path):
    doc = fitz.open(pdf_path)
    documents = []
    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    processed_xrefs = set()  # <- Track image references globally

    for i, page in enumerate(doc):
        print(f"Processing page {i+1}/{len(doc)}")
        page_text = page.get_text() or ""
        image_texts = []
        image_paths = []

        page_images = page.get_images(full=True)
        print(f"  Found {len(page_images)} images on page {i+1}")

        for img_index, img_info in enumerate(page_images):
            xref = img_info[0]

            if xref in processed_xrefs:
                continue  # Already processed this image elsewhere
            processed_xrefs.add(xref)

            try:
                img = doc.extract_image(xref)
                img_bytes = img["image"]
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                # Save image
                unique_id = str(uuid.uuid4())[:8]
                img_filename = f"{base_filename}_p{i+1}_img{img_index}_{unique_id}.png"
                img_path = os.path.join(images_folder, img_filename)
                pil_img.save(img_path)
                image_paths.append(img_path)

                # OCR
                extracted_text = pytesseract.image_to_string(pil_img)
                if extracted_text.strip():
                    image_texts.append(extracted_text)

            except Exception as e:
                print(f"  [WARNING] Could not process image xref={xref}: {e}")
                continue

        # Combine native + OCR text
        combined_text = (page_text + "\n" + "\n".join(image_texts)).strip()

        # Fallback: if still no text
        if not combined_text:
            try:
                fallback_image = convert_from_path(pdf_path, first_page=i+1, last_page=i+1, dpi=150)[0]
                fallback_filename = f"{base_filename}_page{i+1}_fallback.png"
                fallback_path = os.path.join(images_folder, fallback_filename)
                fallback_image.save(fallback_path)

                fallback_text = pytesseract.image_to_string(fallback_image)
                if fallback_text.strip():
                    combined_text = fallback_text.strip()
                    image_paths.append(fallback_path)
            except Exception as e:
                print(f"  [Fallback Failed] Page {i+1}: {e}")

        if combined_text:
            documents.append(
                Document(
                    page_content=combined_text,
                    metadata={
                        "source": pdf_path,
                        "page": i + 1,
                        "image_paths": image_paths
                    }
                )
            )

    return documents




def load_and_split(pdf_path):
    raw_docs = extract_text_and_images(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(raw_docs)
    return chunks

def embedd(chunks):
    # Define the embedding model
    embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key="wNVyBAARBAah94Jwl9WtLFpGT7sM9xFj")
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
            print("Processing:",f'{filename} ...')
            chunks = load_and_split(pdf_path)
            embedd(chunks)
            print("Done!")