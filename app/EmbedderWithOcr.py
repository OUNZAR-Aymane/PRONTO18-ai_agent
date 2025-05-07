from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import io 
from Embedder import Embedder

class EmbedderWithOcr(Embedder):
    def __init__(self,api_key):
        super().__init__(api_key)

    def extract_text_and_images(self,pdf_path):
        """
        Extract text and images from a PDF file, performing OCR on images when needed.
        
        Args:
            pdf_path (str): Path to the PDF file
        
        Returns:
            list: List of Document objects containing text and metadata
        """
        doc = fitz.open(pdf_path)
        documents = []
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


    def load_and_split(self,pdf_path):
        """
        Load PDF, extract text/images, and split into chunks for embedding.
        
        Args:
            pdf_path (str): Path to the PDF file
        
        Returns:
            list: List of Document chunks suitable for embedding
        """
        raw_docs = self.extract_text_and_images(pdf_path)
        # Split documents into smaller chunks for better embedding
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(raw_docs)
        return chunks
    
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    from pathlib import Path
    load_dotenv()
    DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
    api_key = os.getenv("MISTRAL_API_KEY")
    embedder = EmbedderWithOcr(api_key)
    save_path = os.path.join(DOCS_DIR, "astro-procedures-resume-anon.pdf")
    embedder.embed(save_path)