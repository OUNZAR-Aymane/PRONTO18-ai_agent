from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
from pathlib import Path
import json
import uuid

# Load ou init index externe


class Embedder:
    def __init__(self, api_key):
        self.api_key = api_key
        self.faiss_index_path = Path(__file__).resolve().parent.parent / "faiss_index"
        self.document_index_path = Path(__file__).resolve().parent.parent / "document_index.json"  # pour traquer les IDs des documents
        if self.document_index_path.exists():
            with open(self.document_index_path, "r") as f:
                self.document_index = json.load(f)
        else:
            self.document_index = {}
    def load_and_split(self,**args):
        pass 
    def embed(self,pdf_path):
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
        chunks = self.load_and_split(pdf_path)

        # Ajouter doc_id dans les métadonnées pour chaque chunk
        for chunk in chunks:
            chunk.metadata["doc_id"] = doc_id
    
        # Define the embedding model using MistralAI
        embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=self.api_key)
        
        if not self.faiss_index_path.exists():
            self.faiss_index_path.mkdir(parents=True, exist_ok=True)
            vector = FAISS.from_documents(chunks, embeddings)
        else:
            vector = FAISS.load_local(str(self.faiss_index_path), embeddings=embeddings, allow_dangerous_deserialization=True)
            vector.add_documents(chunks)
        
        # Save the updated vector store to disk
        vector.save_local(str(self.faiss_index_path))
        # Enregistrer la correspondance document ↔ doc_id
        self.document_index[os.path.basename(pdf_path)] = doc_id
        with open(self.document_index_path, "w") as f:
            json.dump(self.document_index, f, indent=2)
            print(self.document_index)


    def delete_document(self,doc_name: str) -> None:
        """
        Remove every vector that belongs to the PDF *doc_name*
        without re-embedding anything.
        """
        if doc_name not in self.document_index:
            print(f"{doc_name} not present.")
            return

        target_doc_id = self.document_index[doc_name]

        embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=self.api_key,
        )
        store = FAISS.load_local(
            self.faiss_index_path,
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
            return

        # -------- constant-time in-place deletion --------------
        store.delete(ids=ids_to_delete)

        # -------- persist & housekeeping -----------------------
        store.save_local(str(self.faiss_index_path))
        del self.document_index[doc_name]
        with open(self.document_index_path, "w") as f:
            json.dump(self.document_index, f, indent=2)

    def vectors_for_pdf(self, doc_id) -> int:
        """
        Return the number of vectors stored for *pdf_name*.
        If the PDF is not in the index, the function returns 0.
        """
        # ------- 1. read the document↔doc_id mapping  --------------------------

        target_doc_id = doc_id

        embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=self.api_key or os.getenv("MISTRAL_API_KEY"),
        )

        store = FAISS.load_local(
            str(self.faiss_index_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

        # ------- 3. count matching vectors ------------------------------------
        ids = [
            ds_id
            for ds_id in store.index_to_docstore_id.values()
            if store.docstore.search(ds_id).metadata.get("doc_id") == target_doc_id
        ]
        return len(ids)

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    from pathlib import Path
    load_dotenv()
    DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
    api_key = os.getenv("MISTRAL_API_KEY")
    embedder = Embedder(api_key)
    print(embedder.vectors_for_pdf("b6c0c558-5582-4aeb-9d8a-293dfe04a32d"))