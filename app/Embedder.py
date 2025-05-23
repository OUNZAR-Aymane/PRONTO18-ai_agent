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
    def embed(self,pdf_path,store,save=False):
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
        if store is None :
            # Create a new FAISS vector store
            embeddings = MistralAIEmbeddings(
                model="mistral-embed",
                mistral_api_key=self.api_key,
            )
            store = FAISS.from_documents(chunks, embeddings)
        else : 
            store.add_documents(chunks)
        
        if save :
            # Save the updated vector store to disk
            store.save_local(str(self.faiss_index_path))
            # Enregistrer la correspondance document ↔ doc_id
            self.document_index[os.path.basename(pdf_path)] = doc_id
            with open(self.document_index_path, "w", encoding="utf-8") as f:
                json.dump(self.document_index, f, indent=2)
        return store


    def delete_document(self,doc_name: str, store,save=False) -> None:
        """
        Remove every vector that belongs to the PDF *doc_name*
        without re-embedding anything.
        """
        if doc_name not in self.document_index or store is None:
            return False
        
        target_doc_id = self.document_index[doc_name]


        # -------- collect the doc-store IDs to delete ----------
        ids_to_delete = [
            ds_id
            for row, ds_id in store.index_to_docstore_id.items()
            if store.docstore.search(ds_id).metadata.get("doc_id") == target_doc_id
        ]
        if not ids_to_delete:
            return True

        # -------- constant-time in-place deletion --------------
        store.delete(ids=ids_to_delete)
        if save:
            # -------- persist & housekeeping -----------------------
            store.save_local(str(self.faiss_index_path))
            del self.document_index[doc_name]
            with open(self.document_index_path, "w") as f:
                json.dump(self.document_index, f, indent=2)
        return True

    def vectors_for_pdf(self, doc_id=None) -> int:
        """
        Return the number of vectors stored for *pdf_name*.
        If the PDF is not in the index, the function returns 0.
        """
        embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=self.api_key,
        )

        store = FAISS.load_local(
            str(self.faiss_index_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        # ------- 1. read the document↔doc_id mapping  --------------------------
        if doc_id is not None:
            target_doc_id = doc_id
            # ------- 3. count matching vectors ------------------------------------
            ids = [
                ds_id
                for ds_id in store.index_to_docstore_id.values()
                if store.docstore.search(ds_id).metadata.get("doc_id") == target_doc_id
            ]
            return len(ids)
        else :
            available_ids = [
                ds_id
                for ds_id in store.index_to_docstore_id.values()
            ]
            available_doc_ids = [
                store.docstore.search(ds_id).metadata.get("doc_id")
                for ds_id in available_ids
            ]
            return set(available_doc_ids)

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    from pathlib import Path
    load_dotenv()
    DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
    api_key = os.getenv("MISTRAL_API_KEY")
    embedder = Embedder(api_key)
    print(embedder.vectors_for_pdf())