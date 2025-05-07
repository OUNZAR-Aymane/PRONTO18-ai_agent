import streamlit as st
from chat_bot import load_vector_store, get_response, build_chains, model_and_embedding_function, create_prompt, create_contextualize_q_system_prompt
from Embedder import Embedder
from TextEmbedder import TextEmbedder
from EmbedderWithOcr import EmbedderWithOcr
from MultimodalEmbedder import MultimodalEmbedder
from langchain_core.messages import HumanMessage, AIMessage
import os

from dotenv import load_dotenv
from pathlib import Path
import torch, types
torch.classes.__path__ = types.SimpleNamespace(_path=[])

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if api_key is None:
    raise RuntimeError("MISTRAL_API_KEY not set in environment")
# Chemin vers le dossier où tu veux stocker les PDFs
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
# Path to the FAISS index containing our vector embeddings
INDEX_DIR = Path(__file__).resolve().parent.parent / "faiss_index"

model, embedding_function = model_and_embedding_function(api_key)
contextualize_q_prompt = create_contextualize_q_system_prompt()
prompt = create_prompt()
if "vector" not in st.session_state:
    st.session_state.vector = load_vector_store(INDEX_DIR, embedding_function)
if "chain" not in st.session_state:
    st.session_state.chain  = build_chains(
        vector=st.session_state.vector,
        model=model,
        prompt=prompt,
        contextualize_q_prompt=contextualize_q_prompt,
    )


# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "available_documents" not in st.session_state:
    st.session_state.available_documents = []

# Titre de la page
# Page configuration
st.set_page_config(
    page_title="Chatbot Observatoire Astronomique", 
    page_icon=":astronaut:", 
    layout="wide",
    menu_items={
        'About': "Observatoire Astronomique - IMT Atlantique, campus de Brest"
    }
)

# Function to refresh available documents
def refresh_document_list():
    if DOCS_DIR.exists():
        st.session_state.available_documents = [f for f in os.listdir(DOCS_DIR) if f.endswith(('.pdf', '.PDF'))]
    else:
        st.session_state.available_documents = []

# Initialize document list
refresh_document_list()

with st.container():
    st.markdown("""
    Bonjour! Je suis votre assistant virtuel pour l'observatoire astronomique de l'IMT Atlantique. 
    Je peux vous aider avec:
    - Les informations sur l'observatoire
    - L'utilisation des équipements
    - Les objets astronomiques visibles
    - Les procédures d'observation
    
    Posez-moi vos questions, je suis là pour vous aider!
    """)


# Ajouter un titre
st.title("Bienvenue dans l'Observatoire Astronomique 🚀")




with st.sidebar:
    st.subheader("Gérer vos documents")
    # Document upload section
    with st.expander("Télécharger un document", expanded=False):
        pdf = st.file_uploader("Téléchargez vos documents ici et cliquez sur 'Traiter'")
        options = ["Text only embedder", "Embedder with OCR", "Multimodal embedder "]
        if pdf is not None:
            selected_embedder = st.selectbox("Choisissez l'embedder", options, index=None, placeholder="— choisir un embedder —" )
            if selected_embedder == "Text only embedder":
                embedder = TextEmbedder(api_key)
            elif selected_embedder == "Embedder with OCR":
                embedder = EmbedderWithOcr(api_key)
            elif selected_embedder == "Multimodal embedder":
                embedder = MultimodalEmbedder(api_key)
            else :
                embedder = Embedder(api_key)

            if st.button("Traiter" , key="process_button"):
                try :
                    # Construire le chemin de destination
                    save_path = os.path.join(DOCS_DIR, pdf.name)
                    with open(save_path, "wb") as f:
                        f.write(pdf.read())
                    st.success(f"Fichier enregistré dans : {save_path}")
                    with st.spinner("Indexation du document…"):
                        # Process the document in a thread
                        embedder.embed(save_path)
                        st.session_state.vector = load_vector_store(INDEX_DIR, embedding_function)
                        st.session_state.chain  = build_chains(
                            vector=st.session_state.vector,
                            model=model,
                            prompt=prompt,
                            contextualize_q_prompt=contextualize_q_prompt,
                        )
                        refresh_document_list()
                    st.success("Documents traités avec succès !")
                except Exception as e:
                    st.error(f"Erreur lors du traitement du document : {e}")

    # Document deletion section
    with st.expander("Supprimer un document", expanded=False):
        if st.session_state.available_documents:
            selected_doc = st.selectbox(
                'Sélectionnez le document à supprimer',
                st.session_state.available_documents,
                index=None,                         # ⟵ aucun élément pré-sélectionné
                placeholder="— choisir un document —" 
            )
            
            if st.button("Supprimer", key="delete_btn"):
                try:
                    with st.spinner("Suppression en cours..."):
                        file_path = os.path.join(DOCS_DIR, selected_doc)
                        Embedder(api_key).delete_document(selected_doc)
                        os.remove(file_path)
                        st.session_state.vector = load_vector_store(INDEX_DIR, embedding_function)
                        st.session_state.chain  = build_chains(
                            vector=st.session_state.vector,
                            model=model,
                            prompt=prompt,
                            contextualize_q_prompt=contextualize_q_prompt,
                        )
                        refresh_document_list()
                        st.success(f"Document '{selected_doc}' supprimé avec succès!")
                except Exception as e:
                    st.error(f"Erreur lors de la suppression: {e}")
        else:
            st.info("Aucun document disponible à supprimer.")



    # Document list section
    with st.expander("Documents disponibles", expanded=False):
        if st.session_state.available_documents:
            st.write(f"Nombre de documents: {len(st.session_state.available_documents)}")
            for doc in st.session_state.available_documents:
                st.text(f"📄 {doc}")
        else:
            st.info("Aucun document disponible.")




# Display chat history
st.header("Conversation")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chatbot - Interface d'entrée de texte
user_input = st.chat_input("Posez votre question ici...")

if user_input:
    # 1) Afficher / mémoriser le message utilisateur
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 2) Créer UN SEUL placeholder pour la réponse de l’assistant
    assistant_slot = st.empty()                 # ← conteneur “changeable”

    # 3) Boucle de génération
    while True:
        try:
            with st.spinner("Réflexion en cours..."):
                response = get_response(user_input, st.session_state["chat_history"], st.session_state.vector, st.session_state.chain)
            break                               # sortie si tout va bien
        except Exception as e:
            # on ré-écrit dans le même conteneur → l’ancien texte est remplacé
            with assistant_slot.chat_message("assistant"):
                st.write(f"Une erreur est survenue : {e}. Nouvelle tentative…")

    # 4) Réponse finale : on écrase le placeholder par le vrai contenu
    with assistant_slot.chat_message("assistant"):
        st.write(response)

    # 5) Historique pour RAG et pour la page
    st.session_state["chat_history"].extend(
        [HumanMessage(content=user_input), AIMessage(content=response)]
    )
    st.session_state.messages.append({"role": "assistant", "content": response})
