import streamlit as st, yaml, streamlit_authenticator as stauth
from yaml import SafeLoader
from chat_bot import load_vector_store, get_response, build_chains, model_and_embedding_function, create_prompt, create_contextualize_q_system_prompt
from Embedder import Embedder
from TextEmbedder import TextEmbedder
from EmbedderWithOcr import EmbedderWithOcr
from MultimodalEmbedder import MultimodalEmbedder
from langchain_core.messages import HumanMessage, AIMessage
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
import torch, types
torch.classes.__path__ = types.SimpleNamespace(_path=[])

# Page configuration
st.set_page_config(
    page_title="Chatbot Observatoire Astronomique", 
    page_icon=":astronaut:", 
    layout="wide",
    menu_items={
        'About': "Observatoire Astronomique - IMT Atlantique, campus de Brest"
    }
)
config_path = Path(__file__).resolve().parent.parent / "config.yaml"



# --- load config dynamically so new users persist -------------
authenticator = stauth.Authenticate(str(config_path))
# ------------- Register new user panel ------------------------

with st.sidebar:
    st.subheader("Gestion des utilisateurs")
    with st.expander("Actions", expanded=True):
        action = st.radio(
            "Sélectionnez une action",
            ["Se connecter", "Créer un compte", "Changer le mot de passe", "Modifier mes informations"],
            index=0
        )

        if action == "Se connecter":
            try:
                authenticator.login()
            except Exception as e:
                st.sidebar.error(e)

        elif action == "Créer un compte":
            try:
                email, username, name = authenticator.register_user(password_hint=False)
                if email and username and name:
                    st.sidebar.success(f"Utilisateur `{username}` enregistré avec succès !")
            except Exception as e:
                st.sidebar.error(e)

        elif action == "Changer le mot de passe":
            if st.session_state.get('authentication_status'):
                try:
                    if authenticator.reset_password(st.session_state.get('username')):
                        st.sidebar.success("Mot de passe modifié avec succès")
                except Exception as e:
                    st.sidebar.error(e)
            else:
                st.sidebar.info("Vous devez être connecté pour modifier le mot de passe.")

        elif action == "Modifier mes informations":
            if st.session_state.get('authentication_status'):
                try:
                    if authenticator.update_user_details(st.session_state.get('username')):
                        st.sidebar.success("Informations mises à jour avec succès")
                except Exception as e:
                    st.sidebar.error(e)
            else:
                st.sidebar.info("Vous devez être connecté pour modifier vos informations.")
    if st.session_state.get('authentication_status'):
            authenticator.logout()


if st.session_state.get('authentication_status') is False:
    st.sidebar.error('Nom d\'utilisateur ou mot de passe incorrect. Veuillez réessayer.')
elif st.session_state.get('authentication_status') is None:
    st.sidebar.warning('Merci de vous connecter pour accéder à l\'application.')
# --- Main application logic ---
elif st.session_state.get('authentication_status'):
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if api_key is None:
        raise RuntimeError("MISTRAL_API_KEY not set in environment")

    # Directory paths
    DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
    INDEX_DIR = Path(__file__).resolve().parent.parent / "faiss_index"
    HISTORY_DIR = Path(__file__).resolve().parent.parent / "chat_histories"

    # Create directories if they don't exist
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)

    # --- User input at the beginning of the session ---
    user_name_input = st.session_state.get("name")
    if "user_name" not in st.session_state:
        st.session_state["user_name"] = user_name_input
    
    user_role = st.session_state.get("roles")
  

    # --- Loading user history ---
    file_safe_name = st.session_state['user_name'].lower().replace(' ', '_')
    user_history_path = HISTORY_DIR / file_safe_name
    os.makedirs(user_history_path, exist_ok=True)

    # Initialize model and vector store
    model, embedding_function = model_and_embedding_function(api_key)
    contextualize_q_prompt = create_contextualize_q_system_prompt()
    prompt = create_prompt()

    if "vector" not in st.session_state:
        st.session_state.vector = load_vector_store(INDEX_DIR, embedding_function)
    if "chain" not in st.session_state:
        st.session_state.chain = build_chains(
            vector=st.session_state.vector,
            model=model,
            prompt=prompt,
            contextualize_q_prompt=contextualize_q_prompt,
        )

    # Initialize session state
    if "available_documents" not in st.session_state:
        st.session_state.available_documents = []
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "history_path" not in st.session_state:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        st.session_state.history_path = user_history_path / f"{ts}.json"

    # Function to refresh available documents
    def refresh_document_list():
        if DOCS_DIR.exists():
            st.session_state.available_documents = [f for f in os.listdir(DOCS_DIR) if f.endswith(('.pdf', '.PDF'))]
        else:
            st.session_state.available_documents = []

    # Function to save chat history
    def save_chat_history():
        try:
            history_data = {
                "messages": st.session_state.messages,
                "chat_history": [
                    {"type": "human", "content": msg.content} if isinstance(msg, HumanMessage)
                    else {"type": "ai", "content": msg.content}
                    for msg in st.session_state.chat_history
                ]
            }
            
            # Convert Path to string to ensure serializability
            history_path_str = str(st.session_state.history_path)
            
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(history_path_str), exist_ok=True)
            
            # Save with proper encoding
            with open(history_path_str, "w", encoding="utf-8") as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
                
            return True
        except Exception as e:
            st.error(f"Error saving chat history: {e}")
            return False

    # Initialize document list
    refresh_document_list()

    # --- Introduction ---
    st.markdown(f"""
    Salut **{st.session_state['user_name']}** ! 👋  
    Je suis ton assistant virtuel pour l'observatoire astronomique. 😊  
    Je peux t'aider avec :
    - L'utilisation des équipements
    - Les objets astronomiques que tu observes

    Pose-moi une question ou demande-moi de l'aide !
    """)

    # Add a title
    st.title("Bienvenue dans l'Observatoire Astronomique 🚀")

    with st.sidebar:
        st.subheader("Historique de la discussion")
        with st.expander("Historique de la discussion", expanded=False):
            if user_role == "admin":
                # List all user directories
                user_directories = [d for d in os.listdir(HISTORY_DIR) if os.path.isdir(os.path.join(HISTORY_DIR, d))]
                if not user_directories:
                    st.info("Aucun utilisateur enregistré pour le moment.")
                else:
                    selected_user = st.selectbox("Choisissez un utilisateur :", user_directories, index=None, placeholder="— choisir un utilisateur —") 
                    if selected_user:
                        user_history_path = HISTORY_DIR / selected_user
            # Get list of history files for this user
            history_files = []
            if user_history_path.exists():
                history_files = sorted([f for f in os.listdir(user_history_path) if f.endswith('.json')], 
                                        reverse=True)  # Most recent first
            
            if history_files:
                selected_history = st.selectbox(
                    "Charger une conversation précédente:",
                    options=history_files,
                    index=None,
                    placeholder="— choisir une conversation —"
                )
                
                if selected_history and st.button("Charger cette conversation"):
                    try:
                        history_file_path = user_history_path / selected_history
                        with open(history_file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            st.session_state.messages = data.get("messages", [])
                            st.session_state.chat_history = [
                                HumanMessage(content=msg["content"]) if msg["type"] == "human"
                                else AIMessage(content=msg["content"])
                                for msg in data.get("chat_history", [])
                            ]
                            st.session_state.history_path = history_file_path
                            st.success(f"Conversation chargée avec succès!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Erreur lors du chargement: {e}")
            else:
                st.info("Aucun historique disponible pour cet utilisateur.")
                
            if st.button("Nouvelle conversation"):
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                st.session_state.history_path = user_history_path / f"{ts}.json"
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.success("Nouvelle conversation créée!")
                st.rerun()

        st.subheader("Gérer vos documents")
        # Document upload section
        with st.expander("Télécharger un document", expanded=False):
            pdf = st.file_uploader("Téléchargez vos documents ici et cliquez sur 'Traiter'")
            options = ["Text only embedder", "Embedder with OCR", "Multimodal embedder"]
            if pdf is not None:
                selected_embedder = st.selectbox(
                    "Choisissez l'embedder", 
                    options, 
                    index=None, 
                    placeholder="— choisir un embedder —"
                )
                
                embedder = Embedder(api_key)  # Default
                if selected_embedder == "Text only embedder":
                    embedder = TextEmbedder(api_key)
                elif selected_embedder == "Embedder with OCR":
                    embedder = EmbedderWithOcr(api_key)
                elif selected_embedder == "Multimodal embedder":
                    embedder = MultimodalEmbedder(api_key)
                add_permanently = False
                if user_role == "admin":
                    add_permanently = st.checkbox("Ajouter un document de manière permanente")
                if st.button("Traiter", key="process_button"):
                    try:
                        # Build destination path
                        save_path = os.path.join(DOCS_DIR, pdf.name)
                        if add_permanently:
                            with open(save_path, "wb") as f:
                                f.write(pdf.read())
                            st.success(f"Fichier enregistré dans : {save_path}")
                        with st.spinner("Indexation du document…"):
                            # Process the document
                            embedder.embed(save_path, st.session_state.vector, save=add_permanently)
                            st.session_state.chain = build_chains(
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
                    index=None,
                    placeholder="— choisir un document —" 
                )
                delete_permanently = False
                if user_role == "admin":
                    add_permanently = st.checkbox("Supprimer le document de manière permanente")
                if selected_doc and st.button("Supprimer", key="delete_btn"):
                    try:
                        with st.spinner("Suppression en cours..."):
                            file_path = os.path.join(DOCS_DIR, selected_doc)
                            Embedder(api_key).delete_document(selected_doc, st.session_state.vector, save=delete_permanently)
                            if delete_permanently:
                                os.remove(file_path)
                            st.session_state.chain = build_chains(
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
            st.markdown("---")
            
        st.subheader("🛠️ Espace responsable")
        show_admin = st.checkbox("Afficher toutes les discussions")

    # Display chat history
    st.header("Conversation")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chatbot - Text input interface
    user_input = st.chat_input("Posez votre question ici...")

    if user_input:
        # 1) Display/store user message
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # 2) Create ONE placeholder for assistant response
        assistant_slot = st.empty()

        # 3) Generation loop
        while True:
            try:
                with st.spinner("Réflexion en cours..."):
                    response = get_response(user_input, st.session_state["chat_history"], st.session_state.vector, st.session_state.chain)
                break  # Exit if all goes well
            except Exception as e:
                # Rewrite in the same container → old text is replaced
                with assistant_slot.chat_message("assistant"):
                    st.write(f"Une erreur est survenue : {e}. Nouvelle tentative…")

        # 4) Final response: overwrite placeholder with real content
        with assistant_slot.chat_message("assistant"):
            st.write(response)

        # 5) History for RAG and for the page
        st.session_state["chat_history"].extend(
            [HumanMessage(content=user_input), AIMessage(content=response)]
        )
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Save chat history after each exchange
        save_chat_history()

    # --- Admin Space ---
    if show_admin:
        st.markdown("## 🔍 Historique complet des utilisateurs")
        
        # List all user directories
        user_directories = [d for d in os.listdir(HISTORY_DIR) if os.path.isdir(os.path.join(HISTORY_DIR, d))]
        
        if not user_directories:
            st.info("Aucun utilisateur enregistré pour le moment.")
        else:
            selected_user = st.selectbox("Choisissez un utilisateur :", user_directories)
            
            if selected_user:
                user_dir = os.path.join(HISTORY_DIR, selected_user)
                history_files = [f for f in os.listdir(user_dir) if f.endswith(".json")]
                
                if not history_files:
                    st.info(f"Aucune discussion enregistrée pour {selected_user}.")
                else:
                    selected_file = st.selectbox("Choisissez une conversation :", history_files)
                    
                    if selected_file:
                        try:
                            with open(os.path.join(user_dir, selected_file), "r", encoding="utf-8") as f:
                                data = json.load(f)
                                messages = data.get("messages", [])
                                st.markdown(f"### 💬 Conversation de `{selected_user}` - {selected_file}")

                                for msg in messages:
                                    role = "👤 Utilisateur" if msg["role"] == "user" else "🤖 Assistant"
                                    st.markdown(f"**{role}** : {msg['content']}")
                                    st.markdown("---")
                        except Exception as e:
                            st.error(f"Erreur lors du chargement de {selected_file} : {e}")
