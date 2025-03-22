import streamlit as st
from chatBot import get_response
from embedder import embedd, load_and_split
from langchain_core.messages import HumanMessage ,AIMessage

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


if "messages" not in st.session_state:
    st.session_state.messages = []

# Titre de la page
st.set_page_config(page_title="Chatbot Observatoire Astronomique", page_icon=":astronaut:", layout="wide")


# Ajouter un titre
st.title("Bienvenue dans l'Observatoire Astronomique 🚀")

# Ajouter une introduction
st.markdown("""
    Salut! Je suis ton assistant virtuel pour l'observatoire astronomique. 😊
    Je peux t'aider avec :
    - L'utilisation des équipements.
    - Les objets astronomiques que tu observes.
    Pose-moi une question ou demande-moi de l'aide!
""")

# Ajouter un bouton pour entrer les documents
with st.sidebar:
    st.subheader("Vos documents")
    pdf = st.file_uploader("Téléchargez vos documents ici et cliquez sur 'Traiter'")
    if st.button("Traiter"):
        chunks = load_and_split(pdf)
        embedd(chunks)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chatbot - Interface d'entrée de texte
user_input = st.chat_input("Pose ta question ici...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Ici, tu pourrais ajouter un modèle NLP pour répondre à la question ou simuler une réponse intelligente.
    with st.spinner("Réponse en cours de génération"):
        response = get_response(user_input,st.session_state['chat_history'])
        st.session_state["chat_history"].append(HumanMessage(content=user_input))
        st.session_state["chat_history"].append(AIMessage(content=response))
    with st.chat_message("assistant"):
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})




