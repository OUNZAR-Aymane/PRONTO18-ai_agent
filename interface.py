import streamlit as st


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
    - Les phénomènes astronomiques en cours.
    - Et plus encore! ✨

    Pose-moi une question ou demande-moi de l'aide!
""")

# Ajouter un bouton pour entrer les documents
with st.sidebar:
    st.subheader("Vos documents")
    st.file_uploader("Téléchargez vos documents ici et cliquez sur 'Traiter'")
    st.button("Traiter")

# Chatbot - Interface d'entrée de texte
user_input = st.text_input("Pose ta question ici:")

if user_input:
    # Réponse simple pour l'instant
    st.write(f"Tu as demandé: {user_input}")
    # Ici, tu pourrais ajouter un modèle NLP pour répondre à la question ou simuler une réponse intelligente.
    st.write("")

# Ajouter un footer
st.markdown("""
    ---
    Fait avec ❤️ par l'équipe de l'observatoire.
""")



