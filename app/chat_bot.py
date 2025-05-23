from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage ,AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from dotenv import load_dotenv
from pathlib import Path
import os

def model_and_embedding_function(api_key):
    # Create embedding function using Mistral AI's embedding model
    embedding_function = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)

    # Initialize the language model with Mistral AI
    model = ChatMistralAI(mistral_api_key=api_key, model="mistral-large-latest")
    return model, embedding_function
def create_contextualize_q_system_prompt():
    # System prompt for contextualizing questions based on chat history
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    # Create a chat prompt template for contextualizing questions
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),  # Placeholder for chat history
            ("human", "{input}"),  # Placeholder for user input
        ]
    )
    return contextualize_q_prompt

def load_vector_store(index_dir: Path,embedding_function):
    if not index_dir.exists() or len(os.listdir(index_dir)) == 0 :
        return None
    return FAISS.load_local(index_dir, embeddings=embedding_function,allow_dangerous_deserialization=True)




# Define prompt template function

def create_prompt():
    """
    Returns a prompt instructed to produce a rephrased question based on the user's
    last question, but referencing previous messages (chat history).
    """
    # System instruction in French for the astronomy observatory chatbot
    system_instruction = """Tu es un ChatBot qui va répondre aux questions des utilisateurs d'observatoire astronomique de l'école IMT ATlantique campus de Brest.\
        Si l'utilisateur pose des questions sur l'observatoire. Tu doit répondre en se basant seulement sur les données fournis.\
        Tes réponses doivent être courtes et concises et bien structurées.\
        Veiller à que tous les formules mathématique sont bien formaté pour LaTeX. Et vérifier cela avant d'envoyer ta réponse.\
        Si l'utilisateur ne demande pas des formules mathématiques, tu ne dois pas en fournir.\
        Si l'utilisateur pose des questions sur quelque chose autre que l'observatoire, tu refuses de répondre.\
        Répond toujours en français.\
        Utiliser le context : {context}"""

    # Create chat prompt template with system instruction, chat history and user input
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")])
    return prompt

def build_chains(vector, model,prompt,contextualize_q_prompt):
    retriever  = vector.as_retriever()
    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=contextualize_q_prompt
    )
    document_chain = create_stuff_documents_chain(model, prompt)
    return create_retrieval_chain(history_aware_retriever, document_chain)



def get_response(user_input: str, chat_history: list, vector, chain):
    if vector is None:
        return ("Je n'ai trouvé aucun document. "
        "Veuillez d'abord en téléverser dans la barre latérale.")
    response = chain.invoke({"input": user_input, "chat_history": chat_history})
    return response['answer']


if __name__ == '__main__' :
    chat_history = []
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    model, embedding_fn = model_and_embedding_function(api_key)
    vector = load_vector_store(Path("faiss_index"), embedding_fn)
    prompt = create_prompt()
    contextual_prompt = create_contextualize_q_system_prompt()
    chain = build_chains(vector, model, prompt, contextual_prompt)
    while True:
        user_input = input("user : ")
        while True:
            try:
                response =  get_response(user_input,chat_history,vector,chain)
            except Exception as e :
                print(e)
                continue
            break
        chat_history.extend(
        [
            HumanMessage(content=user_input),
            AIMessage(content=response),
        ]
        )

        print("assisatnt : ",response)