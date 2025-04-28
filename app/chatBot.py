from langchain_mistralai.chat_models import ChatMistralAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage ,AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from dotenv import load_dotenv

import os
load_dotenv()


from pathlib import Path
faiss_index_path = Path("..") / "faiss_index"

api_key = os.environ["MISTRAL_API_KEY"]
embedding_function = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)
vector = FAISS.load_local(faiss_index_path, embeddings=embedding_function, allow_dangerous_deserialization=True)

# Define LLM
model = ChatMistralAI(mistral_api_key=api_key,model="mistral-large-latest")

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Define a retriever interface
retriever = vector.as_retriever()
history_aware_retriever = create_history_aware_retriever(
    llm= model,
    retriever=retriever,
    prompt=contextualize_q_prompt
)

# Define prompt template

def create_prompt():
    """
    Returns a prompt instructed to produce a rephrased question based on the user's
    last question, but referencing previous messages (chat history).
    """
    system_instruction = """Tu es un ChatBot qui va répondre aux questions des utilisateurs d'observatoire astronomique de l'école IMT ATlantique campus de Brest.\
        Si l'utilisateur pose des questions sur l'observatoire. Tu doit répondre en se basant seulement sur les données fournis.\
        Respecter toujours la format LaTex.\
        Écris toujours les formules mathématiques en les entourant de $ pour qu'elles soient compilées en format LaTeX. Au lieu d'écrire [formule en LaTex] écrit $ formule en LaTex $\
        Si tu n'arrive pas a trouver l'onformations tu dit que tu ne sais pas.\
        Répond toujours en français.\
        Utiliser le context : {context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")])
    return prompt
prompt = create_prompt()


# Create a retrieval chain to answer questions
document_chain = create_stuff_documents_chain(model, prompt)
retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)
def get_response(user_input,chat_history):
    response = retrieval_chain.invoke({"input": user_input,"chat_history":chat_history})
    return response['answer']

if __name__ == '__main__' :
    chat_history = []
    while True:
        user_input = input("user : ")
        while True:
            try:
                response =  get_response(user_input,chat_history)
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