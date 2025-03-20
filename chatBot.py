from langchain_mistralai.chat_models import ChatMistralAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage ,AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(
    requests_per_second=1,  # <-- Super slow! We can only make a request once every 10 seconds!!
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=10,  # Controls the maximum burst size.
)
from langchain.chains.history_aware_retriever import create_history_aware_retriever


import os


workspace_path = os.getcwd()  # Get current working directory
faiss_index_path = os.path.join(workspace_path, "faiss_index")

embedding_function = MistralAIEmbeddings(model="mistral-embed", mistral_api_key="wNVyBAARBAah94Jwl9WtLFpGT7sM9xFj")
vector = FAISS.load_local(faiss_index_path, embeddings=embedding_function, allow_dangerous_deserialization=True)

# Define LLM
model = ChatMistralAI(mistral_api_key="wNVyBAARBAah94Jwl9WtLFpGT7sM9xFj",rate_limiter=rate_limiter,model="mistral-large-latest")

retreiver_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("human","{input}"),
    ("human","Utiliser l'hitorique de la conversation pour bien contextualiser tes réponses.")
])

# Define a retriever interface
retriever = vector.as_retriever()
history_aware_retriever = create_history_aware_retriever(
    llm= model,
    retriever=retriever,
    prompt=retreiver_prompt
)

# Define prompt template

def create_prompt():
    """
    Returns a prompt instructed to produce a rephrased question based on the user's
    last question, but referencing previous messages (chat history).
    """
    system_instruction = """Tu es un ChatBot qui va répondre aux questions des utilisateurs d'observatoire astronomique de l'école IMT ATlantique campus de Brest.\
          Si l'utilisateur pose des questions sur l'observatoire. Tu doit répondre en se basant sur les données fournis.\
          Si tu n'arrive pas a trouver l'onformations tu dit que tu ne sait pas.\
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
    while True :
        try:
            response = retrieval_chain.invoke({"input": user_input,"chat_history":chat_history})
            return response['answer']
        except :
            continue
        break


print(get_response("bonjour",[]))