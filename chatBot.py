from langchain_mistralai.chat_models import ChatMistralAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_mistralai.embeddings import MistralAIEmbeddings
import os


workspace_path = os.getcwd()  # Get current working directory
faiss_index_path = os.path.join(workspace_path, "faiss_index")

embedding_function = MistralAIEmbeddings(model="mistral-embed", mistral_api_key="wNVyBAARBAah94Jwl9WtLFpGT7sM9xFj")
vector = FAISS.load_local(faiss_index_path, embeddings=embedding_function, allow_dangerous_deserialization=True)

# Define a retriever interface
retriever = vector.as_retriever()
# Define LLM
model = ChatMistralAI(mistral_api_key="wNVyBAARBAah94Jwl9WtLFpGT7sM9xFj")
# Define prompt template
prompt = ChatPromptTemplate.from_template("""Tu es un ChatBot qui va répondre aux questions des utilisateurs d'observatoire astronomique de l'école IMT ATlantique campus de Brest. Tu répond seuelemnt sur la base des données que tu as. Tu tu ne trouve pas la réponse, dit le. Répond toujours en français, même si les données sont en anglais.

<context>
"{context}"
</context>

Question: {input}""")

# Create a retrieval chain to answer questions
document_chain = create_stuff_documents_chain(model, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)
response = retrieval_chain.invoke({"input": "Comment configurer la carte cu ciel dans le logiciel prism"})
print(response["answer"])