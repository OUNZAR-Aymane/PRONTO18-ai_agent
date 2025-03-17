from langchain_mistralai.chat_models import ChatMistralAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_mistralai.embeddings import MistralAIEmbeddings
import os
from dotenv import load_dotenv

embedding_function = MistralAIEmbeddings(model="mistral-embed", mistral_api_key="wNVyBAARBAah94Jwl9WtLFpGT7sM9xFj")
vector = FAISS.load_local(r"C:\Users\ounza\IMT\Semestre 6\PRONTO\ai-agent\faiss_index\index.faiss", embeddings=embedding_function, allow_dangerous_deserialization=True)

# Define a retriever interface
retriever = vector.as_retriever()
# Define LLM
model = ChatMistralAI(mistral_api_key="wNVyBAARBAah94Jwl9WtLFpGT7sM9xFj")
# Define prompt template
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {Comment}""")

# Create a retrieval chain to answer questions
document_chain = create_stuff_documents_chain(model, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)
response = retrieval_chain.invoke({"input": "Comment marche le coronographe de Lyot?"})
print(response["answer"])