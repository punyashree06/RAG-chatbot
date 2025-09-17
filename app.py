import os
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain

# --- SETUP ---
load_dotenv()
PERSIST_DIRECTORY = "vector_db"

# Initialize embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load vector store
if not os.path.exists(PERSIST_DIRECTORY):
    st.error(f"Vector store directory '{PERSIST_DIRECTORY}' not found.")
    st.stop()
else:
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding
    )

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Define RAG prompt
template = """
You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer, just say that you don't know. Be very detailed and comprehensive in your answer, providing a thorough summary based on the given context.

Context:
{context}

Question:
{input}

Helpful Answer:"""
prompt = PromptTemplate.from_template(template)

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.7,
    google_api_key=os.environ.get('GOOGLE_API_KEY')
)

# Build RAG chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- STREAMLIT UI ---
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🧠 RAG Chatbot with LangChain + Gemini")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat input
user_input = st.chat_input("Ask me anything...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        result = rag_chain.invoke({"input": user_input})
        answer = result['answer']
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
