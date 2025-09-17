# rag_backend.py

import os
import time
import threading
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()

# Spinner flag
model_loaded = False

# Spinner function
def spinner(msg="Loading model..."):
    while not model_loaded:
        for cursor in '|/-\\':
            print(f'\r{msg} {cursor}', end='', flush=True)
            time.sleep(0.1)

# Start spinner thread
spinner_thread = threading.Thread(target=spinner)
spinner_thread.start()

# Load the SentenceTransformer model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Stop spinner
model_loaded = True
spinner_thread.join()
print("\n✅ Model loaded successfully!")

# Load vector store
PERSIST_DIRECTORY = "vector_db"
if not os.path.exists(PERSIST_DIRECTORY):
    raise FileNotFoundError(f"Vector store directory '{PERSIST_DIRECTORY}' not found.")
vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Define prompt
template = """
You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer, just say that you don't know. Be very detailed and comprehensive in your answer, providing a detailed on the given context.

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

# Inference function
def inference_with_rag(query):
    result = rag_chain.invoke({"input": query})
    return result['answer']
