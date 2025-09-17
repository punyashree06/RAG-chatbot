from mcp.server.fastmcp import FastMCP
import google.generativeai as genai
import os
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

# Define the persistence directory for your vector store
PERSIST_DIRECTORY = "vector_db"

load_dotenv()
genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))

# 1. Initialize the embedding model (only once)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Load the persisted vector store (only once)
if not os.path.exists(PERSIST_DIRECTORY):
    print(f"Error: Vector store directory '{PERSIST_DIRECTORY}' not found.")
else:
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding
    )

# 3. Create the retriever (only once)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# 4. Define the RAG prompt (only once)
template = """
You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer, just say that you don't know. Be very detailed and comprehensive in your answer, providing a thorough summary based on the given context.

Context:
{context}

Question:
{input}

Helpful Answer:"""
prompt = PromptTemplate.from_template(template)

# 5. Initialize the LLM (only once)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.7,
    google_api_key=os.environ.get('GOOGLE_API_KEY')
)

# 6. Build the RAG chain (only once)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


mcp = FastMCP("dataagent")

@mcp.tool()
def dataagent(query):
    """
    Performs a RAG query using the pre-loaded chain.
    """
    print(f"\nQuerying: '{query}'")
    if 'rag_chain' in globals():
        result = rag_chain.invoke({"input": query})
        return result['answer']
    else:
        return "RAG chain not initialized. Check for errors during setup."
