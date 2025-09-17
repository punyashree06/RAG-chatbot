# app.py

import streamlit as st
from rag_backend import inference_with_rag  # Your inference function
from visualizer import generate_visualization

st.set_page_config(page_title="RAG Chatbot", layout="centered")

# Title
st.title("💬 LTS Chatbot")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Append user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        # Call your inference function
        response = inference_with_rag(user_input)
    
    # Example image path to append with assistant response
    response_image=generate_visualization(response)
      # Make sure this path is correct relative to app.py

    # Append assistant response with optional image
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response,
        "image": response_image  # Set to None or "" if no image
    })

# Display chat history messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "image" in msg and msg["image"]:
            st.image(msg["image"])
