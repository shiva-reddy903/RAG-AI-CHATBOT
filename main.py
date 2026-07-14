import os
import streamlit as st

from document_processor import create_vector_database
from chatbot import ask_question
from settings import UPLOAD_FOLDER

st.set_page_config(
    page_title="Smart Doc Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Smart Doc Chatbot")

st.write("Upload PDF documents and ask questions from them.")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

with st.sidebar:

    st.header("📄 Upload Documents")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        if st.button("📤 Upload Document"):

            save_path = os.path.join(
                UPLOAD_FOLDER,
                uploaded_file.name
            )

            with open(save_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

            with st.spinner("Updating knowledge base..."):
                create_vector_database()

            st.success("Knowledge base updated successfully!")

    st.divider()

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your documents...")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Thinking..."):

        answer = ask_question(question)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
    