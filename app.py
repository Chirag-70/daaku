# app.py
import streamlit as st
from rag_system import load_and_split_pdf, create_vectorstore, create_rag_chain

st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("📄 PDF RAG Chatbot")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        # Load PDF and split text
        docs = load_and_split_pdf(uploaded_file)
        vectorstore = create_vectorstore(docs)
        qa_chain, memory = create_rag_chain(vectorstore)

    st.success("PDF processed! You can now ask questions.")

    # Ask questions
    query = st.text_input("Ask something about the PDF:")
    if query:
        with st.spinner("Generating answer..."):
            answer = qa_chain.run(query)
        st.write("**Answer:**", answer)



