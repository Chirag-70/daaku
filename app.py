import streamlit as st
from rag_system import (
    setup_vectorstore,
    get_qa_chain,
    get_consensus_answer
)

st.set_page_config(
    page_title="Consensus RAG with Pinecone",
    layout="wide"
)

st.title("📄 PDF RAG Chatbot (Pinecone + Consensus)")

with st.spinner("Initializing vector database..."):
    vectorstore = setup_vectorstore()
    qa_chain = get_qa_chain(vectorstore)

query = st.text_input("Ask a question from the PDF")

if query:
    with st.spinner("Generating answer..."):
        final_answer, all_answers = get_consensus_answer(query, qa_chain)

    st.subheader("✅ Final Answer (Most Reliable)")
    st.write(final_answer)

    with st.expander("🔍 Show all paraphrased Q&A"):
        for q, a in all_answers:
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")

