import os
from collections import Counter
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from pdf_loader import load_and_split_pdf


load_dotenv()


def setup_vectorstore(pdf_path: str, index_name: str):
    """
    Creates or connects to a Pinecone index and uploads PDF embeddings
    """
    embeddings = OpenAIEmbeddings()

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    existing_indexes = [idx["name"] for idx in pc.list_indexes().get("indexes", [])]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=os.getenv("PINECONE_ENV")
            )
        )

    documents = load_and_split_pdf(pdf_path)

    vectorstore = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=index_name
    )

    return vectorstore


def get_qa_chain(vectorstore):
    """
    Builds a RetrievalQA chain
    """
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo"
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=False
    )

    return qa_chain


def get_consensus_answer(question: str, qa_chain):
    """
    Generates 4 paraphrased questions, gets answers,
    and returns the most repeated answer
    """
    variations = [
        question,
        f"Explain this: {question}",
        f"What is meant by {question}?",
        f"Give details about {question}"
    ]

    answers = []

    for q in variations:
        result = qa_chain.invoke({"query": q})
        answers.append(result["result"])

    most_common_answer = Counter(answers).most_common(1)[0][0]
    return most_common_answer


