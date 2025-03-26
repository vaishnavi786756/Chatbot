import streamlit as st
import asyncio
import os
import sys
import tempfile
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ Fix: Ensure correct event loop policy on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ✅ Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# ✅ Streamlit UI setup
st.set_page_config(page_title="Chat with Your Documents (RAG using Groq)", layout="wide")
st.title("📄 Chat with Your Documents (RAG using Groq)")

# ✅ File uploader
uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

if not api_key:
    st.error("❌ GROQ API key not found. Please add it to your .env file.")
elif uploaded_files:
    documents = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt" if uploaded_file.name.endswith(".txt") else ".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        # Load documents
        try:
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(temp_path)
            else:
                loader = TextLoader(temp_path)

            documents.extend(loader.load())
        finally:
            os.remove(temp_path)  # ✅ Proper cleanup of temporary files

    # ✅ Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # ✅ Use a stable embedding model (Alternative to MiniLM)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

    # ✅ Create FAISS vector store
    vectorstore = FAISS.from_documents(texts, embeddings)
    retriever = vectorstore.as_retriever()

    # ✅ Set up Groq's Llama 3 model
    llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=api_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    st.success("✅ Documents processed successfully. You can now ask questions!")

    query = st.text_input("Ask a question based on your documents")

    if st.button("Submit"):
        if query:
            response = qa_chain.invoke(query)
            st.write("### Answer:")
            st.write(response)
        else:
            st.warning("⚠️ Please enter a question before submitting.")
