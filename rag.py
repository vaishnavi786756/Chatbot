import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from .env
api_key = os.getenv("GROQ_API_KEY")

# Set up Streamlit UI
st.set_page_config(page_title="Chat with Your Documents (RAG using Groq)", layout="wide")
st.title("📄 Chat with Your Documents (RAG using Groq)")

# File uploader
uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

if not api_key:
    st.error("❌ GROQ API key not found. Please add it to your .env file.")
elif uploaded_files:
    documents = []
    
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name
        
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        else:
            loader = TextLoader(temp_path)
        
        documents.extend(loader.load())
        os.remove(temp_path)
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    # Embed documents using Hugging Face embeddings (no OpenAI dependency)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    retriever = vectorstore.as_retriever()
    
    # Set up LLM (Groq API)
    llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=api_key)# Using Groq’s Llama 3 model
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    
    st.success("✅ Documents processed successfully. You can now ask questions!")

    query = st.text_input("Ask a question based on your documents")

    if st.button("Submit"):
        if query:
            response = qa_chain.run(query)
            st.write("### Answer:")
            st.write(response)
        else:
            st.warning("⚠️ Please enter a question before submitting.")
