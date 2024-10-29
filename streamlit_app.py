import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os

# Initialize the Streamlit app
st.title("Document Comparer with ChatGroq LLM")

# Initialize document loader, embeddings, and FAISS vector store
@st.cache_resource  # Use cache to avoid re-initializing on every interaction
def initialize_faiss_vectorstore():
    # Load and embed documents
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS(embeddings)
    return vectorstore

vectorstore = initialize_faiss_vectorstore()

# Upload and load documents
uploaded_files = st.file_uploader("Upload PDF documents", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    for uploaded_file in uploaded_files:
        with open(f"temp_{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.read())
        loader = PyPDFLoader(f"temp_{uploaded_file.name}")
        documents = loader.load()

        # Embed and store documents in FAISS
        vectorstore.add_documents(documents)

    st.success("Documents uploaded and embedded successfully!")

# Initialize ChatGroq LLM
@st.cache_resource
def initialize_chatgroq():
    return ChatGroq(model="llama3-70b-8192")

llm = initialize_chatgroq()

# Set up RetrievalQA chain with LangChain
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA(llm=llm, retriever=retriever)

# Define comparison function
def compare_documents(query):
    response = qa_chain({"query": query})
    return response["result"]

# Interface to compare documents
st.subheader("Compare Documents")
query = st.text_input("Enter your comparison query:")

if st.button("Compare"):
    if not query:
        st.warning("Please enter a query to compare.")
    else:
        result = compare_documents(query)
        st.write("Comparison Result:")
        st.write(result)
