import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os
from difflib import ndiff
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz

# Initialize the Streamlit app
st.title("Document Comparer!")
llm = ChatGroq(groq_api_key="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", model_name="Llama3-8b-8192")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 450, chunk_overlap = 100)

st.title("Contract Document Comparer")

uploaded_file1 = st.file_uploader("Upload Document 1", type=["txt", "pdf", "docx"])
uploaded_file2 = st.file_uploader("Upload Document 2", type=["txt", "pdf", "docx"])

def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

if uploaded_file1 and uploaded_file2:
    doc1 = read_pdf(uploaded_file1)
    doc2 = read_pdf(uploaded_file2)

    chunks1 = text_splitter.split_text(doc1)
    chunks2 = text_splitter.split_text(doc2)

    embeddings1 = embeddings.embed_texts(chunks1)
    embeddings2 = embeddings.embed_texts(chunks2)

    vector_db1 = FAISS.from_embeddings(embeddings1)
    vector_db2 = FAISS.from_embeddings(embeddings2)

    retrieval_chain = RetrievalQA(llm=llm, retriever=vector_db1.as_retriever())

    results1 = retrieval_chain.run(vector_db1)
    results2 = retrieval_chain.run(vector_db2)

    differences = find_differences(results1, results2)
    st.write(differences)

def find_differences(results1, results2):
    differences = []
    for chunk1, chunk2 in zip(results1, results2):
        if chunk1 != chunk2:
            differences.append((chunk1, chunk2))
    return differences
