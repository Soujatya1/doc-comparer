import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os
from difflib import ndiff
from langchain.chains.question_answering import load_qa_chain
fro langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize the Streamlit app
st.title("Document Comparer!")
llm = ChatGroq(groq_api_key="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", model_name="Llama3-8b-8192")

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 450, chunk_overlap = 100)
chunks = text_splitter.split_text(text_splitter)

vector_db = FAISS.from_texts(chunks)

retrieval_chain = RetrievalQA(llm = llm, retriever = vector_db.as_retriever())

import streamlit as st

st.title("Contract Document Comparer")

doc1 = st.text_area("Document 1")
doc2 = st.text_area("Document 2")

if st.button("Compare"):
    chunks1 = text_splitter.split_text(doc1)
    chunks2 = text_splitter.split_text(doc2)

    embeddings1 = embeddings.embed_texts(chunks1)
    embeddings2 = embeddings.embed_texts(chunks2)

    vector_db1 = FAISS.from_embeddings(embeddings1)
    vector_db2 = FAISS.from_embeddings(embeddings2)

    results1 = retrieval_chain.run(vector_db1)
    results2 = retrieval_chain.run(vector_db2)

    differences = find_differences(results1, results2)
    st.write(differences)
