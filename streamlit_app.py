import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os
import difflib
from langchain.chains.question_answering import load_qa_chain

# Initialize the Streamlit app
st.title("Document Comparer!")
llm = ChatGroq(groq_api_key="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", model_name="Llama3-8b-8192")

# Function to initialize FAISS vector store without caching
def initialize_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# Function to clean up temporary files
def cleanup_temp_files(uploaded_files):
    for uploaded_file in uploaded_files:
        temp_file_path = f"temp_{uploaded_file.name}"
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Function to compare text blocks between two documents
def compare_text_blocks(text1, text2):
    blocks1 = set(text1.split('\n'))  # Split text into blocks
    blocks2 = set(text2.split('\n'))  # Split text into blocks

    unique_to_doc1 = blocks1 - blocks2  # Blocks in Doc A not in Doc B
    unique_to_doc2 = blocks2 - blocks1  # Blocks in Doc B not in Doc A

    return unique_to_doc1, unique_to_doc2

# Upload and load documents
uploaded_files = st.file_uploader("Upload PDF documents", accept_multiple_files=True, type=["pdf"])

documents = []
texts = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        with open(f"temp_{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.read())
        loader = PyPDFLoader(f"temp_{uploaded_file.name}")
        doc = loader.load()
        documents.extend(doc)
        texts.append(" ".join([d.page_content for d in doc]))  # Collect text from documents

    # Initialize FAISS vector store with all documents (no caching)
    vectorstore = initialize_vectorstore(documents)
    st.success("Documents uploaded and embedded successfully!")

    # Set up RetrievalQA chain with LangChain
    retriever = vectorstore.as_retriever()
    combine_chain = load_qa_chain(llm, chain_type="stuff")
    qa_chain = RetrievalQA(combine_documents_chain=combine_chain, retriever=retriever)

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

            # Highlight differences between the first two documents if they exist
            if len(texts) >= 2:
                unique_to_doc1, unique_to_doc2 = compare_text_blocks(texts[0], texts[1])
                st.subheader("Unique Text Blocks:")
                
                if unique_to_doc1:
                    st.markdown("**Unique to Document A:**")
                    for block in unique_to_doc1:
                        st.write(f"- {block}")
                else:
                    st.write("No unique blocks in Document A.")
                
                if unique_to_doc2:
                    st.markdown("**Unique to Document B:**")
                    for block in unique_to_doc2:
                        st.write(f"- {block}")
                else:
                    st.write("No unique blocks in Document B.")

    # Cleanup temporary files
    cleanup_temp_files(uploaded_files)
else:
    st.warning("Please upload documents to enable comparison.")
