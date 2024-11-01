import streamlit as st
import pdfplumber
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.schema import Document
from difflib import unified_diff

# Initialize ChatGroq model
groq_api_key = "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"
model = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Initialize HuggingFace Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create Streamlit App
st.title("Document Comparer")

# Upload documents
uploaded_files = st.file_uploader("Choose PDF files (contracts) to compare", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    documents = []
    
    # Load and process the documents
    for uploaded_file in uploaded_files:
        with pdfplumber.open(uploaded_file) as pdf_document:
            doc_text = ""
            for page in pdf_document.pages:
                page_text = page.extract_text() or ""  # Extract text from each page
                doc_text += page_text
            
            # Wrap the extracted text in a LangChain Document
            documents.append(Document(page_content=doc_text, metadata={"name": uploaded_file.name}))

    # Compare documents
    if st.button("Compare Documents"):
        if len(documents) < 2:
            st.warning("Please upload at least two documents to compare.")
        else:
            # Compare each document with every other document
            results = []
            for i in range(len(documents)):
                for j in range(i + 1, len(documents)):
                    doc1_content = documents[i].page_content.splitlines(keepends=True)
                    doc2_content = documents[j].page_content.splitlines(keepends=True)

                    # Use difflib to find differences
                    diff = list(unified_diff(doc1_content, doc2_content, 
                                              fromfile=documents[i].metadata['name'], 
                                              tofile=documents[j].metadata['name'], 
                                              lineterm=''))

                    # Format the differences for display
                    if diff:
                        results.append(f"**Differences between {documents[i].metadata['name']} and {documents[j].metadata['name']}:**")
                        results.append("```diff")
                        results.extend(diff)
                        results.append("```")
                    else:
                        results.append(f"No differences found between {documents[i].metadata['name']} and {documents[j].metadata['name']}.")

            # Display the results
            st.subheader("Comparative Results")
            for result in results:
                st.write(result)

        st.success("Comparison completed!")
