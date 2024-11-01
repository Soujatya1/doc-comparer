import streamlit as st
import pdfplumber
import faiss
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.schema import Document

# Initialize ChatGroq model
groq_api_key = "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"
model = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Initialize HuggingFace Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create Streamlit App
st.title("Document Comparer with FAISS")

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
            # Create FAISS vector store
            embeddings = []
            for doc in documents:
                embedding = embedding_model.embed([doc.page_content])[0]  # Embed document content
                embeddings.append(embedding)  # Store the embedding
            
            # Convert embeddings to numpy array for FAISS
            embedding_array = np.array(embeddings, dtype=np.float32)

            # Create a FAISS index for the embeddings
            faiss_index = faiss.IndexFlatL2(embedding_array.shape[1])
            faiss_index.add(embedding_array)  # Add all embeddings to the FAISS index

            # Prepare input for LLM
            formatted_input = (
                "Please compare the following two documents and highlight only the significant textual differences, "
                "ignoring variations due to whitespace or formatting.\n\n"
                f"Document 1: {documents[0].metadata['name']}\n{documents[0].page_content}\n\n"
                f"Document 2: {documents[1].metadata['name']}\n{documents[1].page_content}"
            )

            # Get the LLM response
            try:
                llm_response = model(formatted_input)
                st.subheader(f"Differences between {documents[0].metadata['name']} and {documents[1].metadata['name']}:")
                st.write(llm_response)

            except Exception as e:
                st.error(f"Error processing LLM: {str(e)}")

        st.success("Comparison completed!")
