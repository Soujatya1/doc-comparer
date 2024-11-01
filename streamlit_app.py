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
            faiss_index = faiss.IndexFlatL2(embedding_model.output_dim)
            embeddings = []

            # Embed document contents and store them in FAISS
            for doc in documents:
                embedding = embedding_model.embed([doc.page_content])  # Embed document content
                embeddings.append(embedding[0])  # Store the first (and only) embedding
                faiss_index.add(np.array([embedding[0]], dtype=np.float32))  # Add embedding to FAISS

            # Perform a search to compare the embeddings
            results = []
            comparison_results = []

            # Generate input for the LLM to compare documents based on their embeddings
            formatted_input = (
                "Please compare the following two documents and highlight only the significant textual differences, "
                "ignoring variations due to whitespace or formatting.\n\n"
                f"Document 1: {documents[0].metadata['name']}\n{documents[0].page_content}\n\n"
                f"Document 2: {documents[1].metadata['name']}\n{documents[1].page_content}"
            )

            # Get the LLM response
            try:
                llm_response = model(formatted_input)
                comparison_results.append(f"**Differences between {documents[0].metadata['name']} and {documents[1].metadata['name']}:**")
                comparison_results.append(llm_response)

            except Exception as e:
                st.error(f"Error processing LLM: {str(e)}")

            # Display the results
            st.subheader("Comparative Results")
            for result in comparison_results:
                st.write(result)

        st.success("Comparison completed!")
