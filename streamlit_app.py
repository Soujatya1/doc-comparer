import streamlit as st
import pdfplumber
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.schema import Document

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
            results = []
            doc_contents = [doc.page_content for doc in documents]
            
            # Use LLM to analyze the differences
            prompt = (
                "Compare the following two documents and highlight only the significant textual differences, "
                "ignoring variations due to whitespace or formatting. "
                "Here are the texts:\n\n"
                f"Document 1:\n{doc_contents[0]}\n\n"
                f"Document 2:\n{doc_contents[1]}"
            )

            # Get the LLM response
            llm_response = model(prompt)
            results.append(f"**Differences between {documents[0].metadata['name']} and {documents[1].metadata['name']}:**")
            results.append(llm_response)

            # Display the results
            st.subheader("Comparative Results")
            for result in results:
                st.write(result)

        st.success("Comparison completed!")
