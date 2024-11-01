import streamlit as st
import pdfplumber
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.schema import Document  # Import Document class

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

    # Split documents into chunks for better processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    split_documents = text_splitter.split_documents(documents)  # Split directly on the Document list
    
    # Create a FAISS vector store
    vector_store = FAISS.from_documents(split_documents, embedding_model)

    # Set up the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    )

    # Comparing the documents
    if st.button("Compare Documents"):
        comparisons = []
        for doc in documents:
            response = qa_chain.run(doc.page_content)  # Use the page_content attribute
            comparisons.append(response)

        # Display the comparisons
        st.subheader("Comparative Results")
        for i, comparison in enumerate(comparisons):
            st.write(f"**Comparison for {documents[i].metadata['name']}:**")
            st.write(comparison)

        st.success("Comparison completed!")
