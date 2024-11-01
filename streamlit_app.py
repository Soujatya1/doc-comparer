import streamlit as st
import pdfplumber
from langchain_groq import ChatGroq
from langchain.schema import Document

# Initialize ChatGroq model
groq_api_key = "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"
model = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

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
            
            # Create a formatted input for the LLM
            formatted_input = (
                "Please compare the following two documents and highlight only the significant textual differences, "
                "ignoring variations due to whitespace or formatting.\n\n"
                f"Document 1: {documents[0].metadata['name']}\n{doc_contents[0]}\n\n"
                f"Document 2: {documents[1].metadata['name']}\n{doc_contents[1]}"
            )

            try:
                # Get the LLM response
                llm_response = model(formatted_input)
                results.append(f"**Differences between {documents[0].metadata['name']} and {documents[1].metadata['name']}:**")
                results.append(llm_response)

            except Exception as e:
                st.error(f"Error processing LLM: {str(e)}")

            # Display the results
            st.subheader("Comparative Results")
            for result in results:
                st.write(result)

        st.success("Comparison completed!")
