import streamlit as st
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import os
import time
import pandas as pd
import io
from difflib import SequenceMatcher

st.title("Document Comparer")
st.subheader("Compare your Documents")

if not os.path.exists("uploaded_files"):
    os.makedirs("uploaded_files")

if 'history' not in st.session_state:
    st.session_state.history = []

if 'last_context' not in st.session_state:
    st.session_state.last_context = ""

uploaded_files = st.file_uploader("Upload a file", type=["pdf"], accept_multiple_files=True)

# Load and process documents on initial upload
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save each file temporarily in the created directory
        file_path = os.path.join("uploaded_files", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    if "vectors" not in st.session_state:
        # Load documents once
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("uploaded_files")
        st.session_state.docs = st.session_state.loader.load()
        st.write(f"Loaded {len(st.session_state.docs)} documents.")

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

llm = ChatGroq(groq_api_key="your_groq_api_key", model_name="Llama3-70b-8192")

def compare_documents(documents):
    comparisons = []

    for i, doc_a in enumerate(documents):
        for j, doc_b in enumerate(documents[i + 1:], start=i + 1):
            # Extract text content from Document objects
            text_a = doc_a.page_content
            text_b = doc_b.page_content

            # Split document content into sentences
            sentences_a = text_a.split('. ')
            sentences_b = text_b.split('. ')

            for sentence_a in sentences_a:
                most_similar_b = "[No similar text in Document B]"
                highest_similarity = 0

                for sentence_b in sentences_b:
                    similarity = SequenceMatcher(None, sentence_a, sentence_b).ratio()
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        most_similar_b = sentence_b

                # Record as a difference if below threshold (e.g., 0.8)
                if highest_similarity < 0.8:
                    comparisons.append({
                        "Document A": f"Document {i + 1}",
                        "Document B": f"Document {j + 1}",
                        "Text in Document A": sentence_a,
                        "Text in Document B": most_similar_b,
                        "Similarity Score": round(highest_similarity, 2)
                    })

    return comparisons

def display_comparisons(comparisons):
    # Prepare data for tabular format
    data = {
        "Comparison ID": [f"{i + 1}" for i in range(len(comparisons))],
        "Document A": [comp['Document A'] for comp in comparisons],
        "Document B": [comp['Document B'] for comp in comparisons],
        "Text in Document A": [comp["Text in Document A"] for comp in comparisons],
        "Text in Document B": [comp["Text in Document B"] for comp in comparisons],
        "Similarity Score": [comp["Similarity Score"] for comp in comparisons]
    }

    # Create DataFrame and display table
    comparison_df = pd.DataFrame(data)
    st.table(comparison_df)

    # Convert the DataFrame to an Excel file
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        comparison_df.to_excel(writer, index=False, sheet_name='Comparison Results')
    excel_buffer.seek(0)

    # Download button for the Excel file
    st.download_button(
        label="Download Comparison Results",
        data=excel_buffer,
        file_name='comparison_results.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

if "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, create_prompt("document comparison"))
    retriever = st.session_state.vectors.as_retriever(search_type="similarity", k=2)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    if st.button("Compare Documents"):
        start = time.process_time()
        response = retrieval_chain.invoke({'input': "document comparison"})
        st.write("Response time:", time.process_time() - start)

        if response.get("context"):
            # Assuming response["context"] is a list of Document objects
            comparisons = compare_documents(response["context"])

            # Display the comparisons in the table
            display_comparisons(comparisons)
