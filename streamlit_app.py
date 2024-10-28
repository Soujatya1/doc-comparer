import streamlit as st
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
import os
import time
import requests
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import pandas as pd
import io
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from difflib import SequenceMatcher
from langchain.document_loaders import PyPDFLoader
from docx import Document

st.title("Document Comparer")
st.subheader("Compare your Documents")

if not os.path.exists("uploaded_files"):
    os.makedirs("uploaded_files")


if 'history' not in st.session_state:
    st.session_state.history = []

if 'last_context' not in st.session_state:
    st.session_state.last_context = ""

uploaded_files = st.file_uploader("Upload a file", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save each file temporarily in the created directory
        file_path = os.path.join("uploaded_files", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("uploaded_files")
        st.session_state.docs = st.session_state.loader.load()
        st.write(f"Loaded {len(st.session_state.docs)} documents.")

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

topic_extraction_prompt = PromptTemplate(
    input_variables=["document"],
    template="Identify and list the main topics or sections in the following document:\n\n{document}\n\nReturn each topic as a separate line."
)

comparison_prompt = PromptTemplate(
    input_variables=["topic", "doc_a_content", "doc_b_content"],
    template="For the topic '{topic}', compare the content from two documents. "
             "List any differences between Document A and Document B, focusing on meaning and important details. "
             "If no differences, state 'No significant difference'.\n\n"
             "Document A:\n{doc_a_content}\n\nDocument B:\n{doc_b_content}\n\nDifferences:"
)
def read_file_with_fallback(uploaded_file):
    try:
        # Try reading with UTF-8 encoding first
        content = uploaded_file.read().decode("utf-8")
    except UnicodeDecodeError:
        # Fallback to ISO-8859-1 encoding
        content = uploaded_file.read().decode("ISO-8859-1")
    return content

def extract_topics(llm_chain, document_text):
    response = llm_chain.run({"document": document_text})
    return response.splitlines()

def compare_topics(llm_chain, topics, doc_a, doc_b):
    comparisons = []
    for topic in topics:
        response = llm_chain.run({
            "topic": topic,
            "doc_a_content": doc_a,
            "doc_b_content": doc_b
        })
        
        if "No significant difference" not in response:
            comparisons.append({
                "Topic": topic,
                "Differences": response.strip()
            })
    return comparisons

# Display the comparison results in a table
def display_comparisons(comparisons):
    # Prepare data for tabular format
    data = {
        "Comparison ID": [f"{i + 1}" for i in range(len(comparisons))],
        "Topic": [comp['Topic'] for comp in comparisons],
        "Differences": [comp["Differences"] for comp in comparisons]
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
        label="Download Comparison Results as Excel",
        data=excel_buffer,
        file_name='comparison_results.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

llm = ChatGroq(groq_api_key="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", model_name="Llama3-8b-8192")

uploaded_file_a = st.file_uploader("Upload Document A", type=["txt", "pdf"])
uploaded_file_b = st.file_uploader("Upload Document B", type=["txt", "pdf"])

if uploaded_file_a and uploaded_file_b:
    # Load document content
    doc_a = read_file_with_fallback(uploaded_file_a)
    doc_b = read_file_with_fallback(uploaded_file_b)
    
    # Initialize LLM chains
    topic_llm_chain = LLMChain(prompt=topic_extraction_prompt, llm=llm)
    comparison_llm_chain = LLMChain(prompt=comparison_prompt, llm=llm)

    # Extract topics from Document A as a reference
    topics = extract_topics(topic_llm_chain, doc_a)
    st.write("Extracted Topics:", topics)

    # Compare topics between Document A and Document B
    if st.button("Compare Documents by Topic"):
        start = time.process_time()
        comparisons = compare_topics(comparison_llm_chain, topics, doc_a, doc_b)
        st.write("Response time:", time.process_time() - start)

        # Display the distinct comparisons
        display_comparisons(comparisons)
        
def compare_documents(documents):
    comparisons = []
    
    for i, doc_a in enumerate(documents):
        for j, doc_b in enumerate(documents[i + 1:], start=i + 1):
            # Extract text content from Document objects
            text_a = doc_a.page_content  # Adjust based on your Document structure
            text_b = doc_b.page_content

            # Split document content into sentences or phrases
            phrases_a = text_a.split('. ')
            phrases_b = text_b.split('. ')

            # Compare phrases between Document A and Document B
            for phrase_a in phrases_a:
                # Find most similar phrase in Document B
                highest_similarity = 0
                most_similar_b = None

                for phrase_b in phrases_b:
                    similarity = SequenceMatcher(None, phrase_a, phrase_b).ratio()
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        most_similar_b = phrase_b

                # If phrases are different, find specific word-level differences
                if highest_similarity < 0.8:  # Adjust threshold if needed
                    words_a = phrase_a.split()
                    words_b = most_similar_b.split() if most_similar_b else []

                    # Compare each word in the phrase for fine-grained differences
                    for idx, word_a in enumerate(words_a):
                        word_b = words_b[idx] if idx < len(words_b) else "[No matching word]"
                        if word_a != word_b:
                            comparisons.append({
                                "Document A": f"Document {i + 1}",
                                "Document B": f"Document {j + 1}",
                                "Text in Document A": word_a,
                                "Text in Document B": word_b
                            })

    return comparisons

def create_prompt(input_text):
    previous_interactions = "\n".join(
        [f"You: {h['question']}\nBot: {h['answer']}" for h in st.session_state.history[-5:]]
    )
    return ChatPromptTemplate.from_template(
        f"""
        Compare the uploaded documents based on their differences and form a good context on the same.
        Previous Context: {st.session_state.last_context}
        Previous Interactions:\n{previous_interactions}
        <context>
        {{context}}
        <context>
        Questions: {input_text}
        """
    )
    
def display_comparisons(comparisons):
    # Prepare data for tabular format
    data = {
        "Comparison ID": [f"{i + 1}" for i in range(len(comparisons))],
        "Document A": [comp['Document A'] for comp in comparisons],
        "Document B": [comp['Document B'] for comp in comparisons],
        "Text in Document A": [comp["Text in Document A"] for comp in comparisons],
        "Text in Document B": [comp["Text in Document B"] for comp in comparisons]
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
        label="Download Comparison Results as Excel",
        data=excel_buffer,
        file_name='comparison_results.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

# Main Streamlit app code
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

            # Display the distinct comparisons
            display_comparisons(comparisons)
