import streamlit as st
import difflib
import pdfplumber
import pandas as pd
from langchain_groq import ChatGroq
import re
import nltk
from nltk.tokenize import sent_tokenize

# Function to read PDF text
def read_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text
    
# Function to clean and normalize each sentence, removing bullet points and extra spaces
def preprocess_sentence(sentence):
    # Remove common bullet points or symbols at the beginning of each sentence
    sentence = re.sub(r'^[\s•*\-]+', '', sentence)  # Ensure '-' is correctly escaped or positioned
    # Normalize whitespace within the sentence
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    return sentence

# Function to preprocess each document and return a list of normalized sentences
def normalize_sentences(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    return [preprocess_sentence(sentence) for sentence in sentences]

# Function to find differences and format them in a tabular format, focusing on meaningful content changes
def find_differences_table(text1, text2):
    # Normalize each sentence of both texts
    normalized_sentences1 = normalize_sentences(text1)
    normalized_sentences2 = normalize_sentences(text2)
    
    # Use unified diff to capture only content additions/deletions
    diff = difflib.unified_diff(
        normalized_sentences1,
        normalized_sentences2,
        lineterm=''
    )

    differences = []
    for line in diff:
        # Capture only meaningful content additions or deletions, ignoring structural markers
        if line.startswith('+') and not line.startswith('+++'):
            differences.append({"Document": "Document 2", "Change Type": "Addition", "Text": line[1:].strip()})
        elif line.startswith('-') and not line.startswith('---'):
            differences.append({"Document": "Document 1", "Change Type": "Deletion", "Text": line[1:].strip()})

    return pd.DataFrame(differences)


# Function to summarize differences using an LLM
def summarize_differences(diff_text):
    model = ChatGroq(groq_api_key="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", model_name="Llama3-8b-8192")
    response = model.generate(diff_text)
    return response

# Streamlit app
st.title("Document Comparison Bot")

# Upload PDF document files
uploaded_file1 = st.file_uploader("Upload Document 1 (PDF only)", type=["pdf"])
uploaded_file2 = st.file_uploader("Upload Document 2 (PDF only)", type=["pdf"])

if uploaded_file1 and uploaded_file2:
    # Load the documents
    doc1_text = read_pdf(uploaded_file1)
    doc2_text = read_pdf(uploaded_file2)

    st.subheader("Document 1")
    st.text_area("Document 1 Text", value=doc1_text, height=300)

    st.subheader("Document 2")
    st.text_area("Document 2 Text", value=doc2_text, height=300)

    # Show differences in a table format
    diff_table = find_differences_table(doc1_text, doc2_text)
    st.subheader("Differences")
    st.write(diff_table)

    # Summarize the differences using LLM
    if st.button("Summarize Differences"):
        diff_text = '\n'.join(diff_table["Text"].tolist())
        summary = summarize_differences(diff_text)
        st.subheader("Summary of Differences")
        st.text(summary)
