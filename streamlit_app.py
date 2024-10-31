import streamlit as st
import difflib
import pdfplumber
import pandas as pd
from langchain_groq import ChatGroq
import re

# Function to read PDF text
def read_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Function to normalize text by splitting into words and removing unwanted characters
def normalize_text(text):
    # Remove unwanted characters and normalize spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()  # Split by whitespace
    return [word.lower() for word in words]  # Normalize to lowercase

# Function to find differences and format them in a tabular format, focusing solely on text additions and deletions
def find_differences_table(text1, text2):
    # Normalize the entire texts into lists of words
    normalized_text1 = normalize_text(text1)
    normalized_text2 = normalize_text(text2)

    # Use ndiff to capture only content additions/deletions
    diff = difflib.ndiff(normalized_text1, normalized_text2)

    differences = []
    current_addition = []
    current_deletion = []

    for line in diff:
        if line.startswith('+ '):  # Addition
            current_addition.append(line[2:])
        elif line.startswith('- '):  # Deletion
            current_deletion.append(line[2:])
        else:  # Neutral line (no change)
            if current_addition:
                differences.append({"Document": "Document 2", "Change Type": "Addition", "Text": ' '.join(current_addition)})
                current_addition = []
            if current_deletion:
                differences.append({"Document": "Document 1", "Change Type": "Deletion", "Text": ' '.join(current_deletion)})
                current_deletion = []

    # Check for any remaining additions or deletions at the end
    if current_addition:
        differences.append({"Document": "Document 2", "Change Type": "Addition", "Text": ' '.join(current_addition)})
    if current_deletion:
        differences.append({"Document": "Document 1", "Change Type": "Deletion", "Text": ' '.join(current_deletion)})

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
