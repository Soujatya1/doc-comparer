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

def preprocess_line(line):
    # Remove common bullet points or symbols at the beginning of each line
    line = re.sub(r'^[\s•*\-]+', '', line)  # Remove leading bullet points
    # Normalize whitespace within the line but keep significant formatting intact
    return re.sub(r'\s+', ' ', line).strip()

# Function to normalize each document and return a list of normalized lines
def normalize_lines(text):
    lines = text.splitlines()
    normalized_lines = [preprocess_line(line) for line in lines]
    return [line for line in normalized_lines if line]  # Remove empty lines

# Function to find differences and format them in a tabular format, focusing on meaningful content changes
def find_differences_table(text1, text2):
    # Normalize each line of both texts
    normalized_text1 = normalize_lines(text1)
    normalized_text2 = normalize_lines(text2)

    # Use unified diff to capture only content additions/deletions
    diff = difflib.unified_diff(
        normalized_text1,
        normalized_text2,
        lineterm=''
    )

    differences = []
    for line in diff:
        # Capture only meaningful content additions or deletions, ignoring structural markers
        if line.startswith('+') and not line.startswith('+++'):
            changed_part = line[1:].strip()
            # Ensure we only capture significant changes
            if changed_part and not changed_part in normalized_text1:
                differences.append({"Document": "Document 2", "Change Type": "Addition", "Text": changed_part})
        elif line.startswith('-') and not line.startswith('---'):
            changed_part = line[1:].strip()
            # Ensure we only capture significant changes
            if changed_part and not changed_part in normalized_text2:
                differences.append({"Document": "Document 1", "Change Type": "Deletion", "Text": changed_part})

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
