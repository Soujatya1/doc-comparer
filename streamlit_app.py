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
    line = re.sub(r'^[\s•*\-]+', '', line)
    # Normalize whitespace within the line and convert to lowercase for uniformity
    line = re.sub(r'\s+', ' ', line).strip().lower()  # Normalize spaces and make lowercase
    return line

# Function to preprocess each document and return a list of normalized lines
def normalize_lines(text):
    return [preprocess_line(line) for line in text.splitlines()]

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
            # Check if the addition is not just whitespace or line change
            if line[1:].strip():  # Only add if there is meaningful text
                differences.append({"Document": "Document 2", "Change Type": "Addition", "Text": line[1:].strip()})
        elif line.startswith('-') and not line.startswith('---'):
            # Check if the deletion is not just whitespace or line change
            if line[1:].strip():  # Only add if there is meaningful text
                differences.append({"Document": "Document 1", "Change Type": "Deletion", "Text": line[1:].strip()})

    return pd.DataFrame(differences)

# Function to summarize differences using an LLM
def summarize_differences(diff_text):
    model = ChatGroq(groq_api_key="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", model_name="Llama3-8b-8192")
    response = model.generate(diff_text)
    return response

# Streamlit app
st.title("Document Comparer!")
st.subheader("Find the Differences...")

# Upload PDF document files
uploaded_file1 = st.file_uploader("Upload Document 1 (PDF only)", type=["pdf"])
uploaded_file2 = st.file_uploader("Upload Document 2 (PDF only)", type=["pdf"])

# Compare Documents button
if st.button("Compare Documents") and uploaded_file1 and uploaded_file2:
    # Load the documents
    doc1_text = read_pdf(uploaded_file1)
    doc2_text = read_pdf(uploaded_file2)

    # Show differences in a table format
    diff_table = find_differences_table(doc1_text, doc2_text)
    st.subheader("Differences")
    st.write(diff_table)
