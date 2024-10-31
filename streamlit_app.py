import streamlit as st
import difflib
import pdfplumber
import pandas as pd
from langchain_groq import ChatGroq
import re
import numpy as np

# Function to read PDF text
def read_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def preprocess_paragraph(paragraph):
    # Normalize whitespace and remove any leading/trailing whitespace
    return re.sub(r'\s+', ' ', paragraph).strip()

# Function to preprocess each document and return a list of normalized paragraphs
def normalize_paragraphs(text):
    return [preprocess_paragraph(p) for p in text.split('\n\n')]  # Split by double newline for paragraphs

# Function to find differences and format them in a tabular format, focusing on meaningful content changes
def find_differences_table(paragraphs1, paragraphs2):
    differences = []
    
    # Compare paragraphs from both documents
    for i, para1 in enumerate(paragraphs1):
        if i < len(paragraphs2):
            para2 = paragraphs2[i]
            if para1 != para2:
                differences.append({
                    "Document": "Document 1",
                    "Change Type": "Change",
                    "Text": f"Old: {para1}\nNew: {para2}"
                })
        else:
            differences.append({
                "Document": "Document 1",
                "Change Type": "Deletion",
                "Text": para1
            })
    
    # Check for any additional paragraphs in Document 2
    for j in range(len(paragraphs1), len(paragraphs2)):
        differences.append({
            "Document": "Document 2",
            "Change Type": "Addition",
            "Text": paragraphs2[j]
        })

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

    # Normalize paragraphs
    paragraphs1 = normalize_paragraphs(doc1_text)
    paragraphs2 = normalize_paragraphs(doc2_text)

    # Show differences in a table format
    diff_table = find_differences_table(paragraphs1, paragraphs2)
    st.subheader("Differences")
    st.write(diff_table)

    # Summarize the differences using LLM
    if st.button("Summarize Differences"):
        diff_text = '\n'.join(diff_table["Text"].tolist())
        summary = summarize_differences(diff_text)
        st.subheader("Summary of Differences")
        st.text(summary)
