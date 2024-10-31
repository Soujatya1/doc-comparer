import streamlit as st
import difflib
import pdfplumber
import pandas as pd
from langchain_groq import ChatGroq
import re

# Initialize ChatGroq model
model = ChatGroq(groq_api_key="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", model_name="Llama3-8b-8192")

st.write(dir(model))

# Function to read PDF text
def read_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Function to normalize text by splitting into words and removing unwanted characters
def normalize_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()  # Split by whitespace
    return [word.lower() for word in words]  # Normalize to lowercase

# Function to find differences and format them in a tabular format
def find_differences_table(text1, text2):
    normalized_text1 = normalize_text(text1)
    normalized_text2 = normalize_text(text2)

    diff = difflib.unified_diff(
        normalized_text1,
        normalized_text2,
        lineterm=''
    )

    differences = []
    for line in diff:
        if line.startswith('+') and not line.startswith('+++'):
            changed_part = line[1:].strip()
            if changed_part:
                differences.append({"Document": "Document 2", "Change Type": "Addition", "Text": changed_part})
        elif line.startswith('-') and not line.startswith('---'):
            changed_part = line[1:].strip()
            if changed_part:
                differences.append({"Document": "Document 1", "Change Type": "Deletion", "Text": changed_part})

    return pd.DataFrame(differences)

# Function to analyze differences using the ChatGroq model
def analyze_differences(doc1_text, doc2_text):
    prompt = (
        f"Analyze the following documents and highlight the key differences:\n\n"
        f"Document 1:\n{doc1_text}\n\n"
        f"Document 2:\n{doc2_text}\n\n"
        "Please summarize the differences in a concise manner."
    )
    
    # Directly use the prompt string for the model
    response = model.invoke(prompt)  # Use chat() or generate() as per library expectations

    # Extract the generated text from the response
    return response.get("generation", "No response generated.")

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

    # Get analysis from ChatGroq model
    analysis = analyze_differences(doc1_text, doc2_text)
    st.subheader("LLM Analysis of Differences")
    st.write(analysis)
