import streamlit as st
import difflib
import pdfplumber
import pandas as pd

# Function to read PDF text
def read_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()  # Remove trailing whitespace

# Function to find differences and format them in a tabular format
def find_differences_table(text1, text2):
    # Split the texts into lines for comparison
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()

    # Use unified diff to capture content differences
    diff = difflib.unified_diff(
        lines1,
        lines2,
        lineterm='',
        fromfile='Document 1',
        tofile='Document 2'
    )

    differences = []
    addition_found = False  # Flag to track if we've found an addition to Document 1
    deletion_found = False  # Flag to track if we've found a deletion from Document 2

    for line in diff:
        # Capture only meaningful content additions or deletions
        if line.startswith('+') and not line.startswith('+++'):
            changed_part = line[1:].strip()
            if changed_part and not addition_found:  # Only consider non-empty additions
                differences.append({"Document": "Document 2", "Change Type": "Addition", "Text": changed_part})
                addition_found = True  # Set flag to True after finding the first addition
        elif line.startswith('-') and not line.startswith('---'):
            changed_part = line[1:].strip()
            if changed_part and not deletion_found:  # Only consider non-empty deletions
                differences.append({"Document": "Document 1", "Change Type": "Deletion", "Text": changed_part})
                deletion_found = True  # Set flag to True after finding the first deletion

    return pd.DataFrame(differences)

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
