import streamlit as st
import PyPDF2
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import io

# Function to extract text from PDF using PyPDF2
def extract_text_from_pdf(pdf_file):
    text = ""
    
    # Read the PDF file and extract text
    try:
        pdf_bytes = pdf_file.read()
        pdf_buffer = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_buffer)
        
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
                
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
    
    return text

# Step 1: Load and Split Documents
def load_and_split_document(content):
    text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=500)
    return text_splitter.split_text(content)

# Step 2: Compare Documents Using LLM
def compare_documents(doc1_sections, doc2_sections, llm):
    prompt_template = """
    Compare the following sections from two documents and provide a summary of the differences:
    Document 1: {doc1_section}
    Document 2: {doc2_section}
    Differences:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    
    differences = []
    for sec1, sec2 in zip(doc1_sections, doc2_sections):
        comparison_prompt = prompt.format(doc1_section=sec1, doc2_section=sec2)
        response = llm(comparison_prompt)
        differences.append(response)
    
    return differences

# Streamlit Application
st.title("PDF Document Comparison Tool")
st.write("Upload two PDF documents to compare their content differences.")

# File Upload
doc1 = st.file_uploader("Upload Document 1 (PDF)", type="pdf")
doc2 = st.file_uploader("Upload Document 2 (PDF)", type="pdf")

# Trigger Comparison
if doc1 and doc2:
    # Extract Text from PDF Documents
    doc1_content = extract_text_from_pdf(doc1)
    doc2_content = extract_text_from_pdf(doc2)

    if not doc1_content or not doc2_content:
        st.error("One of the documents is empty or couldn't be read. Please check the files.")
    else:
        # Split Documents into Sections
        doc1_sections = load_and_split_document(doc1_content)
        doc2_sections = load_and_split_document(doc2_content)

        # Initialize LLM
        llm = ChatGroq(groq_api_key="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", model_name="Llama3-70b-8192")

        # Compare Documents
        differences = compare_documents(doc1_sections, doc2_sections, llm)

        # Display Differences
        st.write("### Differences between the Documents:")
        for idx, diff in enumerate(differences):
            st.write(f"**Difference in Section {idx + 1}:**")
            st.write(diff)
else:
    st.write("Please upload both PDF documents to proceed with the comparison.")
