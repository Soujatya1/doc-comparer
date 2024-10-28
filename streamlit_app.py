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

DetectorFactory.seed = 0

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return None

st.markdown("""
    <style>
    .input-box {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: white;
        padding: 10px;
        box-shadow: 0 -1px 5px rgba(0, 0, 0, 0.1);
        z-index: 999;
    }
    .conversation-history {
        max-height: 75vh;
        overflow-y: auto;
    }
    </style>
    """, unsafe_allow_html=True)

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

llm = ChatGroq(groq_api_key="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", model_name="Llama3-8b-8192")

def read_docx(file):
    # Read from the BytesIO object directly
    doc = Document(io.BytesIO(file.read()))
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return '\n'.join(text)

# Function to compare documents and find precise differences
def compare_documents(context):
    comparisons = []
    for i, doc_a in enumerate(context):
        for j, doc_b in enumerate(context[i + 1:], start=i + 1):
            text_a = doc_a["page_content"]  # Adjust based on your Document structure
            text_b = doc_b["page_content"]  # Adjust based on your Document structure

            sentences_a = text_a.split('. ')
            sentences_b = text_b.split('. ')

            unique_a = []
            unique_b = []

            for sentence_a in sentences_a:
                highest_similarity = 0
                most_similar_b = None

                for sentence_b in sentences_b:
                    similarity = SequenceMatcher(None, sentence_a, sentence_b).ratio()
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        most_similar_b = sentence_b

                if highest_similarity < 0.8:  # Adjust similarity threshold as needed
                    unique_a.append(sentence_a.strip())
                    if most_similar_b:
                        unique_b.append(most_similar_b.strip())
                    else:
                        unique_b.append("[No matching text in Document B]")

            if unique_a or unique_b:
                comparisons.append({
                    "Document A": f"Document {i + 1}",
                    "Document B": f"Document {j + 1}",
                    "Unique in Document A": " | ".join(unique_a),
                    "Unique in Document B": " | ".join(unique_b)
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

def translate_text(text, source_language, target_language):
    api_url = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline/"
    user_id = "bdeee189dc694351b6b248754a918885"
    ulca_api_key = "099c9c6409-1308-4503-8d33-64cc5e49a07f"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ulca_api_key}",
        "userID": user_id,
        "ulcaApiKey": ulca_api_key
    }

    payload = {
        "pipelineTasks": [
            {
                "taskType": "translation",
                "config": {
                    "language": {
                        "sourceLanguage": source_language,
                        "targetLanguage": target_language
                    }
                }
            }
        ],
        "pipelineRequestConfig": {
            "pipelineId": "64392f96daac500b55c543cd"
        }
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers)
        if response.status_code == 200:
            response_data = response.json()
            service_id = response_data["pipelineResponseConfig"][0]["config"][0]["serviceId"]
        else:
            return text

    except Exception as e:
        return text

    compute_payload = {
        "pipelineTasks": [
            {
                "taskType": "translation",
                "config": {
                    "language": {
                        "sourceLanguage": source_language,
                        "targetLanguage": target_language
                    },
                    "serviceId": service_id
                }
            }
        ],
        "inputData": {
            "input": [
                {
                    "source": text
                }
            ]
        }
    }

    callback_url = response_data["pipelineInferenceAPIEndPoint"]["callbackUrl"]
    headers2 = {
        "Content-Type": "application/json",
        response_data["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["name"]:
            response_data["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["value"]
    }

    try:
        compute_response = requests.post(callback_url, json=compute_payload, headers=headers2)
        if compute_response.status_code == 200:
            compute_response_data = compute_response.json()
            translated_content = compute_response_data["pipelineResponse"][0]["output"][0]["target"]
            return translated_content
        else:
            return ""

    except Exception as e:
        return ""

language_mapping = {
    "Auto-detect": "",
    "English": "en",
    "Kashmiri": "ks",
    "Nepali": "ne",
    "Bengali": "bn",
    "Marathi": "mr",
    "Sindhi": "sd",
    "Telugu": "te",
    "Gujarati": "gu",
    "Gom": "gom",
    "Urdu": "ur",
    "Santali": "sat",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Manipuri": "mni",
    "Tamil": "ta",
    "Hindi": "hi",
    "Punjabi": "pa",
    "Odia": "or",
    "Dogri": "doi",
    "Assamese": "as",
    "Sanskrit": "sa",
    "Bodo": "brx",
    "Maithili": "mai"
}

language_options = list(language_mapping.keys())

st.header("Conversation History")
for interaction in st.session_state.history:
    st.write(f"**You:** {interaction['question']}")
    st.write(f"**Bot:** {interaction['answer']}")
    st.write("---")

st.write("---")

with st.sidebar:
    st.header("Language Selection")
    selected_language = st.selectbox("Select language for translation:", language_options, key="language_selection")
    
def display_comparisons(comparisons):
    data = {
        "Comparison ID": [f"{i + 1}" for i in range(len(comparisons))],
        "Document A": [comp['Document A'] for comp in comparisons],
        "Document B": [comp['Document B'] for comp in comparisons],
        "Unique in Document A": [comp["Unique in Document A"] for comp in comparisons],
        "Unique in Document B": [comp["Unique in Document B"] for comp in comparisons]
    }

    comparison_df = pd.DataFrame(data)
    st.table(comparison_df)

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        comparison_df.to_excel(writer, index=False, sheet_name='Comparison Results')
    excel_buffer.seek(0)

    st.download_button(
        label="Download Comparison Results as Excel",
        data=excel_buffer,
        file_name='comparison_results.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

# Main Streamlit app code
st.title("Document Comparison Tool")

# Upload documents
uploaded_files = st.file_uploader("Upload Document A and Document B", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) != 2:
        st.warning("Please upload exactly two documents for comparison.")
    else:
        doc_name_a = st.text_input("Name for Document A", value="Document A")
        doc_name_b = st.text_input("Name for Document B", value="Document B")

        documents = []
        for uploaded_file in uploaded_files:
            if uploaded_file.type == "application/pdf":
                loader = PyPDFLoader(uploaded_file)
                document = loader.load()
                documents.extend(document)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = read_docx(uploaded_file)  # Read DOCX content
                documents.append({"page_content": text})  # Adjust to your Document structure

        st.session_state.vectors = documents  # Store loaded documents in session state

        if st.button("Compare Documents"):
            document_chain = create_stuff_documents_chain(llm, create_prompt("document comparison"))
            retriever = st.session_state.vectors  # Use the loaded documents directly
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start = time.process_time()
            response = retrieval_chain.invoke({'input': "document comparison"})
            st.write("Response time:", time.process_time() - start)

            if response.get("context"):
                comparisons = compare_documents(response["context"])

                # Filter only distinct document pairs
                distinct_comparisons = [
                    comp for comp in comparisons if comp["Document A"] != comp["Document B"]
                ]

                # Update document names in comparisons
                for comp in distinct_comparisons:
                    comp["Document A"] = doc_name_a
                    comp["Document B"] = doc_name_b

                display_comparisons(distinct_comparisons)
