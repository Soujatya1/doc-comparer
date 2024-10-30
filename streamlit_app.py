

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

st.title("Document Comparer!")
st.subheader("Compare Your Documents")

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

# Ensure directories for both upload slots
os.makedirs("uploaded_files_1", exist_ok=True)
os.makedirs("uploaded_files_2", exist_ok=True)

# Session state for history and context
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_context' not in st.session_state:
    st.session_state.last_context = ""
    
llm = ChatGroq(groq_api_key="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", model_name="Llama3-70b-8192")
# First uploader for Document Set 1
uploaded_files_1 = st.file_uploader("Upload files for Document Set 1", type=["pdf"], accept_multiple_files=True, key="uploader_1")
if uploaded_files_1:
    for uploaded_file in uploaded_files_1:
        file_path = os.path.join("uploaded_files_1", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' uploaded to Document Set 1.")

    if "vectors_1" not in st.session_state:
        st.session_state.embeddings_1 = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.loader_1 = PyPDFDirectoryLoader("uploaded_files_1")
        st.session_state.docs_1 = st.session_state.loader_1.load()
        st.write(f"Loaded {len(st.session_state.docs_1)} documents for Set 1.")

        st.session_state.text_splitter_1 = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        st.session_state.final_documents_1 = st.session_state.text_splitter_1.split_documents(st.session_state.docs_1)
        st.session_state.vectors_1 = FAISS.from_documents(st.session_state.final_documents_1, st.session_state.embeddings_1)

# Second uploader for Document Set 2
uploaded_files_2 = st.file_uploader("Upload files for Document Set 2", type=["pdf"], accept_multiple_files=True, key="uploader_2")
if uploaded_files_2:
    for uploaded_file in uploaded_files_2:
        file_path = os.path.join("uploaded_files_2", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' uploaded to Document Set 2.")

    if "vectors_2" not in st.session_state:
        st.session_state.embeddings_2 = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.loader_2 = PyPDFDirectoryLoader("uploaded_files_2")
        st.session_state.docs_2 = st.session_state.loader_2.load()
        st.write(f"Loaded {len(st.session_state.docs_2)} documents for Set 2.")

        st.session_state.text_splitter_2 = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        st.session_state.final_documents_2 = st.session_state.text_splitter_2.split_documents(st.session_state.docs_2)
        st.session_state.vectors_2 = FAISS.from_documents(st.session_state.final_documents_2, st.session_state.embeddings_2)

def create_comparison_prompt(input_text):
    # Update the prompt to expect a single 'context' variable
    prompt_template = (
        "Compare the following document sets based on their content:\n\n"
        "{context}\n\n"
        "Comparison based on the input question: {input_text}"
    )
    return PromptTemplate(input_variables=["context", "input_text"], template=prompt_template)

def generate_comparison(input_text, context1, context2):
    # Combine both contexts into a single variable
    combined_context = f"Document Set 1: {context1}\n\nDocument Set 2: {context2}"
    
    # Create prompt and chain with the combined context
    comparison_prompt = create_comparison_prompt(input_text)
    comparison_chain = create_stuff_documents_chain(llm, comparison_prompt)
    
    # Generate the comparison response using the combined context
    comparison_response = comparison_chain.invoke({"context": combined_context, "input_text": input_text})
    return comparison_response['answer']


# Language selection (reuse existing code for language mapping and detection)
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

# Input box for user query
input_box = st.empty()
with input_box.container():
    prompt1 = st.text_input("Enter your question here...", key="user_input", placeholder="Type your question...")

if prompt1 and "vectors_1" in st.session_state and "vectors_2" in st.session_state:
    retriever_1 = st.session_state.vectors_1.as_retriever(search_type="similarity", k=3)
    retriever_2 = st.session_state.vectors_2.as_retriever(search_type="similarity", k=3)
    
    response_1_docs = retriever_1.get_relevant_documents(prompt1)
    response_2_docs = retriever_2.get_relevant_documents(prompt1)
    
    # Extract content from documents
    context1 = " ".join([doc.page_content for doc in response_1_docs])
    context2 = " ".join([doc.page_content for doc in response_2_docs])
    
    # Generate comparison using combined context
    comparison_result = generate_comparison(prompt1, context1, context2)
    
    # Display comparison results
    st.write("### Comparison Results")
    st.write(comparison_result)
