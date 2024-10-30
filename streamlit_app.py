import streamlit as st
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Initialize embeddings and document databases
embeddings = HuggingFaceEmbeddings()
docsearch_1 = Chroma(persist_directory="path/to/docset1", embedding_function=embeddings)
docsearch_2 = Chroma(persist_directory="path/to/docset2", embedding_function=embeddings)

# Function to retrieve documents based on the query
def retrieve_documents(query):
    response_1 = docsearch_1.similarity_search(query)
    response_2 = docsearch_2.similarity_search(query)
    return response_1, response_2

# Function to extract unique or differing content from each document set
def extract_differences(response_1, response_2):
    doc_set_1_texts = [doc.page_content for doc in response_1]
    doc_set_2_texts = [doc.page_content for doc in response_2]

    differences = []
    for idx, (text1, text2) in enumerate(zip(doc_set_1_texts, doc_set_2_texts)):
        if text1 != text2:
            differences.append({
                "Doc Set 1 Content": text1 if text1 else "Not Available",
                "Doc Set 2 Content": text2 if text2 else "Not Available"
            })
    
    return differences

# Streamlit app UI
st.title("Knowledge Management Chatbot")
query = st.text_input("Enter your query:")

if st.button("Retrieve"):
    if query:
        response_1, response_2 = retrieve_documents(query)

        # Display the retrieved content for each document set
        st.write("### Retrieved Content from Document Set 1")
        for doc in response_1:
            st.write(doc.page_content)

        st.write("### Retrieved Content from Document Set 2")
        for doc in response_2:
            st.write(doc.page_content)

        # Display differences in tabular format if they exist
        differences = extract_differences(response_1, response_2)
        if differences:
            st.write("### Differences in Retrieved Content Between Document Sets")
            differences_df = pd.DataFrame(differences)
            st.table(differences_df)
        else:
            st.write("No significant differences found between Document Set 1 and Document Set 2.")
    else:
        st.write("Please enter a query to retrieve documents.")
