import streamlit as st
import os
import PyPDF2
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()

## load the GROQ And OpenAI API KEY 
groq_api_key=os.getenv('GROQ_API_KEY')

st.title("Test Dynamic - Test Scenario Generator")

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

def vector_embedding():
    if "vectors" not in st.session_state:

        # Initialize embeddings
        st.session_state.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",      #sentence-transformers/all-MiniLM-l6-v2
            model_kwargs={'device':'cpu'},
            encode_kwargs={'normalize_embeddings':True}
        )
        

        # ------------------------ Reading Java files for context -----------------------

        # java_files_directory = "./java_files"  # Update to point to the 'src' folder
        # java_documents = []

        # # Read all Java files from the 'src' directory and its subdirectories
        # for root, _, files in os.walk(java_files_directory):
        #     for file in files:
        #         if file.endswith(".java"):
        #             file_path = os.path.join(root, file)
        #             with open(file_path, "r", encoding="utf-8") as f:
        #                 content = f.read()
        #                 java_documents.append(Document(page_content=content, metadata={"source": file_path}))



        # ------------------------ Reading PDF files for context -----------------------

        # Define the directory containing PDF files
        pdf_files_directory = "./pdf_files"  # Update to point to your target folder
        pdf_documents = []

        # Recursively traverse the directory to find all PDF files
        for root, _, files in os.walk(pdf_files_directory):
            for file in files:
                if file.endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        content = ""
                        for page in range(len(reader.pages)):
                            content += reader.pages[page].extract_text()
                        pdf_documents.append({"page_content": content, "metadata": {"source": file_path}})

        #converting name from pdf_docs -> java_docs
        java_documents = pdf_documents

        # Check if any Java files were found
        if not java_documents:
            st.error("No Java files found in the specified directory.")
            return

        # Text splitting to handle large files
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(java_documents)

        # Generate vector embeddings and store them
        st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
        st.success("Vector Store DB with Java Code is Ready")



prompt1=st.text_input("Enter Your Question From Code")


if st.button("Code Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

import time



if prompt1:
    start=time.process_time()
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
