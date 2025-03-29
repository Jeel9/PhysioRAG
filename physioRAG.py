import streamlit as st
import os
import io
import PyPDF2
import fitz  # PyMuPDF
import pytesseract
import pickle
from PIL import Image
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
st.title("Test Dynamic - Test Scenario Generator")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""
)

VECTOR_STORE_PATH = "./faiss_index"
VECTOR_STORE_FILE = "./faiss_index.pkl"

def vector_embedding():
    if os.path.exists(VECTOR_STORE_FILE):
        with open(VECTOR_STORE_FILE, "rb") as f:
            st.session_state.vectors = pickle.load(f)
        st.success("Loaded existing vector store from file.")
        return

    st.session_state.embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    pdf_files_directory = "./java_files"
    pdf_documents = []

    def extract_text_from_pdf(pdf_path):
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            content = ""
            n = len(reader.pages)
            for page_num in range(n):
                if page_num%25 == 0:
                    print("Pages = ",page_num,"/",n)
                page = reader.pages[page_num]
                text = page.extract_text()
                if text and text.strip():
                    content += text
                else:
                    ocr_text = extract_text_with_ocr(pdf_path, page_num)
                    content += ocr_text
        return content

    def extract_text_with_ocr(pdf_path, page_num):
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        images = page.get_images(full=True)
        ocr_text = ""
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            ocr_text += pytesseract.image_to_string(image)
        return ocr_text

    for root, _, files in os.walk(pdf_files_directory):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                extracted_content = extract_text_from_pdf(file_path)
                if extracted_content.strip():
                    pdf_documents.append(Document(page_content=extracted_content, metadata={"source": file_path}))

    if not pdf_documents:
        st.error("No PDF files found in the specified directory.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=400)
    final_documents = text_splitter.split_documents(pdf_documents)

    st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
    with open(VECTOR_STORE_FILE, "wb") as f:
        pickle.dump(st.session_state.vectors, f)
    st.success("Vector Store DB with PDFs is Ready and Saved.")

prompt1 = st.text_input("Enter Your Question From Code")

if st.button("Code Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

import time

if prompt1:
    start = time.process_time()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': prompt1})
    print("Response time :", time.process_time() - start)
    st.write(response['answer'])
    
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
