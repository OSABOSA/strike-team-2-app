import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

if __name__ == "__main__":
    pdf = st.file_uploader("Upload PDF", type="pdf")
    if pdf:
        pdf_reader = PdfReader(pdf)
        st.write("Number of pages:", len(pdf_reader.pages))

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        st.write("Text extracted from PDF:")

        text_splitter = RecursiveCharacterTextSplitter()
        chunks = text_splitter.split_text(text)
        st.write(chunks)

        embeddings = HuggingFaceEmbeddings()
        # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        vector_store = Chroma(
            collection_name="pdf_text",
            embedding_function=embeddings,
            persist_directory=
        )

