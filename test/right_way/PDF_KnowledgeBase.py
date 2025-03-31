import os
from typing import Optional
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFKnowledgeBase:
    def __init__(self, pdf_source_folder_path: str) -> None:
        """
        Loads pdf and creates a Knowledge base using the Chroma
        vector DB.
        Args:
            pdf_source_folder_path (str): The source folder containing
            all the pdf documents
        """
        self.pdf_source_folder_path = pdf_source_folder_path

    def load_pdfs(self):
        # method to load all the pdf's inside the directory
        # using DirectoryLoader
        pass

    def split_documents(self, loaded_docs, chunk_size=1000):
        # split the documents into chunks and return the
        # chunked documents
        pass

    def convert_document_to_embeddings(
        self, chunked_docs, embedder
    ):
        # convert the chunked docs to embeddings and add that
        # to our vector db
        pass

    def return_retriever_from_persistant_vector_db(
        self, embedding_function
    ):
        # return a retriever object which will retrieve the
        # relevant chunks
        pass