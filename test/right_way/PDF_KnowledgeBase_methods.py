
from test.right_way.PDF_KnowledgeBase import *

def load_pdfs(self):

    loader = DirectoryLoader(
        self.pdf_source_folder_path
    )
    loaded_pdfs = loader.load()
    return loaded_pdfs


def split_documents(
        self,
        loaded_docs,
        chunk_size: Optional[int] = 500,
        chunk_overlap: Optional[int] = 20,
):
    # instantiate the RecursiveCharacterTextSplitter class
    # by providing the chunk_size and chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Now split the documents into chunks and return
    chunked_docs = splitter.split_documents(loaded_docs)
    return chunked_docs


def convert_document_to_embeddings(
        self, chunked_docs, embedder
):
    # instantiate the Chroma db python client
    # embedder will be our embedding function that will map our chunked
    # documents to embeddings

    vector_db = Chroma(
        persist_directory=CHROMA_DB_DIRECTORY,
        embedding_function=embedder,
        client_settings=CHROMA_SETTINGS,
    )

    # now once instantiated we tell our db to inject the chunks
    # and save all inside the db directory
    vector_db.add_documents(chunked_docs)
    vector_db.persist()

    # finally return the vector db client object
    return vector_db


def return_retriever_from_persistant_vector_db(
        self, embedder
):
    # first check whether the database is created or not
    # if not then throw error
    # because if the database is not instantiated then
    # we can not get the retriever

    if not os.path.isdir(CHROMA_DB_DIRECTORY):
        raise NotADirectoryError(
            "Please load your vector database first."
        )

    vector_db = Chroma(
        persist_directory=CHROMA_DB_DIRECTORY,
        embedding_function=embedder,
        client_settings=CHROMA_SETTINGS,
    )

    # used the returned embedding function to provide the retriver object
    # with number of relevant chunks to return will be = 4
    # based on the one we set inside our settings

    return vector_db.as_retriever(
        search_kwargs={"k": TARGET_SOURCE_CHUNKS}
    )

PDFKnowledgeBase.load_pdfs = load_pdfs
