import chromadb

chroma_client = chromadb.PersistentClient(
    path="./../../chroma_db")
collection = chroma_client.get_or_create_collection(name="pdf_collection")
def store_in_chroma(text_chunks):
    for i, chunk in enumerate(text_chunks):
        # embedding = embedding_model.encode(chunk).tolist()
        collection.add(
            ids=[f"doc_{i}"],
            documents=[chunk]
        )