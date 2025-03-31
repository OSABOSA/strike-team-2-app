import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import fitz

pdf_path = "C:/Users/0oski/Desktop/Pulpitowe/1. Opisy obiektów dynamicznych, sterowanie_adaptacyjne i wielopoziomowe.pdf"
pdf_document = fitz.open(pdf_path)
documents = []
ids = []

# Ustawienie lokalizacji bazy danych na dysku
database_path = "./chroma_db"

# Tworzenie klienta z określoną bazą danych
# chroma_client = chromadb.Client()

client = chromadb.PersistentClient(
    path=database_path,
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

# Użycie `get_or_create_collection`, aby pobrać istniejącą kolekcję lub utworzyć nową
collection = client.get_or_create_collection(name="pdf_collection")

# Loop through each page and extract text
for page_num in range(len(pdf_document)):
    page = pdf_document.load_page(page_num)
    text = page.get_text()

    # Use page number as ID to keep track of pages
    doc_id = f"{pdf_path}_page_{page_num + 1}"
    documents.append(text)
    ids.append(doc_id)

# Upsert the extracted text to the collection
collection.upsert(
    documents=documents,
    ids=ids
)

# Confirm that the documents were added
print(f"Added {len(documents)} pages to the vector database.")

# Wykonaj zapytanie
results = collection.query(
    query_texts=["Opis modelu transmitancji"],
    n_results=8
)

print(results["distances"])
