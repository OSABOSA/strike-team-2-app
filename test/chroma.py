import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings


# Ustawienie lokalizacji bazy danych na dysku
database_path = "./../chroma_db"

# Tworzenie klienta z określoną bazą danych
# chroma_client = chromadb.Client()

client = chromadb.PersistentClient(
    path=database_path,
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

# Użycie `get_or_create_collection`, aby pobrać istniejącą kolekcję lub utworzyć nową
collection = client.get_or_create_collection(name="my_collection")

# Sprawdź, czy kolekcja jest pusta, jeśli tak - dodaj dane
if len(collection.get(ids=["id1", "id2"])["documents"]) == 0:
    collection.upsert(
        documents=[
            "This is a document about pineapple",
            "This is a document about oranges"
        ],
        ids=["id1", "id2"]
    )
else:
    print("Collection already contains documents")

# Wykonaj zapytanie
results = collection.query(
    query_texts=["Pineapple"],
    n_results=2
)

print(results)
