import pinecone
import pinecone.grpc

from pinecone.grpc import PineconeGRPC as Pinecone

from abc import ABC, abstractmethod
from typing import Iterator

from src import get_config

class VectorDatabaseInterface(ABC):

    @abstractmethod
    def upsert_data() -> bool: 
        pass

    @abstractmethod
    def query_data() -> bool: 
        pass

    @abstractmethod
    def fetch_data() -> bool: 
        pass

    @abstractmethod
    def delete_data() -> bool:
        pass


class PineconeVectorDatabase(VectorDatabaseInterface): 

    pc: Pinecone
    index: pinecone.grpc.GRPCIndex
    index_data: dict

    def __init__(self):
        super().__init__()
        get_config.Settings.load()

        self.pc = Pinecone(api_key=get_config.Settings.pinecone_api_key)

        indexes_list: pinecone.IndexList = self.pc.list_indexes()
        self.index_data = indexes_list[0]
        
        self.index = self.pc.Index(host=self.index_data["host"])


    def get_index_description(self) -> pinecone.IndexModel: 
        return self.pc.describe_index(self.index_data["name"])

    def upsert_data(self, records: dict | list[dict], batch_size: int) -> bool: 
        """ Record format is: [
            ("id1", embedding_vector1, {"metadata_key": "value"}),
            ("id2", embedding_vector2, {"metadata_key": "value"}),
            ]
        """
        # max 2 MB size in a batch

        upsert_batch: list[dict] = []

        if type(records) == dict: upsert_batch.append(records)

        low_index: int = 0
        high_index: int = batch_size
        
        # Transfer in chunks
        while high_index < len(records):
            try:
                responce = self.index.upsert(namespace="default-namespace", vectors=upsert_batch[low_index:high_index])
                assert responce["upsertedCount"] == batch_size
            except AssertionError: print("Upsert did not succeed"); return

            low_index += batch_size
            high_index += batch_size

        # Transfer reminder
        try:
            self.index.upsert(namespace="default-namespace", vectors=upsert_batch[low_index:])
            assert responce["upsertedCount"] == batch_size
        except AssertionError: print("Upsert did not succeed"); return
        


    def query_data(self) -> bool: 
        pass

    def fetch_data(self) -> bool: 
        pass

    def delete_data(self) -> bool:
        pass


if __name__ == "__main__": 
    database: PineconeVectorDatabase = PineconeVectorDatabase()
    print(database.get_index_description())
    

# pinecone.init(api_key="YOUR_API_KEY", environment="YOUR_ENVIRONMENT")  # Replace with your API key and environment
# index_name = "your-index-name"

# if index_name not in pinecone.list_indexes():
#     pinecone.create_index(index_name, dimension=384)  # From embedding model
# index = pinecone.Index(index_name)


# # Load the CSV file
# csv_file = "/path/to/your/csv_file.csv"
# df = pd.read_csv(csv_file)

# # Initialize a pre-trained model for generating embeddings
# model = SentenceTransformer('all-MiniLM-L6-v2')  # Replace with your preferred model

# # Prepare data for Pinecone
# vectors = []
# for i, row in df.iterrows():
#     # Generate a dense vector for the text (e.g., a review column)
#     vector = model.encode(row['review_text'])  # Replace 'review_text' with the appropriate column name
#     # Create a unique ID for the vector
#     vector_id = f"row-{i}"
#     # Append the vector and metadata
#     vectors.append((vector_id, vector, {"metadata_key": row['metadata_column']}))  # Replace metadata_key/column as needed

# # Upsert vectors into Pinecone
# index.upsert(vectors)

