from pinecone.core.openapi.db_data.model.fetch_response import FetchResponse
from pinecone.grpc import PineconeGRPC as Pinecone, PineconeGrpcFuture

import pinecone

from abc import ABC, abstractmethod
import pinecone.grpc
from src import get_config

from generate_embeddings import Embedding

class VectorDatabaseInterface(ABC):

    @abstractmethod
    def upsert_data(self) -> bool:
        pass

    @abstractmethod
    def query_data(self, text: str):
        pass

    @abstractmethod
    def fetch_data(self) -> FetchResponse | PineconeGrpcFuture:
        pass

    @abstractmethod
    def delete_data(self) -> None:
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

    def upsert_data(self) -> bool:
        pass

    def query_data(self, text: str):
        return self.index.query(
            vector=Embedding.create_embedding(text).tolist(),
            top_k=3,
            include_metadata=True,
            include_values=False
        )

    def fetch_data(self, ids: str | list[str]) -> FetchResponse | PineconeGrpcFuture:
        return self.pc.fetch_index([ids]) if type(ids) == str else self.pc.fetch_index(ids)

    def delete_data(self, ids: str | list[str]) -> None:
        self.pc.delete_index(ids) if type(ids) == str else (self.pc.delete_index(idx) for idx in ids)


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

