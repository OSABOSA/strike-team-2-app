import pinecone
import pinecone.grpc

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone.grpc import PineconeGrpcFuture
from pinecone.core.openapi.db_data.model.fetch_response import FetchResponse

from abc import ABC, abstractmethod
from src import get_config

from src.data.generate_embeddings import Embedding


class VectorDatabaseInterface(ABC):

    @abstractmethod
    def upsert_data(self) -> bool:
        pass

    @abstractmethod
    def query_data(self, text: str, top_k: int = 3) -> dict:
        pass

    @abstractmethod
    def fetch_data(self, ids: str | list[str]) -> FetchResponse | PineconeGrpcFuture:
        pass

    @abstractmethod
    def delete_data(self, ids: str | list[str]) -> None:
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


    def query_data(self, text: str, top_k: int = 3) -> dict:
        return self.index.query(
            vector=Embedding.create_embedding(text).tolist(),
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )


    def fetch_data(self, ids: str | list[str]) -> FetchResponse | PineconeGrpcFuture:
        return self.index.fetch([ids]) if type(ids) == str else self.index.fetch(ids)


    def delete_data(self, ids: str | list[str]) -> None:
        self.index.delete([ids]) if type(ids) == str else self.index.delete(ids)


if __name__ == "__main__": 
    database: PineconeVectorDatabase = PineconeVectorDatabase()
    print(database.get_index_description())
