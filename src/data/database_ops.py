import pinecone
import pinecone.grpc

from pinecone import IndexModel
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone.grpc import PineconeGrpcFuture
from pinecone.core.openapi.db_data.model.fetch_response import FetchResponse

import msgpack
import tqdm
from src import EMBEDDINGS_FOLDER
from abc import ABC, abstractmethod
from src import get_config
from path import Path


from src.data.generate_embeddings import Embedding


class VectorDatabaseInterface(ABC):

    @abstractmethod
    def upsert_data(self, records: dict | list[dict], batch_size: int) -> bool:
        pass

    @abstractmethod
    def query_data(self, text: str, top_k: int = 3) -> list[str]:
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
        print(indexes_list)
        self.index = self.pc.Index(host=self.index_data["host"])


    def get_index_description(self) -> IndexModel: 
        return self.pc.describe_index(self.index_data["name"])


    def upsert_data(self, records: dict | list[dict], batch_size: int) -> bool: 
        """ Record format is: list[dict] - [
            {"id1", embedding_vector1, {"metadata_key": "value"}},
            {"id2", embedding_vector2, {"metadata_key": "value"}},
            ]
        """
        # max 2 MB size in a batch

        upsert_batch: list[dict] = []

        if type(records) == dict: upsert_batch = [records]
        else: upsert_batch = records

        low_index: int = 0
        high_index: int = batch_size
        
        # Transfer in chunks
        progress_bar: tqdm.tqdm = tqdm.tqdm(total=len(records), desc=f"Upserting...")

        while high_index < len(records):
            try:
                response = self.index.upsert(namespace="default-namespace", vectors=upsert_batch[low_index:high_index])
                assert response.upserted_count == batch_size
            except: print("Upsert did not succeed"); return False

            low_index += batch_size
            high_index += batch_size
            progress_bar.update(batch_size)

        # Transfer reminder
        try:
            response = self.index.upsert(namespace="default-namespace", vectors=upsert_batch[low_index:])
            assert response.upserted_count == len(records)%batch_size
        except AssertionError: print("Upsert did not succeed"); return

        progress_bar.update(len(records)%batch_size)

        return True


    def query_data(self, text: str, top_k: int = 3) -> list[str]:
        """
        Query the Pinecone index for the most similar text chunks.

        Args:
            text (str): The text to query.
            top_k (int): The number of most similar text chunks to return.

        Returns:
            list[str]: The most similar text chunks.
        """
        data = self.index.query(
            vector=Embedding.create_embedding(text).tolist(),
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
        return [match['metadata']['chunk_text'] for match in data['matches']]

    def fetch_data(self, ids: str | list[str]) -> FetchResponse | PineconeGrpcFuture:
        return self.index.fetch([ids]) if type(ids) == str else self.index.fetch(ids)


    def delete_data(self, ids: str | list[str]) -> None:
        self.index.delete([ids]) if type(ids) == str else self.index.delete(ids)


def upsert_file_to_database(path_to_file: Path, batch_size: int) -> bool: 

    database: PineconeVectorDatabase = PineconeVectorDatabase()

    with open(path_to_file, "rb") as file:
        bin_data = file.read() 
        embeddings = msgpack.unpackb(bin_data)
        print("Upserting data from: {}".format(path_to_file.name))
        database.upsert_data(records=embeddings, batch_size=batch_size)



if __name__ == "__main__": 

    input_files: list[Path] = EMBEDDINGS_FOLDER.listdir()

    for embeddings_file in input_files:
        upsert_file_to_database(embeddings_file, 100)

