import pandas as pd
import tqdm
import msgpack
import json
import sys
import re

from os import listdir
from src import CLEARED_DATA_FOLDER, EMBEDDINGS_FOLDER
from path import Path
from numpy import ndarray
from typing import Union
import concurrent.futures

from sentence_transformers import SentenceTransformer

class Embedding: 

    model: SentenceTransformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # Adjust ? Move to .env ? 

    id: str
    vector: ndarray
    metadata: dict[str, str]
    _default_str: str

    def __init__(self, id: str, content_to_embed: str, metadata: dict[str, str]):
        self.id = id
        self._default_str = content_to_embed
        self.metadata = metadata

        self.vector = Embedding.create_embedding(content_to_embed)

    def to_pinecone_record(self) -> dict[str, Union[str, ndarray, dict]]:
        return { "id": self.id, "values": self.vector.tolist(), "metadata": self.metadata }
    
    def __str__(self):
        return "[id: {}\nvector: {}\nmetadata: {}]".format(self.id, self.vector, self.metadata) 
    
    def get_size(self) -> int:
        total_size: int = 0
        total_size += self.vector.nbytes

        for val in self.metadata.values():
            if type(val) == str:
                total_size += 2 * len(val)

        return total_size

    @classmethod
    def create_embedding(self, text: str) -> ndarray:
        embeddings: ndarray = Embedding.model.encode(text, convert_to_numpy=True)
        return embeddings


def get_deep_size(obj):
    if isinstance(obj, (str, bytes)):
        return sys.getsizeof(obj) + len(obj.encode("utf-8"))  
    elif isinstance(obj, dict):
        return sys.getsizeof(obj) + sum(get_deep_size(k) + get_deep_size(v) for k, v in obj.items())
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return sys.getsizeof(obj) + sum(get_deep_size(item) for item in obj)
    elif isinstance(obj, float) or isinstance(obj, int):
        return sys.getsizeof(obj)
    else:
        return sys.getsizeof(obj)


def generate_embeddings(input_file: Path, output_file: Path = None, show_progress_bar: bool = True, save_to_json: bool = False, save_to_bin: bool = True) -> list[Embedding] | None: 

    assert input_file.exists()

    embeddings: list[Embedding] = []

    brand_name: str = input_file.name[:-4].split("_")[-1]
    dataset_combined: pd.DataFrame = pd.read_csv(input_file, index_col=0)
    dataset_combined  = dataset_combined.drop(['Review_Date', 'Author_Name'], axis=1)
    new_indexes = [f"{brand_name}-{i}" for i in range(dataset_combined.shape[0])]
    dataset_combined.index = pd.Index(new_indexes)
    dataset_combined = dataset_combined.fillna('')

    if show_progress_bar: progress_bar: tqdm.tqdm = tqdm.tqdm(total=dataset_combined.shape[0], desc=f"Generating for {input_file.name}")

    total_size: int = 0
    size_limit: int = 30 * 2**20 # 30 MB, 40 MB max

    for index, data in dataset_combined.iterrows(): 
        if size_limit < total_size: break
        new_embedding: Embedding = Embedding(    index, 
                                        data["Review"], 
                                        {
                                            "text":            data["Review"],
                                            "review_title":    data["Review_Title"],
                                            "rating":          data["Rating"],
                                            "vehicle_model":   data["Vehicle_Title"]
                                        }
                                          )
        
        embeddings.append(new_embedding.to_pinecone_record())
        total_size += new_embedding.get_size()
        if show_progress_bar: progress_bar.update(1)

    if output_file == None: return embeddings

    if save_to_bin:
        output_file: Path = Path(re.sub(pattern=r'\.[a-z]+', repl="", string=output_file) + ".msgpack")
        with open(output_file, "wb") as out_file: 
            binary_pack = msgpack.packb(embeddings)
            out_file.write(binary_pack)

    if save_to_json:
        output_file: Path = Path(re.sub(pattern=r'\.[a-z]+', repl="", string=output_file) + ".json")
        with open(output_file, "w") as out_file: 
            json_format: str = json.dumps(embeddings)
            out_file.write(json_format)
            return

if __name__ == "__main__": 

    # file: Path = Path("/home/nsjg/Desktop/Vstorm/Chat_prep_prj/strike-team-2-app/resources/cleared_data/Scraped_Car_Review_ford.csv")
    # out = Path("/home/nsjg/Desktop/Vstorm/Chat_prep_prj/strike-team-2-app/src/data/test")
    # generate_embeddings(file, out, True, True, True)

    input_files: list[Path] = CLEARED_DATA_FOLDER.listdir()
    output_files: list[Path] = [EMBEDDINGS_FOLDER / x[:-4] for x in listdir(CLEARED_DATA_FOLDER)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures: list[concurrent.futures.Future] = []

        for input_file, output_file in zip(input_files, output_files):
            futures.append(executor.submit(generate_embeddings, input_file, output_file, True, False, True))

        concurrent.futures.wait(futures)

        """
        Read like this: 
            with open("embeddings.bin", "rb") as file:
                data = file.read()
                obj = msgpack.unpackb(data) # outputs list[dict]
        """
