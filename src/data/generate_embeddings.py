import pandas as pd

from os import listdir
from src import CLEARED_DATA_FOLDER
from path import Path
from numpy import array
from typing import Union

from sentence_transformers import SentenceTransformer

file = CLEARED_DATA_FOLDER / listdir(CLEARED_DATA_FOLDER)[0]

class Embedding: 

    model: SentenceTransformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    id: str
    vector: array
    metadata: dict[str, str]
    _default_str: str

    def __init__(self, id: str, content_to_embed: str, metadata: dict[str, str]):
        self.id = id
        self._default_str = content_to_embed
        self.metadata = metadata

        self.vector = Embedding._create_embedding(content_to_embed)

    @classmethod
    def _create_embedding(self, text: str) -> array: 
        embeddings = Embedding.model.encode(text)
       
       # Convert tensor to array, return array


        return array([0])

    def get_pinecone_record(self) -> dict[str, Union[str, array, dict]]:
        return { "id": self.id, "values": self.vector, "metadata": self.metadata }
    
    def __str__(self):
        return "[id: {}\nvector: {}\nmetadata: {}]".format(self.id, self.vector, self.metadata) 

def create_embeddings(path_to_file: Path) -> None: 
    upsert_batch: list[Embedding] = []

    brand_name: str = path_to_file.name[:-4].split("_")[-1]
    dataset_combined: pd.DataFrame = pd.read_csv(path_to_file, index_col=0)
    dataset_combined  = dataset_combined.drop(['Review_Date', 'Author_Name'], axis=1)
    new_indexes = [f"{brand_name}-{i}" for i in range(dataset_combined.shape[0])]
    dataset_combined.index = pd.Index(new_indexes)

    for index, data in dataset_combined.iterrows(): upsert_batch.append(Embedding(index, 
                                                                                  data["Review"], 
                                                                                  {
                                                                                      "text": data["Review"], 
                                                                                      "review_title": data["Review_Title"],
                                                                                      "rating": data["Rating"],
                                                                                      "vehicle_model": data["Vehicle_Title"]
                                                                                  }
                                                                                    ))
    

    print(dataset_combined.head())
    print(upsert_batch[0])

# create_embeddings(file)

test = Embedding("test-1", "I like cats and i walk with them", {})





# for file in listdir(DATA_FOLDER)[:3]:
#     if file.endswith(".csv"):
#         print(DATA_FOLDER / file)
#         pd.concat([dataset_combined, pd.read_csv(DATA_FOLDER / file)], ignore_index=True)

# dataset_combined.head()
# dataset_combined.tail()

# import pinecone
# from sentence_transformers import SentenceTransformer

# # Initialize Pinecone
# pinecone.init(api_key="YOUR_API_KEY", environment="YOUR_ENVIRONMENT")  # Replace with your API key and environment
# index_name = "your-index-name"

# # Create or connect to an index
# if index_name not in pinecone.list_indexes():
#     pinecone.create_index(index_name, dimension=768)  # Adjust dimension based on your vector size
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

# print("Data successfully uploaded to Pinecone!")