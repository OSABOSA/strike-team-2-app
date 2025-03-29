import os

from dotenv import load_dotenv
from src import MAIN_FOLDER

load_dotenv(dotenv_path=MAIN_FOLDER / ".env")

class Settings: 

    # Keys
    pinecone_api_key: str = ""
    
    @classmethod
    def load(cls):
        if key := os.getenv("PINECONE_API_KEY"): cls.pinecone_api_key = key
        else: raise ValueError("Pinecone API key not found in environment variables.")
