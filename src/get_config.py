import os

from dotenv import load_dotenv
from src import MAIN_FOLDER

load_dotenv(dotenv_path=MAIN_FOLDER / ".env")

class Settings: 

    # Keys
    pinecone_api_key: str = ""
    openai_api_key: str = ""
    
    @classmethod
    def load(cls):
        try: 
            cls.pinecone_api_key = os.getenv("PINECONE_API_KEY")
            cls.openai_api_key = os.getenv("OPENAI_API_KEY")

        except ValueError: raise ValueError("Pinecone API key not found in environment variables.")

        if key := os.getenv("PINECONE_API_KEY"): cls.pinecone_api_key = key
        else: raise ValueError("Pinecone API key not found in environment variables.")
