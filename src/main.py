import requests
from fastapi import HTTPException
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())
api_key = os.getenv("Langchain_HuggingFace_Strike2_Access_Token")
API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
headers = {'Authorization': f'Bearer {api_key}'}


def query(payload):
    """Query the HuggingFace model API."""
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)


def analyze_text(text: str):
    """Analyze the input text using the HuggingFace model."""
    result = query({"inputs": text})
    return result
