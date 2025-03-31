from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())
api_key = os.getenv("Langchain_HuggingFace_Strike2_Access_Token")
API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
headers = {'Authorization': f'Bearer {api_key}'}


client = InferenceClient(
	provider="hf-inference",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
	api_key=api_key,
)

messages = [{"role": "user", "content": "What is the capital of France?"}]
client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")

result = client.chat_completion(messages, max_tokens=100)

print(result)
