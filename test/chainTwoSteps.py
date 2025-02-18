import os
import requests
from fastapi import HTTPException
from dotenv import load_dotenv, find_dotenv

# Ładowanie zmiennych środowiskowych
load_dotenv(find_dotenv())
api_key = os.getenv("Langchain_HuggingFace_Strike2_Access_Token")

# API do analizy sentymentu
API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
headers = {'Authorization': f'Bearer {api_key}'}


def query(payload):
    """Zapytanie do API modelu Hugging Face."""
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)


def analyze_text(text: str):
    """Analiza sentymentu tekstu przy użyciu modelu Hugging Face."""
    result = query({"inputs": text})
    return result


# -------------------- Chain Logika przy użyciu LangChain --------------------
# Upewnij się, że masz zainstalowany pakiet langchain:
# pip install langchain

from langchain.chains import TransformChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub


# 1. Łańcuch do analizy sentymentu – opakowujemy naszą funkcję analyze_text
class SentimentChain(TransformChain):
    def _call(self, inputs: dict) -> dict:
        text = inputs["text"]
        result = analyze_text(text)
        # Załóżmy, że API zwraca listę słowników, pobieramy pierwszy wynik
        sentiment_label = result[0]['label'] if result and "label" in result[0] else "N/A"
        sentiment_score = result[0]['score'] if result and "score" in result[0] else 0
        return {"sentiment": f"Sentyment: {sentiment_label} (score: {sentiment_score:.2f})"}

    @property
    def input_keys(self):
        return ["text"]

    @property
    def output_keys(self):
        return ["sentiment"]


sentiment_chain = SentimentChain()

# 2. Łańcuch do podsumowania tekstu
# Korzystamy z HuggingFaceHub do załadowania modelu do podsumowywania
llm_summary = HuggingFaceHub(
    repo_id="facebook/bart-large-cnn",
    model_kwargs={"temperature": 0, "max_length": 512}
)

prompt_template = PromptTemplate(
    template="Podsumuj następujący tekst:\n\n{text}\n\nPodsumowanie:",
    input_variables=["text"]
)
summarization_chain = LLMChain(llm=llm_summary, prompt=prompt_template)


# 3. Łączymy oba kroki w jedną funkcję – wykonujemy operacje równolegle
def combined_chain(text: str) -> dict:
    sentiment_result = sentiment_chain({"text": text})["sentiment"]
    summary_result = summarization_chain.run(text=text)
    return {"sentiment": sentiment_result, "summary": summary_result}


# Przykładowe użycie:
if __name__ == "__main__":
    sample_text = (
        "Dzisiaj miałem fantastyczny dzień! Pogoda była piękna, a spotkanie z przyjaciółmi "
        "dodało mi energii. Czułem się naprawdę szczęśliwy i pełen pozytywnej energii."
    )
    results = combined_chain(sample_text)
    print("Wyniki łańcucha operacji:")
    print("Analiza sentymentu:", results["sentiment"])
    print("Podsumowanie:", results["summary"])
