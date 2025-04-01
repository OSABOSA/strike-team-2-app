from huggingface_hub import InferenceClient, login
from dotenv import load_dotenv, find_dotenv
from langchain_community.llms import LlamaCpp
import os


class LocalModels:

    @staticmethod
    def ask_local_llm(question: str, model_path: str = "C:/Users/0oski/.lmstudio/models/lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf") -> str:
        
        llm = LlamaCpp(
            model_path=model_path)
        response = llm.invoke(question)
        return response


class HuggingFaceModels:
    def __init__(self):
        load_dotenv(find_dotenv())
        api_key = os.getenv("Langchain_HuggingFace_Strike2_Access_Token")
        login(token=api_key)
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=api_key,
        )

    def sentiment_analysis(self, text: str, model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """Analyze the input text using the HuggingFace model."""
        return self.client.text_classification(text, model=model)

    def image_to_text(self, image_path: str, model: str = "Salesforce/blip-image-captioning-base"):
        """Convert an image to text using the HuggingFace model."""
        return self.client.image_to_text(image_path, model=model)

    def text_to_image(self, text: str, model: str = "black-forest-labs/FLUX.1-dev"):
        """Convert text to an image using the HuggingFace model."""
        return self.client.text_to_image(text, model=model)

    def llm(self, message: str, max_tokens: int = 500, model: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        response = ""
        self.client.provider = "fireworks-ai"
        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"{message}"
                }
            ],
            max_tokens=max_tokens,
        )
        self.client.provider = "hf-inference"
        response = completion.choices[0].message
        return response
