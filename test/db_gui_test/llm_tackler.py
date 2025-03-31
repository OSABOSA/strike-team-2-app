from langchain_community.llms import LlamaCpp

def ask_local_llm(question: str) -> str:
    llm = LlamaCpp(model_path="C:/Users/0oski/.lmstudio/models/lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf")
    response = llm(question)
    return response
