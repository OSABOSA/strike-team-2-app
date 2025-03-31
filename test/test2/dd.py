from langchain_community.llms import LlamaCpp

llm = LlamaCpp(model_path="C:/Users/0oski/.lmstudio/models/lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf")

# Query the model
question = "Who won 2010 FIFA World Cup?"
response = llm(question)
print(response)
