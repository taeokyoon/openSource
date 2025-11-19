from langchain_ollama import OllamaLLM

model_name = "qwen3:8b"
# model
llm = OllamaLLM(model=model_name)

# chain 실행
response = llm.invoke("지구의 자전 주기는?")

print(response)
