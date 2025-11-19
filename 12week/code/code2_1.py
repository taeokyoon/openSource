from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
model_name = "qwen3:8b"
# model
llm = OllamaLLM(model=model_name)

# Template
template = "{task}을 수행하는 로직을 {language}으로 작성해줘"

prompt = PromptTemplate.from_template(template=template)

prompt = prompt.format(task="0부터 10까지 계산", language="python")

# chain 실행
response = llm.invoke(prompt)
print(response)
