from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# LLM 설정
llm = OllamaLLM(model="qwen3:8b")

# Chain-of-Thought를 유도하는 프롬프트 작성 (Zero-shot CoT)
template = """질문에 대해 중간 추론 과정을 자세히 설명하면서 답변해줘.

질문: {question}

답변:"""

prompt = PromptTemplate.from_template(template=template)

# 프롬프트에 실제 질문 채워넣기
formatted_prompt = prompt.format(
    question="철수에게 사과 3개, 영희에게 사과 5개를 줬다. 총 몇 개의 사과가 있는가?"
)

# 모델에게 요청
response = llm.invoke(formatted_prompt)

# 결과 출력
print(response)
