from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate

# 예시 데이터 다양화
example = [
    {
        "question": "아이유로 삼행시 써줘",
        "answer": """아: 아침을 깨우는
이: 이 세상의 목소리
유: 유일무이한 아이유"""
    },
    {
        "question": "바나나로 삼행시 써줘",
        "answer": """바: 바람에 살랑이는
나: 나뭇잎처럼 부드럽게
나: 나의 마음을 흔드는 바나나"""
    }
]

# 명확한 instruction 추가
example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="아래 단어의 각 글자로 삼행시를 써주세요.\n{question}\n{answer}"
)

prompt = FewShotPromptTemplate(
    examples=example,
    example_prompt=example_prompt,
    suffix="아래 단어의 각 글자로 삼행시를 써주세요.\n{input}",
    input_variables=["input"],
)

# 모델
llm = OllamaLLM(model="qwen3:8b")

response = llm.invoke(prompt.format(input="기러기로 삼행시 써줘"))

print(response)
