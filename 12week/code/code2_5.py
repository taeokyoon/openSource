from typing import List
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# 1) 구조화 출력 스키마 정의 (Pydantic)
class BlogSummary(BaseModel):
    """블로그 요약 및 키워드 구조."""
    title: str = Field(..., description="블로그 글의 제목 (한 문장)")
    keywords: List[str] = Field(
        ..., description="주제와 관련된 한국어 키워드 목록 (문자열 배열)"
    )

# 2) LLM 준비
llm = ChatOllama(model="qwen3:8b")

# 3) LLM에 구조화 출력 스키마 바인딩
model_with_structure = llm.with_structured_output(BlogSummary)

# 4) 프롬프트 정의
prompt = ChatPromptTemplate.from_template(
    (
        "너는 한국어 블로그 요약 작성기이다.\n"
        "주제 {topic} 에 대해 블로그 요약 제목과 관련 키워드를 생성하라."
    )
)

# 5) 체인 구성: prompt | model_with_structure
chain = prompt | model_with_structure

# 6) 실행
result: BlogSummary = chain.invoke({"topic": "Korea History"})

print(result)
print("제목:", result.title)
print("키워드:", result.keywords)