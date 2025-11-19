import requests
from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. 웹 기사 스크래핑
url = "https://www.etnews.com/20250331000230"
response = requests.get(url)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")
article_div = soup.find("div", class_="article_body")
article = article_div.get_text(strip=True) if article_div else ""

if not article:
    raise ValueError("기사 본문을 찾지 못했습니다. HTML 구조(class='article_body')를 확인해 주세요.")

# 2. 텍스트 전처리 및 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_text(article)

# 3. 임베딩 생성 (최신 경로: langchain_huggingface)
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. 벡터 저장소 생성 (최신 경로: langchain_chroma)
vectorstore = Chroma.from_texts(
    texts=splits,
    embedding=embedding,
    persist_directory="./chroma_db",
)

# 5. Retriever 생성
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 6. LLM 준비 (ChatOllama – Chat 모델)
llm = ChatOllama(model="qwen3:8b", temperature=0.2)

# 7. RAG용 프롬프트 정의
#   - context: 검색된 문서들
#   - question: 사용자의 질문
rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "다음 컨텍스트를 사용해서 사용자의 질문에 한국어로 간결하게 답하라. "
            "모르면 모른다고 말하라.\n\n컨텍스트:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

# 8. 문서들을 문자열로 합치는 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 9. LCEL로 RAG 체인 구성
#   입력: {"question": "..."}
#   흐름:
#     - context: retriever로 문서 검색 후 format_docs 적용
#     - question: 그대로 전달
#     - rag_prompt: context + question으로 프롬프트 생성
#     - llm: 답변 생성
#     - StrOutputParser: AIMessage → 문자열
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# 10. 질문 실행
query = "국정원이 공공기관의 AI 정보화사업을 조사한 이유는 무엇인가요?"
answer = rag_chain.invoke(query)  # {"question": query} 대신 string 전달도 지원됨
print(answer)
