from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. PDF 문서 로드
loader = PyPDFLoader("./kyonngi.pdf")
docs = loader.load()

# 2. 문서 분할
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
splits = splitter.split_documents(docs)  # List[Document]

# 3. 임베딩 생성 (최신: langchain_huggingface)
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. 벡터 저장소 생성 (최신: langchain_chroma)
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory="./person_db",
)

# 5. Ollama 기반 LLM 불러오기 (Chat 모델)
llm = ChatOllama(model="qwen3:8b", temperature=0.2)

# 6. RAG용 프롬프트 정의
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

# 7. 검색된 문서들을 하나의 문자열로 합치는 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 8. Retriever 생성
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 9. LCEL로 RAG 체인 구성
#   입력: "질문 문자열"
#   흐름:
#     - context: retriever로 검색 → format_docs
#     - question: 그대로 전달
#     - rag_prompt: context + question으로 프롬프트 생성
#     - llm: 답변 생성
#     - StrOutputParser: 최종 문자열만 뽑기
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
query = "김경기라는 인물에 대해 궁금합니다."
answer = rag_chain.invoke(query)  # 문자열 하나로 바로 전달 가능
print(answer)
