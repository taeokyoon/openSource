from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# model
llm = OllamaLLM(model="qwen3:8b")

# prompt
prompt_template = ChatPromptTemplate.from_messages(
    [
        "system", "You are a singer. Your name is Jack",
        "human", "What is your name? What about the song {song_name}?, and sing it for me."
        # "ai", "My name is Jack. I love the song {song_name}."
    ]
)

song_name = "Yesterday"

chain = prompt_template | llm
# chain 실행
response = chain.invoke(song_name)

print(response)
