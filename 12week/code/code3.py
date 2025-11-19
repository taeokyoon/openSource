import re
import math
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

@tool
def basic_calculator_tool(query: str) -> str:
    """사칙연산 등 간단한 수식을 계산합니다. 예: '12 * 7 + 3'"""
    print("basic_calculator_tool 호출됨")
    try:
        return str(eval(query, {"__builtins__": {}}, {"math": math}))
    except Exception as e:
        return f"계산 오류: {e}"

@tool
def factorial_calculator_tool(query: str) -> str:
    """정수 n의 팩토리얼을 계산합니다. 예: '5', '5!', '\"5\"'"""
    print("factorial_calculator_tool 호출됨")
    try:
        s = query.strip()
        if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
            s = s[1:-1].strip()
        m = re.search(r"-?\d+", s)
        if not m:
            return "유효한 정수를 찾을 수 없음"
        k = int(m.group())
        if k < 0:
            return "음수의 팩토리얼은 정의되지 않음"
        return str(math.factorial(k))
    except Exception as e:
        return f"계산 오류: {e}"

def main():
    tools = [basic_calculator_tool, factorial_calculator_tool]
    model = ChatOllama(model="qwen3:8b", temperature=0)

    agent = create_agent(model=model, tools=tools)

    user_q = input("질문을 입력하세요: ")

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content=user_q)
            ]
        }
    )

    print("\n=== 답변 ===")
    final_msg = result["messages"][-1]
    print(final_msg.content)

if __name__ == "__main__":
    main()
