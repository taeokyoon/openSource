import os

def main():
    # 환경 변수에서 가져오기
    num1 = int(os.getenv('NUM1', 0))
    
    # Secrets에서 가져오기
    num2 = int(os.getenv('NUM2', 0))

    # 덧셈 결과 출력
    print(f"The sum of {num1} and {num2} is: {num1 + num2}")

if __name__ == "__main__":
    main()
