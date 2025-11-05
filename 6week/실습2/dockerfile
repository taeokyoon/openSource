# 베이스 이미지로 Python 3.11 사용
FROM python:latest

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지 복사
COPY requirements.txt .

# Flask와 필요한 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 애플리케이션 실행 (포트 5001에서 실행)
CMD ["python", "app.py"]

# 컨테이너의 5001번 포트 오픈
EXPOSE 5001
