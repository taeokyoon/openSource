# 베이스 이미지로 Python 3.9 사용
FROM python:latest

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 파일을 컨테이너에 복사
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Flask 애플리케이션 코드 복사
COPY . .

# 컨테이너에서 노출할 포트
EXPOSE 5000

# Flask 서버 실행
CMD ["python", "app.py"]
