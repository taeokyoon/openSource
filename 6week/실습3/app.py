from flask import Flask, render_template
import mysql.connector
import os
import time

app = Flask(__name__)

# MySQL 연결 설정
def connect_to_db(retries=5, delay=5):
    for attempt in range(retries):
        try:
            print(f"Attempt {attempt + 1} to connect to the database...")
            connection = mysql.connector.connect(
                host=os.environ.get('MYSQL_HOST', 'localhost'),
                user=os.environ.get('MYSQL_USER', 'testuser'),
                password=os.environ.get('MYSQL_PASSWORD', 'testpassword'),
                database=os.environ.get('MYSQL_DATABASE', 'testdb')
            )
            print("Connected to the database.")
            return connection
        except mysql.connector.Error as err:
            print(f"Error: {err}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Failed to connect to the database after several attempts.")
                raise Exception("Could not connect to the database.") from err

@app.route('/')
def index():
    # MySQL 데이터베이스 연결
    db = connect_to_db()
    cursor = db.cursor()

    # users 테이블에서 데이터 가져오기
    query = "SELECT id, username, email FROM users"
    cursor.execute(query)
    users = cursor.fetchall()

    # 연결 종료
    cursor.close()
    db.close()

    # 데이터를 HTML 페이지에 전달
    return render_template('index.html', users=users)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
