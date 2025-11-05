from flask import Flask, render_template, request, redirect, url_for, session
import time
import mysql.connector
from mysql.connector import Error
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# MySQL 연결 설정
db_config = {
    'host': os.environ.get('MYSQL_HOST', 'mysql-container'),
    'user': os.environ.get('MYSQL_USER', 'testuser'),
    'password': os.environ.get('MYSQL_PASSWORD', '1234'),
    'database': os.environ.get('MYSQL_DATABASE', 'testdb')
}

def connect_to_db():
    #print all configs
    print(db_config)
    attempts = 5
    while attempts > 0:
        try:
            db = mysql.connector.connect(**db_config)
            print("Connected to the database.")
            return db
        except Error as e:
            print(f"Error: {e}. Retrying in 5 seconds...")
            attempts -= 1
            time.sleep(5)
    raise Exception("Could not connect to the database.")

db = connect_to_db()

guestbook_dir = './guestbook_entries'
if not os.path.exists(guestbook_dir):
    os.makedirs(guestbook_dir)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cursor = db.cursor()
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
        db.commit()
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cursor = db.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        user = cursor.fetchone()

        if user:
            session['username'] = username
            return redirect(url_for('guestbook'))
        else:
            return '로그인 실패! 사용자 이름이나 비밀번호가 잘못되었습니다.'

    return render_template('index.html')

@app.route('/guestbook', methods=['GET', 'POST'])
def guestbook():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        message = request.form['message']
        username = session['username']

        # 방명록 파일 저장
        file_path = f'./guestbook_entries/{username}_entry.txt'
        with open(file_path, 'a') as file:
            file.write(f"{username}: {message}\n")

        return redirect(url_for('guestbook'))

    return render_template('guestbook.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


if __name__ == '__main__':
    port = int(os.environ.get('FLASK_PORT', 5000))
    app.run(host='0.0.0.0', port=port)
    # print envs
    print(os.environ)

