import mysql.connector
import os

def main():
    host = os.environ.get('MYSQL_HOST')
    database = os.environ.get('MYSQL_DATABASE')
    user = os.environ.get('MYSQL_USER')
    password = os.environ.get('MYSQL_PASSWORD')

    # MySQL 데이터베이스에 연결
    connection = mysql.connector.connect(
        host=host,
        database=database,
        user=user,
        password=password
    )

    cursor = connection.cursor()
    cursor.execute("SELECT * FROM students")

    # 결과 출력
    rows = cursor.fetchall()
    for row in rows:
        print(row)

    cursor.close()
    connection.close()


if __name__ == '__main__':
    main()