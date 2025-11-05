from flask import Flask, request, render_template
import os

app = Flask(__name__)

# 방명록 파일 경로
# guestbook_file = "/app/data/story.txt"
if os.environ.get("TXT_FOLDER") is None:
    print("TXT_FOLDER is not exist, set default value")
    guestbook_file = "/app/data/story.txt"
else:
    print("TXT_FOLDER is exist")
    guestbook_file= os.environ.get("TXT_FOLDER")

@app.route("/", methods=["GET", "POST"])
def guestbook():
    if request.method == "POST":
        name = request.form["name"]
        message = request.form["message"]
        with open(guestbook_file, "a") as f:
            f.write(f"{name}: {message}\n")

    # 파일에서 최신 방명록 불러오기
    if os.path.exists(guestbook_file):
        with open(guestbook_file, "r") as f:
            messages = f.readlines()
    else:
        messages = ["<h1 style='color: red;'>Error: 방명록 파일이 없습니다.</h1>"]

    return render_template("index.html", messages=messages)


@app.route("/error", methods=["POST"])
def error():
    # 애플리케이션 종료
    os._exit(1)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
