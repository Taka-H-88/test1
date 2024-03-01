from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

if __name__ == "__main__":
    app.run()

# http://127.0.0.1:5000/  へアクセスすると画面表示確認できる
    