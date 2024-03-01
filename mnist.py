import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# 分類したいクラス名をclassesリストに格納
classes = ["0","1","2","3","4","5","6","7","8","9"]
image_size = 28
# 今回は、MNISTのデータセットを用いたので28

# アップロードされた画像を保存するフォルダ名を渡す。アップロード許可する拡張子を指定。
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# Flaskクラスのインスタンスを作成
app = Flask(__name__)

# ファイルの拡張子のチェックをする関数を定義
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 学習済みモデルをロード
model = load_model('./model.h5')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # POSTリクエストにファイルデータが含まれているか、また、ファイルにファイル名があるかをチェック
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        
        # アップロードされたファイルの拡張子をチェック
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename) # secure_filename()でファイル名に危険な文字列がある場合に無効化（サニタイズ）
            # os.path.join()で引数に与えられたパスをosに応じて結合(Windowsでは￥で結合)し、そのパスにアップされた画像を保存
            file.save(os.path.join(UPLOAD_FOLDER, filename)) 
            filepath = os.path.join(UPLOAD_FOLDER, filename) # その保存先をfilepathに格納

            #受け取った画像を読み込み、np形式に変換. image.load_imgという画像のロードとリサイズを同時にできる関数
            img = image.load_img(filepath, grayscale=True, target_size=(image_size,image_size))
            img = image.img_to_array(img)  # 引数に与えられた画像をNumpy配列に変
            data = np.array([img])   # model.predict()にはNumpy配列のリストを渡す必要があるため、np.array()にimgをリストとして渡す
            #変換したデータをモデルに渡して予測する
            result = model.predict(data)[0]
            predicted = result.argmax()
            pred_answer = "これは " + classes[predicted] + " です"

            # render_templateの引数にanswer=pred_answerと渡すことで、index.htmlに書いたanswerにpred_answerを代入
            return render_template("index.html",answer=pred_answer)

    # POSTリクエストがなされないとき（単にURLにアクセスしたとき）にはindex.htmlのanswerには何も表示しない
    return render_template("index.html",answer="")

# 最後にapp.run()が実行され、サーバが立ち上がる
# if __name__ == "__main__":
#     app.run()

# webアプリをデプロイする
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)
# サーバーを外部からも利用可能にするためにhost='0.0.0.0'と指定
# port = int(os.environ.get('PORT', 8080))では、Renderで使えるポート番号を取得してportに格納。設定されてなければ8080が格納


