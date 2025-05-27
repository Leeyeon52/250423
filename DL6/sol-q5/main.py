from flask import Flask, jsonify, request
import tensorflow as tf
import PIL.Image as image # pip install Pillow

app = Flask(__name__)

def work(img, model): # 이미지를 입력하면 숫자를 출력하는 함수
    pred = model.predict(img) # TODO: 모델에 이미지를 넣고 결과를 pred에 저장
    pred = pred[0] # TODO: batch 단위로 나온 결과에서 이미지 하나의 결과를 추출
    idx = tf.math.argmax(pred)  # TODO: 결과중 가장 확률이 높은 index 가져오기
    return idx

@app.route("/", methods=["GET"]) # @app.route를 작성하고, GET Method만 사용합니다.
def predict():    
    imgurl = request.args.get("img")       # img라는 이름으로 url을 받아옴
    result_string="please input image url"
    if imgurl != None:                     # imgurl을 제대로 받은 경우
        imgurl=imgurl.split("?")[0]
        print(imgurl)
        img = image.open("img/"+imgurl)        # 전달받은 url의 이미지 로드  
        img=tf.keras.utils.img_to_array(img)   # 이미지를 행렬로 변환
        print(img.shape)
        img=tf.expand_dims(img, axis=0)        # batch를 위해 한 차원 높게 변환   
        print(img.shape)
        idx = work(img, model)   # 모델을 work 함수를 통해 사용합니다.    
        result_string = "This number is %d"%(idx)   #사용자에게 보여줄 문자열
    return jsonify(result_string)


# Flask 서버를 실행하는 코드입니다.
if __name__ == "__main__":
    model = tf.keras.models.load_model("saved_model") # TODO: 학습된 모델 "mymodel"을 불러오세요.
    app.run(host="0.0.0.0", port=8080, debug=True) # flask 서비스 시작