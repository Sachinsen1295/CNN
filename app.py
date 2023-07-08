from flask import Flask,request,jsonify,render_template
from utils.utils import decodeimage,encodeimage
import os
from flask_cors import CORS,cross_origin
from predict import dogcat

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self) -> None:
        self.filename = "inputimage.jpg"
        self.classifier = dogcat(self.filename)

@app.route("/", methods =['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods = ['POST'])
@cross_origin()
def predictRoute():
    image = request.json["image"]
    decodeimage(image,clApp.filename)
    result = clApp.classifier.predictdogcat()
    return jsonify(result)


if __name__ == "__main__":
    clApp = ClientApp()
    #app.run(host='0.0.0.0', port=port)
    app.run(host='0.0.0.0', port=8000, debug=True)

