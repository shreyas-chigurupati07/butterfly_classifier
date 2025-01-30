from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.predict import PredictionPipeline


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    os.system('python main.py')
    return 'Training done successfully!'


@app.route("/predict", methods=['POST'])
@cross_origin()
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        # Save the uploaded file
        filepath = f"./uploaded_images/{file.filename}"
        file.save(filepath)

        # Initialize PredictionPipeline and predict
        pipeline = PredictionPipeline(filename=filepath)
        prediction_result = pipeline.predict()

        # Return the prediction result
        return jsonify(prediction_result), 200

# def predictRoute():
#     image = request.json['image']
#     decodeImage(image, clApp.filename)
#     result = clApp.classifier.predict()
#     return jsonify(result)


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=5000)
