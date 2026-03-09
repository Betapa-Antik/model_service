from flask import Flask, request, jsonify
from PIL import Image
from utils.predictor import predict

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "message": "Mosquito Classification API"
    })


@app.route("/predict", methods=["POST"])
def predict_image():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        image = Image.open(file).convert("RGB")

        result = predict(image)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)