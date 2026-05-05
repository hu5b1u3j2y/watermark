from flask import Flask, request, jsonify
import cv2
import numpy as np

from watermark import embed, extract
from attacks import gaussian_noise, jpeg_compression
from metrics import compute_psnr, compute_ber

app = Flask(__name__)
#fhbjdch
np.random.seed(42)

@app.route("/")
def home():
    return "Watermark API is running"

@app.route("/process", methods=["POST"])
def process_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    watermark = np.random.randint(0, 2, 500)

    watermarked = embed(image, watermark)

    attacked = watermarked.copy()

    if request.form.get("attack_noise") == "true":
        attacked = gaussian_noise(attacked)

    if request.form.get("attack_jpeg") == "true":
        attacked = jpeg_compression(attacked)

    psnr = compute_psnr(image, watermarked)

    response = {
        "psnr": float(psnr)
    }

    if request.form.get("attack_noise") == "true" or request.form.get("attack_jpeg") == "true":
        extracted = extract(attacked, len(watermark))
        ber = compute_ber(watermark, extracted)
        response["ber"] = float(ber)

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
