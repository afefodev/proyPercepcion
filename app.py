from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline_batch import DEFAULT_CLASS_NAMES, build_transform, get_device, load_model, predict_image  # noqa: E402

MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT_DIR / "models" / "modelo_final_efficientnetb0.pth"))
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "224"))
PORT = int(os.environ.get("PORT", "5000"))

app = Flask(__name__)
device = get_device()
model = load_model(MODEL_PATH, class_names=DEFAULT_CLASS_NAMES, device=device)
transform = build_transform(IMAGE_SIZE)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health_check():
    return jsonify(
        {
            "status": "ok",
            "model_path": str(MODEL_PATH),
            "classes": list(DEFAULT_CLASS_NAMES),
        }
    )


@app.post("/predict")
def predict():
    if "image" in request.files:
        content = request.files["image"].read()
    else:
        payload = request.get_json(silent=True) or {}
        image_base64 = payload.get("image_base64")
        if not image_base64:
            return jsonify({"error": "Se requiere un archivo 'image' o el campo 'image_base64'."}), 400
        content = base64.b64decode(image_base64)

    predicted_label, confidence = predict_image(
        content=content,
        model=model,
        transform=transform,
        class_names=DEFAULT_CLASS_NAMES,
        device=device,
    )

    return jsonify(
        {
            "predicted_label": predicted_label,
            "confidence": round(confidence, 6),
            "model_path": str(MODEL_PATH),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
