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

DISPLAY_LABELS = {
    "freshapples": "Manzana Fresca",
    "freshbanana": "Banana Fresca",
    "freshoranges": "Naranja Fresca",
    "rottenapples": "Manzana Podrida",
    "rottenbanana": "Banana Podrida",
    "rottenoranges": "Naranja Podrida",
}


def translate_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label)


def quality_status(raw_label: str) -> str:
    return "Apta" if raw_label.startswith("fresh") else "No apta"


def build_prediction_payload(content: bytes) -> dict[str, object]:
    predicted_label, confidence = predict_image(
        content=content,
        model=model,
        transform=transform,
        class_names=DEFAULT_CLASS_NAMES,
        device=device,
    )
    return {
        "predicted_label": translate_label(predicted_label),
        "raw_label": predicted_label,
        "quality": quality_status(predicted_label),
        "confidence": round(confidence, 6),
        "model_path": str(MODEL_PATH),
    }

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

    return jsonify(build_prediction_payload(content))


@app.post("/analyze")
def analyze_batch():
    uploaded_files = request.files.getlist("images")
    if not uploaded_files:
        return jsonify({"error": "Se requiere al menos un archivo en el campo 'images'."}), 400

    items: list[dict[str, object]] = []
    fresh_count = 0
    rotten_count = 0

    for uploaded_file in uploaded_files:
        result = build_prediction_payload(uploaded_file.read())
        item = {
            "filename": uploaded_file.filename,
            **result,
        }
        items.append(item)
        if result["quality"] == "Apta":
            fresh_count += 1
        else:
            rotten_count += 1

    total = len(items)
    return jsonify(
        {
            "total": total,
            "fresh_count": fresh_count,
            "rotten_count": rotten_count,
            "fresh_percentage": round((fresh_count / total) * 100, 2),
            "rotten_percentage": round((rotten_count / total) * 100, 2),
            "items": items,
            "model_path": str(MODEL_PATH),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
