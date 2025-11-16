import json
import numpy as np
from pathlib import Path
from PIL import Image
import onnxruntime as ort
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_unquant.onnx"
LABELS_PATH = BASE_DIR / "updated_hieroglyphs_copy.json"

IMG_SIZE = 224

# load labels
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    LABELS = json.load(f)

# create ORT session
session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def preprocess_image(filename: str) -> np.ndarray:
    p = (BASE_DIR / filename).resolve()
    print("Attempting to load:", p)

    img = Image.open(p).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr / 127.0) - 1.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_hieroglyph(filename: str):
    x = preprocess_image(filename)

    outputs = session.run([output_name], {input_name: x})
    probs = outputs[0][0]

    # sort probs
    top_indices = np.argsort(probs)[::-1][:5]

    # build list of top-5 dicts
    top5 = []
    for i in top_indices:
        top5.append({"index": i, "info": LABELS[i], "confidence": float(probs[i])})

    # top-1 result
    best_index = int(top_indices[0])
    best_info = LABELS[best_index]
    best_conf = float(probs[best_index])

    return best_index, best_info, best_conf, top5


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predictor.py <image>")
        exit()

    img = sys.argv[1]
    idx, info, probs = predict_hieroglyph(img)
    print("\nFinal prediction:", idx, info)
