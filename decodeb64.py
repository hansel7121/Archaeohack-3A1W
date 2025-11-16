import base64
from pathlib import Path

# Path to your base64 text file
b64_path = Path("quant.txt")  # <-- put your base64 in here

# Read the base64 text (no quotes, just raw base64)
b64_text = b64_path.read_text().strip()

# Decode base64 to raw bytes
onnx_bytes = base64.b64decode(b64_text)

# Write ONNX file
output_path = Path("quant.onnx")

with open(output_path, "wb") as f:
    f.write(onnx_bytes)

print("Wrote ONNX model to:", output_path.resolve())
