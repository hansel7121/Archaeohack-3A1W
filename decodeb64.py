import base64
from pathlib import Path

b64_path = Path("quant.txt")

b64_text = b64_path.read_text().strip()

onnx_bytes = base64.b64decode(b64_text)

output_path = Path("quant.onnx")

with open(output_path, "wb") as f:
    f.write(onnx_bytes)

print("Wrote ONNX model to:", output_path.resolve())
