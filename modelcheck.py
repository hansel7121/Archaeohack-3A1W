from pathlib import Path
import onnx
import onnxruntime as ort

path = Path("hieroglyph_model.onnx")
print("File size (bytes):", path.stat().st_size)

model = onnx.load(str(path))
onnx.checker.check_model(model)
print("onnx.checker: model is valid ✅")

sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
print("onnxruntime: session created ✅", sess)
