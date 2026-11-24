import json
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from models import CNN

model = CNN()
model.load_state_dict(torch.load("models/cnn_best.pt", weights_only=True))
model.eval()                                    # BatchNorm uses running stats, Dropout is off

torch.onnx.export(model, torch.randn(1, 1, 28, 28), "models/cnn.onnx",
                  input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                  opset_version=17)
onnx.checker.check_model(onnx.load("models/cnn.onnx"))
print(f"exported models/cnn.onnx ({os.path.getsize('models/cnn.onnx')/1e6:.2f} MB), checker passed")

sess = ort.InferenceSession("models/cnn.onnx", providers=["CPUExecutionProvider"])
x = torch.randn(8, 1, 28, 28)
with torch.no_grad():
    ref = model(x).numpy()
out = sess.run(None, {"input": x.numpy()})[0]
max_diff = float(np.abs(ref - out).max())
print(f"batch-8 output shape {out.shape}, max |torch - onnx| = {max_diff:.2e}")

one = np.random.randn(1, 1, 28, 28).astype(np.float32)
for _ in range(20):
    sess.run(None, {"input": one})
t0 = time.perf_counter()
for _ in range(200):
    sess.run(None, {"input": one})
latency_ms = (time.perf_counter() - t0) / 200 * 1000
print(f"ONNX Runtime batch-1 latency: {latency_ms:.3f} ms")
json.dump({"max_abs_diff": max_diff, "latency_ms": latency_ms}, open("results/onnx_parity.json", "w"))
assert max_diff < 1e-4
