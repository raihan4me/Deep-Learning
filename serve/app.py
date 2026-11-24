import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
MEAN, STD = 0.2860, 0.3530

sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
INPUT_NAME = sess.get_inputs()[0].name
app = FastAPI(title="Fashion-MNIST classifier", version="1.0")


class PredictRequest(BaseModel):
    pixels: list[float]          # 784 raw values in [0, 1], row-major 28x28


def preprocess(pixels: list[float]) -> np.ndarray:
    # TODO 1: convert to a float32 numpy array of shape (1, 1, 28, 28)
    # TODO 2: apply the SAME normalisation as training: (x - MEAN) / STD, keep float32
    x = np.asarray(pixels, dtype=np.float32).reshape(1, 1, 28, 28)
    x = ((x - MEAN) / STD).astype(np.float32)
    return x


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


@app.get("/health")
def health():
    return {"status": "ok", "input": INPUT_NAME}


@app.post("/predict")
def predict(req: PredictRequest):
    if len(req.pixels) != 784:
        raise HTTPException(status_code=422, detail=f"expected 784 pixel values, got {len(req.pixels)}")
    x = preprocess(req.pixels)
    logits = sess.run(None, {INPUT_NAME: x})[0][0]
    probs = softmax(logits)
    k = int(probs.argmax())
    return {"class_id": k, "label": CLASSES[k], "confidence": round(float(probs[k]), 4)}
