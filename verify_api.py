import json
import urllib.request

import torch
from torchvision import datasets, transforms

from common import DATA_DIR, build_transforms
from models import CNN

raw = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=transforms.ToTensor())
norm = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=build_transforms(False))
model = CNN()
model.load_state_dict(torch.load("models/cnn_best.pt", weights_only=True))
model.eval()

agree = 0
total = 50
for i in range(total):
    body = json.dumps({"pixels": raw[i][0].flatten().tolist()}).encode()
    req = urllib.request.Request("http://localhost:8000/predict", data=body,
                                 headers={"Content-Type": "application/json"})
    api_pred = json.load(urllib.request.urlopen(req))["class_id"]
    with torch.no_grad():
        local_pred = model(norm[i][0].unsqueeze(0)).argmax(1).item()
    agree += int(api_pred == local_pred)
print(f"API vs local PyTorch agreement: {agree}/{total}")
json.dump({"agree": agree, "total": total}, open("results/api_agreement.json", "w"))
