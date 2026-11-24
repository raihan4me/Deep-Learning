import json
import sys

from torchvision import datasets, transforms

from common import CLASSES, DATA_DIR

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ds = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=transforms.ToTensor())
img, label = ds[idx]
json.dump({"pixels": img.flatten().tolist()}, open("results/request.json", "w"))
print(f"wrote results/request.json for test[{idx}], true label {label} ({CLASSES[label]})")
