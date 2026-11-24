import json
import os

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

from common import CLASSES, get_loaders, set_seed
from engine import evaluate, fit, predict
from models import CNN, count_params

set_seed(42)
epochs = int(os.environ.get("EPOCHS", "3"))
subset = int(os.environ["TRAIN_SUBSET"]) if os.environ.get("TRAIN_SUBSET") else None
train_loader, val_loader, test_loader = get_loaders(batch_size=128, train_subset=subset)

model = CNN()
print("CNN parameters:", count_params(model), "| epochs:", epochs, "| train subset:", subset)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
history = fit(model, train_loader, val_loader, criterion, optimizer, epochs=epochs,
              ckpt_path="models/cnn_best.pt")

model.load_state_dict(torch.load("models/cnn_best.pt", weights_only=True))
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f"CNN TEST loss {test_loss:.4f} acc {test_acc:.4f}")

y_true, y_pred = predict(model, test_loader)
print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))
cm = confusion_matrix(y_true, y_pred)
off = cm.copy()
np.fill_diagonal(off, 0)
i, j = np.unravel_index(off.argmax(), off.shape)
print(f"most confused: true={CLASSES[i]} predicted={CLASSES[j]} ({off[i, j]} images)")
json.dump({"params": count_params(model), "test_acc": test_acc, "history": history,
           "confusion_matrix": cm.tolist()},
          open("results/cnn_metrics.json", "w"), indent=2)
