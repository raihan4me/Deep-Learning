import json

import torch
from torch import nn

from common import get_loaders, set_seed
from engine import evaluate, fit
from models import MLP, count_params

set_seed(42)
train_loader, val_loader, test_loader = get_loaders(batch_size=128)
model = MLP()
print("MLP parameters:", count_params(model))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

history = fit(model, train_loader, val_loader, criterion, optimizer, epochs=5,
              ckpt_path="models/mlp_best.pt")

model.load_state_dict(torch.load("models/mlp_best.pt", weights_only=True))
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f"MLP TEST loss {test_loss:.4f} acc {test_acc:.4f}")
json.dump({"params": count_params(model), "test_acc": test_acc, "history": history},
          open("results/mlp_metrics.json", "w"), indent=2)
