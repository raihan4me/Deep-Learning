import math

import torch
from torch import nn

from common import get_loaders, set_seed

set_seed(0)
train_loader, _, _ = get_loaders(batch_size=32, train_subset=2000)
X, y = next(iter(train_loader))


def make(softmax_head: bool):
    layers = [nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)]
    if softmax_head:
        layers.append(nn.Softmax(dim=1))
    return nn.Sequential(*layers)


for name, softmax_head in [("logits head ", False), ("softmax head", True)]:
    torch.manual_seed(0)
    model = make(softmax_head)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    for step in range(301):
        opt.zero_grad()
        loss = crit(model(X), y)
        loss.backward()
        opt.step()
        if step in (0, 100, 300):
            print(f"{name} step {step:3d} loss {loss.item():.4f}")
print(f"expected initial loss ln(10) = {math.log(10):.4f}")
print(f"theoretical floor with a softmax head = {math.log(math.e + 9) - 1:.4f}")
