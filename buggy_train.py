import torch
from torch import nn

from common import get_loaders, set_seed

set_seed(0)
train_loader, val_loader, _ = get_loaders(batch_size=64, train_subset=20000)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.body(x)


model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=3.0)

for epoch in range(2):
    model.train()
    running = 0.0
    for X, y in train_loader:
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running += loss.item()
    correct = n = 0
    for X, y in val_loader:
        pred = model(X).argmax(1)
        correct += (pred == y).sum().item()
        n += len(y)
    print(f"epoch {epoch+1}: mean train loss {running/len(train_loader):.4f} val acc {correct/n:.4f}")

acc = correct / n
print(f"FINAL_VAL_ACC={acc:.4f}")
with open("results/debug_result.txt", "w") as f:
    f.write(f"{acc:.4f}\n")
