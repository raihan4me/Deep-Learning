import json
import torch

torch.manual_seed(42)

# ---------- Part A: autograd and gradient accumulation ----------
x = torch.tensor(2.0, requires_grad=True)
for i in range(2):
    y = x ** 3 + 2 * x
    y.backward()
    print(f"after backward #{i+1}: x.grad = {x.grad.item():.1f}  (analytic dy/dx = {3*2**2+2:.1f})")

# ---------- Part B: linear regression with manual gradient descent ----------
X = torch.linspace(-1, 1, 200).unsqueeze(1)          # (200, 1)
y = 3 * X + 2 + 0.1 * torch.randn_like(X)            # true w=3, b=2, noise std 0.1

w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.1

for epoch in range(200):
    # TODO 1: forward pass -> y_hat
    y_hat = X * w + b
    # TODO 2: mean squared error between y_hat and y
    loss = ((y_hat - y) ** 2).mean()
    # TODO 3: compute gradients of loss w.r.t. w and b
    loss.backward()
    # TODO 4: update w and b with plain gradient descent (inside torch.no_grad()),
    #         then reset w.grad and b.grad to zero
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()
    b.grad.zero_()

    if epoch % 50 == 0:
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}  w {w.item():.3f}  b {b.item():.3f}")

print(f"FINAL w={w.item():.3f} b={b.item():.3f} loss={loss.item():.4f}")
json.dump({"w": w.item(), "b": b.item(), "loss": loss.item()}, open("results/linreg.json", "w"))
