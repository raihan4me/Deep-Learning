import torch

torch.manual_seed(0)
x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print("x shape:              ", tuple(x.shape))
print("x.T @ x shape:        ", tuple((x.T @ x).shape))

b = torch.tensor([1.0, 2.0, 3.0, 4.0])          # shape (4,) broadcasts over rows
print("broadcast add, row 0: ", (x + b)[0].tolist())

imgs = torch.randn(32, 1, 28, 28)               # a batch of 32 grayscale images (N, C, H, W)
flat = imgs.view(imgs.size(0), -1)              # what nn.Flatten does
print("flattened batch:      ", tuple(flat.shape))
print("per-channel mean shape:", tuple(imgs.mean(dim=(0, 2, 3)).shape))
