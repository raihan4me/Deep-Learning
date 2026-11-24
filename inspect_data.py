import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from common import CLASSES, DATA_DIR, get_loaders, set_seed

set_seed(42)

# 1) Verify the normalisation constants from raw [0,1] pixels of the training set
raw_train = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
s = ss = n = 0.0
for X, _ in DataLoader(raw_train, batch_size=10000):
    s += X.sum().item(); ss += (X ** 2).sum().item(); n += X.numel()
mean = s / n
std = (ss / n - mean ** 2) ** 0.5
print(f"pixel mean={mean:.4f} std={std:.4f}")

# 2) Split sizes and a batch
train_loader, val_loader, test_loader = get_loaders(batch_size=128)
print("train/val/test sizes:", len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
Xb, yb = next(iter(train_loader))
print("batch X:", tuple(Xb.shape), Xb.dtype, "| batch y:", tuple(yb.shape), yb.dtype)
print(f"normalised batch mean={Xb.mean():.3f} std={Xb.std():.3f}")

# 3) Class balance in the training split (read labels directly, no image decoding)
tr = train_loader.dataset
counts = torch.bincount(tr.dataset.targets[tr.indices], minlength=10)
print("train class counts:", counts.tolist())

# 4) Headless visual check: ASCII render + PNG grid
raw_test = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=transforms.ToTensor())
img, label = raw_test[0]
chars = " .:-=+*#%@"
for row in img[0]:
    print("".join(chars[min(int(v * len(chars)), len(chars) - 1)] for v in row))
print(f"test[0] label={label} ({CLASSES[label]})")
save_image(torch.stack([raw_test[i][0] for i in range(16)]), "results/sample_grid.png", nrow=8)
print("saved results/sample_grid.png")
