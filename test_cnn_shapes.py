import torch

from models import CNN, count_params

m = CNN().eval()
x = torch.randn(4, 1, 28, 28)
for i, layer in enumerate(m.features):
    x = layer(x)
    print(f"features[{i}] {layer.__class__.__name__:<12} -> {tuple(x.shape)}")
out = m(torch.randn(4, 1, 28, 28))
print("output:", tuple(out.shape))
print("params:", count_params(m))
assert out.shape == (4, 10), "output shape must be (4, 10)"
assert count_params(m) == 421834, "parameter count differs from the specification"
print("CNN OK")
