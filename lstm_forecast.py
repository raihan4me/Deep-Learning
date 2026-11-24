import json

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(7)
T, WINDOW = 3000, 30
t = torch.arange(T, dtype=torch.float32)
series = torch.sin(0.1 * t) + 0.5 * torch.sin(0.033 * t) + 0.1 * torch.randn(T)

X_all = series.unfold(0, WINDOW, 1)[:-1].unsqueeze(-1)   # (N, 30, 1): window i = series[i:i+30]
y_all = series[WINDOW:].unsqueeze(-1)                     # (N, 1):     target  = series[i+30]
N = len(X_all)
n_tr, n_va = int(0.7 * N), int(0.15 * N)                  # chronological split, no shuffling
X_tr, y_tr = X_all[:n_tr], y_all[:n_tr]
X_va, y_va = X_all[n_tr:n_tr + n_va], y_all[n_tr:n_tr + n_va]
X_te, y_te = X_all[n_tr + n_va:], y_all[n_tr + n_va:]
print(f"windows: total={N} train={len(X_tr)} val={len(X_va)} test={len(X_te)} | X shape {tuple(X_tr.shape)}")


class Forecaster(nn.Module):
    def __init__(self, hidden: int = 32, layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        # TODO 1: run the LSTM over the sequence
        out, (h_n, c_n) = self.lstm(x)
        # TODO 2: select the output of the LAST time step -> (batch, hidden)
        # TODO 3: project to (batch, 1) with self.head
        return self.head(out[:, -1, :])


model = Forecaster()
assert model(X_tr[:4]).shape == (4, 1), "forward() must return shape (batch, 1)"
print("parameters:", sum(p.numel() for p in model.parameters()))

loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=5e-3)
loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)

best_va, best_state = float("inf"), None
for epoch in range(1, 41):
    model.train()
    max_norm_seen = 0.0
    for xb, yb in loader:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        max_norm_seen = max(max_norm_seen, grad_norm.item())
        opt.step()
    model.eval()
    with torch.no_grad():
        va = loss_fn(model(X_va), y_va).item()
    if va < best_va:
        best_va = va
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if epoch == 1 or epoch % 5 == 0:
        print(f"epoch {epoch:02d} val MSE {va:.4f} | max grad norm before clipping {max_norm_seen:.3f}")

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    lstm_mse = loss_fn(model(X_te), y_te).item()
naive_mse = loss_fn(X_te[:, -1, :], y_te).item()          # "tomorrow = today" baseline
print(f"TEST naive last-value MSE: {naive_mse:.4f}")
print(f"TEST LSTM MSE:             {lstm_mse:.4f}")
print(f"noise floor (0.1^2):       {0.1**2:.4f}")
json.dump({"naive_mse": naive_mse, "lstm_mse": lstm_mse, "best_val_mse": best_va},
          open("results/lstm_metrics.json", "w"), indent=2)
torch.save(model.state_dict(), "models/lstm_forecaster.pt")
