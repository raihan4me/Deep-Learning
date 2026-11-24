import argparse
import csv
import os

import torch
from torch import nn

from common import get_clean_train_loader, get_loaders, set_seed
from engine import evaluate, fit
from models import CNN

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--dropout", type=float, default=0.3)
p.add_argument("--wd", type=float, default=0.0)
p.add_argument("--augment", action="store_true")
p.add_argument("--scheduler", choices=["none", "cosine"], default="none")
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--epochs", type=int, default=8)
p.add_argument("--subset", type=int, default=10000)
p.add_argument("--patience", type=int, default=None)
args = p.parse_args()
print(vars(args), flush=True)

set_seed(42)
train_loader, val_loader, _ = get_loaders(batch_size=128, augment=args.augment, train_subset=args.subset)
clean_train_loader = get_clean_train_loader(train_subset=args.subset)
model = CNN(dropout=args.dropout)
criterion = nn.CrossEntropyLoss()

# TODO 1: AdamW optimiser with lr=args.lr and weight_decay=args.wd
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
# TODO 2: None for "none", CosineAnnealingLR(optimizer, T_max=args.epochs) for "cosine"
scheduler = None if args.scheduler == "none" else \
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

ckpt = f"models/exp_{args.name}.pt"
history = fit(model, train_loader, val_loader, criterion, optimizer, epochs=args.epochs,
              scheduler=scheduler, patience=args.patience, ckpt_path=ckpt)

model.load_state_dict(torch.load(ckpt, weights_only=True))
_, train_acc = evaluate(model, clean_train_loader, criterion)
_, val_acc = evaluate(model, val_loader, criterion)
row = dict(name=args.name, dropout=args.dropout, wd=args.wd, augment=args.augment,
           scheduler=args.scheduler, epochs_run=len(history),
           train_acc=f"{train_acc:.4f}", val_acc=f"{val_acc:.4f}", gap=f"{train_acc - val_acc:.4f}")
path = "results/experiments.csv"
new = not os.path.exists(path)
with open(path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row))
    if new:
        w.writeheader()
    w.writerow(row)
print(f"RESULT {args.name}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
      f"gap={train_acc - val_acc:.4f} epochs_run={len(history)}")
