import time

import torch


def train_one_epoch(model, loader, criterion, optimizer, device="cpu"):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        # TODO 1: clear gradients left over from the previous step
        optimizer.zero_grad()
        # TODO 2: forward pass -> logits of shape [batch, 10]
        logits = model(X)
        # TODO 3: loss between logits and y
        loss = criterion(logits, y)
        # TODO 4: backpropagate
        loss.backward()
        # TODO 5: update the weights
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += X.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device="cpu"):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        total_loss += criterion(logits, y).item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += X.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def predict(model, loader, device="cpu"):
    model.eval()
    preds, labels = [], []
    for X, y in loader:
        preds.append(model(X.to(device)).argmax(1).cpu())
        labels.append(y)
    return torch.cat(labels).numpy(), torch.cat(preds).numpy()


def fit(model, train_loader, val_loader, criterion, optimizer, epochs,
        scheduler=None, patience=None, ckpt_path=None):
    history, best_acc, bad_epochs = [], -1.0, 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = evaluate(model, val_loader, criterion)
        lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, lr=lr))
        print(f"epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | lr {lr:.2e} | {time.time()-t0:.1f}s",
              flush=True)
        if va_acc > best_acc:
            best_acc, bad_epochs = va_acc, 0
            if ckpt_path:
                torch.save(model.state_dict(), ckpt_path)
        else:
            bad_epochs += 1
            if patience is not None and bad_epochs >= patience:
                print(f"early stopping at epoch {epoch} (best val acc {best_acc:.4f})", flush=True)
                break
    return history
