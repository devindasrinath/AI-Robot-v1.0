"""
Train Nexu GRU behavior policy.

Input:  X [N, OBS_SIZE] flat obs (sim generates one-step observations)
Target: y [N] action index

Training uses the GRU in single-step mode with a zero hidden state each batch
(history is implicit in the simulation's sequential ordering, but we train
step-wise since hidden state resets between episodes are handled by the runner).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from pathlib import Path

from gru_policy import NexuGRUPolicy, OBS_SIZE, HIDDEN_SIZE, NUM_ACTIONS

ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

EPOCHS     = 60
BATCH_SIZE = 256
LR         = 1e-3
PATIENCE   = 12


def load_data():
    X_train = np.load(str(DATA_DIR / "X_train.npy"))
    y_train = np.load(str(DATA_DIR / "y_train.npy"))
    X_val   = np.load(str(DATA_DIR / "X_val.npy"))
    y_val   = np.load(str(DATA_DIR / "y_val.npy"))
    return X_train, y_train, X_val, y_val


def make_loader(X, y, shuffle=True, weighted=False):
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # [N,1,16]
    y_t = torch.tensor(y, dtype=torch.long)
    ds  = TensorDataset(X_t, y_t)

    if weighted:
        counts  = np.bincount(y, minlength=NUM_ACTIONS).astype(float)
        weights = 1.0 / (counts + 1e-6)
        sample_w = torch.tensor([weights[a] for a in y], dtype=torch.float32)
        sampler  = WeightedRandomSampler(sample_w, len(sample_w))
        return DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler)

    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def train():
    device = torch.device("cpu")
    print(f"Device: {device}")

    X_train, y_train, X_val, y_val = load_data()
    print(f"Train: {len(X_train)}  Val: {len(X_val)}")

    train_loader = make_loader(X_train, y_train, weighted=True)
    val_loader   = make_loader(X_val,   y_val,   shuffle=False)

    model = NexuGRUPolicy().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    no_improve   = 0
    best_path    = ROOT / "model" / "gru_behavior.pth"
    best_path.parent.mkdir(exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            h = torch.zeros(X_b.size(0), HIDDEN_SIZE, device=device)
            logits, _ = model(X_b, h)
            loss = loss_fn(logits, y_b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(y_b)

        sched.step()

        # Validation
        model.eval()
        correct = 0
        total   = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                h = torch.zeros(X_b.size(0), HIDDEN_SIZE, device=device)
                logits, _ = model(X_b, h)
                preds = logits.argmax(dim=1)
                correct += (preds == y_b).sum().item()
                total   += len(y_b)

        val_acc  = correct / total
        avg_loss = total_loss / len(X_train)

        print(f"Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(best_path))
            print(f"  → Saved best model (val_acc={best_val_acc:.3f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stop at epoch {epoch}")
                break

    print(f"\nTraining complete. Best val_acc: {best_val_acc:.3f}")
    return best_val_acc


if __name__ == "__main__":
    train()
