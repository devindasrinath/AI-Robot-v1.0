"""
Train WakeWordCNN on "Nexu" vs everything-else.

Run: python wake_word_model/model/train.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from tqdm import tqdm
from wake_word_cnn import WakeWordCNN

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS     = 40
BATCH_SIZE = 32
LR         = 1e-3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_OUT = ROOT / "model" / "wake_word.pth"


# ── Dataset ───────────────────────────────────────────────────────────────────
class MelDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            x = self._spec_augment(x)
        return x, self.y[idx]

    @staticmethod
    def _spec_augment(x: torch.Tensor) -> torch.Tensor:
        """SpecAugment: random time and frequency masking."""
        _, n_mels, n_frames = x.shape

        # Frequency masking (mask up to 8 mel bins)
        f = torch.randint(0, 8, (1,)).item()
        f0 = torch.randint(0, n_mels - f, (1,)).item() if f > 0 else 0
        x[:, f0:f0+f, :] = 0.0

        # Time masking (mask up to 15 frames)
        t = torch.randint(0, 15, (1,)).item()
        t0 = torch.randint(0, n_frames - t, (1,)).item() if t > 0 else 0
        x[:, :, t0:t0+t] = 0.0

        return x


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    print(f"Device: {DEVICE}")

    # Load data
    X_train = np.load(str(DATA_DIR / "X_train.npy"))
    y_train = np.load(str(DATA_DIR / "y_train.npy"))
    X_val   = np.load(str(DATA_DIR / "X_val.npy"))
    y_val   = np.load(str(DATA_DIR / "y_val.npy"))

    print(f"Train: {len(X_train)} samples  Val: {len(X_val)} samples")
    print(f"Train class balance — 0: {(y_train==0).sum()}  1: {(y_train==1).sum()}")

    # Weighted sampler to handle class imbalance
    class_counts = np.bincount(y_train)
    weights = 1.0 / class_counts[y_train]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_ds = MelDataset(X_train, y_train, augment=True)
    val_ds   = MelDataset(X_val,   y_val,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = WakeWordCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_recall = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch.to(DEVICE))
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy())

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)

        pos_mask = all_labels == 1
        recall    = (all_preds[pos_mask] == 1).mean() if pos_mask.any() else 0.0
        neg_mask  = all_labels == 0
        fpr       = (all_preds[neg_mask] == 1).mean() if neg_mask.any() else 0.0
        accuracy  = (all_preds == all_labels).mean()

        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}  "
              f"acc={accuracy:.3f}  recall={recall:.3f}  fpr={fpr:.3f}")

        # Save best model by recall (we care more about not missing wake word)
        if recall > best_val_recall:
            best_val_recall = recall
            torch.save(model.state_dict(), str(MODEL_OUT))
            print(f"  → Saved best model (recall={recall:.3f})")

    print(f"\nTraining complete. Best recall: {best_val_recall:.3f}")
    print(f"Model saved: {MODEL_OUT}")


if __name__ == "__main__":
    train()
