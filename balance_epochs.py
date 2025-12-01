"""
balance_epochs.py
-----------------
Balances target vs non-target epochs by upsampling target samples.
Run this after epoch_extraction.py
"""

import numpy as np
import os

DATA_DIR = "data/processed"

def balance_subject(subj):
    path = os.path.join(DATA_DIR, f"Subject_{subj}/train_epochs")
    X = np.load(os.path.join(path, "X.npy"))
    y = np.load(os.path.join(path, "y.npy"))

    print(f"Before balancing: {np.bincount(y)} (non-target, target)")

    target_idx = np.where(y == 1)[0]
    non_target_idx = np.where(y == 0)[0]

    # --- Upsample targets 4x (or adjust ratio dynamically) ---
    repeat_factor = int(len(non_target_idx) / len(target_idx))
    repeat_factor = max(1, min(repeat_factor, 4))  # safety cap

    upsampled_targets_X = np.repeat(X[target_idx], repeat_factor, axis=0)
    upsampled_targets_y = np.repeat(y[target_idx], repeat_factor, axis=0)

    X_balanced = np.concatenate([X[non_target_idx], upsampled_targets_X], axis=0)
    y_balanced = np.concatenate([y[non_target_idx], upsampled_targets_y], axis=0)

    # Shuffle to randomize order
    indices = np.arange(len(y_balanced))
    np.random.shuffle(indices)
    X_balanced, y_balanced = X_balanced[indices], y_balanced[indices]

    np.save(os.path.join(path, "X_balanced.npy"), X_balanced)
    np.save(os.path.join(path, "y_balanced.npy"), y_balanced)

    print(f"After balancing: {np.bincount(y_balanced)} (non-target, target)")
    print(f"✅ Balanced files saved: X_balanced.npy, y_balanced.npy\n")


if __name__ == "__main__":
    for subj in ["A", "B"]:
        balance_subject(subj)
    print("🎯 Balancing complete for all subjects.")
