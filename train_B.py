"""
Final Stable BCNN-LSTM Training Script
--------------------------------------
✅ Handles class imbalance (1:5 ratio)
✅ Balanced sampling + weighted ELBO loss
✅ Label smoothing + KL annealing
✅ Monte Carlo prediction averaging
✅ Compatible with CUDA and AMP
"""

import os
import yaml
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset

# Model import
from src.models.bcnn_lstm import BCNN_LSTM


# -----------------------------
# Utility functions
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    return {"precision": p, "recall": r, "f1": f1, "acc": acc}


# -----------------------------
# ELBO + Label Smoothing
# -----------------------------
def elbo_loss(model, probs_avg, targets, num_train_samples, kl_factor=1.0, class_weights=None, label_smooth=0.05):
    device = targets.device
    eps = 1e-9
    probs = torch.clamp(probs_avg, eps, 1.0)

    # Label smoothing
    n_classes = probs.shape[1]
    smooth_target = torch.zeros_like(probs).fill_(label_smooth / (n_classes - 1))
    smooth_target.scatter_(1, targets.unsqueeze(1), 1.0 - label_smooth)

    log_p_y = torch.log(probs)
    nll_per_sample = -(smooth_target * log_p_y).sum(dim=1)

    if class_weights is not None:
        w = class_weights[targets]
        nll_per_sample = nll_per_sample * w

    nll_sum = nll_per_sample.sum()

    # Bayesian KL terms
    log_prior, log_var_post = 0.0, 0.0
    for m in model.modules():
        if hasattr(m, "log_prior"):
            log_prior += m.log_prior()
        if hasattr(m, "log_variational_posterior"):
            log_var_post += m.log_variational_posterior()

    kl_term = (log_var_post - log_prior) / float(num_train_samples)
    loss_sum = nll_sum + kl_factor * kl_term * targets.shape[0]
    return loss_sum / targets.shape[0]


# -----------------------------
# Dataset Wrapper
# -----------------------------
def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    return xs, ys


def make_tensor_dataset(X, y):
    X = np.expand_dims(X.astype(np.float32), axis=1)  # (n,1,channels,time)
    y = y.astype(np.int64)
    return torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


# -----------------------------
# Main Training Function
# -----------------------------
def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg["train"].get("seed", 42)))

    subj = cfg["data"].get("subject", "B")
    data_dir = cfg["data"].get("processed_dir", "data/processed")
    train_X = os.path.join(data_dir, f"Subject_{subj}/train_epochs/X_balanced.npy")
    train_y = os.path.join(data_dir, f"Subject_{subj}/train_epochs/y_balanced.npy")

    X = np.load(train_X)
    y = np.load(train_y)

    # Print imbalance
    unique, counts = np.unique(y, return_counts=True)
    print("Class counts:", dict(zip(unique, counts)))

    # Compute weights
    total = len(y)
    c0, c1 = counts[0], counts[1]
    class_weights_np = total / (2.0 * np.array([c0, c1]))
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)

    samples_weight = np.array([class_weights_np[int(lbl)] for lbl in y], dtype=np.float32)
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

    # Train-val split
    val_split = float(cfg["data"].get("val_split", 0.15))
    n = len(X)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int((1 - val_split) * n)
    train_idx, val_idx = idx[:split], idx[split:]

    train_ds = make_tensor_dataset(X[train_idx], y[train_idx])
    val_ds = make_tensor_dataset(X[val_idx], y[val_idx])

    # Critical fix — use sampler only on train subset
    train_sampler = WeightedRandomSampler(
        weights=samples_weight[train_idx], num_samples=len(train_idx), replacement=True
    )

    batch_size = int(cfg["train"].get("batch_size_per_step", 16))
    accum_steps = int(cfg["train"].get("accumulate_steps", 1))
    n_workers = int(cfg["train"].get("num_workers", 0))

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                              num_workers=n_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)

    # Model
    model_cfg = cfg["model"]
    model = BCNN_LSTM(
        n_channels=X.shape[1], n_timepoints=X.shape[2],
        spatial_kernels=model_cfg.get("spatial_kernels", 10),
        temporal_kernels=model_cfg.get("temporal_kernels", 50),
        temporal_kernel_width=model_cfg.get("temporal_kernel_width", 13),
        temporal_stride=model_cfg.get("temporal_stride", 13),
        lstm_hidden=model_cfg.get("lstm_hidden", 128),
        lstm_layers=model_cfg.get("lstm_layers", 1),
        bidirectional=model_cfg.get("bidirectional", True),
        bayes_fc_hidden=model_cfg.get("bayes_fc_hidden", 100),
        prior_sigma1=cfg["bayes"].get("prior_sigma1", 0.1),
        prior_sigma2=cfg["bayes"].get("prior_sigma2", 0.0005),
        prior_pi=cfg["bayes"].get("prior_pi", 0.5),
        init_std=cfg["bayes"].get("init_std", 0.05),
        dropout=cfg["model"].get("dropout", 0.3),
    ).to(device)

    # Optimizer, Scheduler, AMP
    lr = float(cfg["train"].get("lr", 1e-4))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    use_amp = bool(cfg["train"].get("fp16", True))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    epochs = int(cfg["train"].get("epochs", 60))
    n_mc_train = int(cfg["train"].get("n_mc_train", 1))
    n_mc_val = int(cfg["train"].get("n_mc_test_debug", 20))

    num_train_samples = len(train_ds)
    best_val_f1 = 0.0
    save_dir = cfg.get("save_dir", "results_B")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Training on {device} | Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"Class weights: {class_weights_np.tolist()}")

    # -----------------------------
    # Epoch loop
    # -----------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        kl_factor = min(1.0, epoch / 300.0)
        epoch_loss, preds_all, trues_all = 0.0, [], []

        optimizer.zero_grad()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100, desc=f"Epoch {epoch}/{epochs}")

        for step, (xb, yb) in pbar:
            xb, yb = xb.to(device), yb.to(device)
            xb = xb + 0.002 * torch.randn_like(xb)  # tiny noise

            with torch.cuda.amp.autocast(enabled=use_amp):
                probs_mc = None
                for _ in range(n_mc_train):
                    logits = model(xb)
                    probs = torch.softmax(logits, dim=1)
                    probs_mc = probs if probs_mc is None else probs_mc + probs
                probs_avg = probs_mc / float(n_mc_train)

                loss = elbo_loss(model, probs_avg, yb, num_train_samples, kl_factor,
                                 class_weights=class_weights, label_smooth=0.05)

            loss = loss / accum_steps
            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * xb.size(0) * accum_steps
            preds_all.append(probs_avg.argmax(dim=1).detach().cpu().numpy())
            trues_all.append(yb.detach().cpu().numpy())
            pbar.set_postfix(loss=f"{epoch_loss / ((step + 1) * batch_size):.4f}")

        preds_all = np.concatenate(preds_all)
        trues_all = np.concatenate(trues_all)
        train_metrics = compute_metrics(trues_all, preds_all)
        print(f"Epoch {epoch} train: f1={train_metrics['f1']:.4f} loss={epoch_loss / len(train_ds):.4f}")

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                probs = model.predict_proba_mc(xb, n_samples=n_mc_val, device=device).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                val_preds.append(preds)
                val_trues.append(yb.numpy())

        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)
        val_metrics = compute_metrics(val_trues, val_preds)
        print(f"Epoch {epoch} val: f1={val_metrics['f1']:.4f} acc={val_metrics['acc']:.4f}")

        scheduler.step(val_metrics["f1"])

        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_epoch{epoch}.pt"))
            print(f"✅ Saved best model (epoch {epoch}, f1={best_val_f1:.4f})")

    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pt"))
    print("🏁 Training complete — final model saved.")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/lstm.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train(cfg)
