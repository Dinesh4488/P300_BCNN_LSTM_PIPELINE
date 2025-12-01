import os
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, precision_score, recall_score

from src.models.bcnn_lstm import BCNN_LSTM  # same model as in train.py

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": acc,
    }

# ------------------------------------------------------------
# Main prediction logic
# ------------------------------------------------------------
def predict(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subj = cfg.get("subject", "A")
    data_dir = cfg.get("data_dir", "data/processed")
    model_path = cfg.get("model_path", "results_A/final_model.pt")
    save_dir = cfg.get("save_dir", f"results/predictions_subject_{subj}")
    os.makedirs(save_dir, exist_ok=True)

    test_X_path = os.path.join(data_dir, f"Subject_{subj}/test_epochs/X.npy")
    test_y_path = os.path.join(data_dir, f"subject_{subj}/test/y.npy")  # optional

    print(f"🔹 Loading test data from {test_X_path}")
    X_test = np.load(test_X_path).astype(np.float32)
    X_test = np.expand_dims(X_test, axis=1)  # (n,1,channels,times)

    has_labels = os.path.exists(test_y_path)
    y_test = np.load(test_y_path).astype(np.int64) if has_labels else None
    print(f"Test shape: {X_test.shape}, Labels available: {has_labels}")

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
    print(f"🔹 Loading model from {model_path}")
    # We infer shape from X_test if needed
    n_channels = X_test.shape[2]
    n_times = X_test.shape[3]

    model = BCNN_LSTM(
        n_channels=n_channels,
        n_timepoints=n_times,
        spatial_kernels=10,
        temporal_kernels=50,
        temporal_kernel_width=13,
        temporal_stride=13,
        lstm_hidden=128,
        lstm_layers=1,
        bidirectional=True,
        bayes_fc_hidden=100,
        prior_sigma1=0.1,
        prior_sigma2=0.0005,
        prior_pi=0.5,
        init_std=0.05,
        dropout=0.5,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Model loaded successfully.")

    # ------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------
    batch_size = 64
    all_probs, all_preds = [], []

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            xb = torch.from_numpy(X_test[i:i+batch_size]).to(device)
            probs = model.predict_proba_mc(xb, n_samples=20, device=device).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_probs.append(probs)
            all_preds.append(preds)

    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)

    np.save(os.path.join(save_dir, "probs.npy"), all_probs)
    np.save(os.path.join(save_dir, "preds.npy"), all_preds)
    print(f"💾 Saved predictions to {save_dir}")

    # ------------------------------------------------------------
    # If ground-truth labels exist, compute metrics
    # ------------------------------------------------------------
    if has_labels:
        metrics = compute_metrics(y_test, all_preds)
        print("\n📊 Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k:>10}: {v:.4f}" if isinstance(v, float) else f"{k:>10}: {v}")
    else:
        print("⚠️ No ground-truth labels found — only predictions saved.")

    print("✅ Prediction complete.")


if __name__ == "__main__":
    # you can edit these or load from a yaml config
    cfg = {
        "subject": "A",                     # or "B"
        "data_dir": "data/processed",
        "model_path": "results_B/final_model.pt",
        "save_dir": "results_B/predictions",
    }
    predict(cfg)
