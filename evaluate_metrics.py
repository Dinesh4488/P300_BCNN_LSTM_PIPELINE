import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# ✅ Import your BCNN_LSTM model class
from src.models.bcnn_lstm import BCNN_LSTM


def evaluate(cfg, subject):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Use TRAIN data (since test set has no labels)
    X_path = f"data/processed/Subject_{subject}/train_epochs/X_balanced.npy"
    y_path = f"data/processed/Subject_{subject}/train_epochs/y_balanced.npy"

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing train files for Subject {subject}")

    X = np.load(X_path).astype(np.float32)
    y_true = np.load(y_path).astype(int)

    print(f"Loaded X: {X.shape}")
    print(f"Loaded y_true: {y_true.shape}")

    # Ensure correct shape: (samples, 1, channels, timepoints)
    if X.ndim == 3:
        X = np.expand_dims(X, axis=1)

    # Build model (adjust based on your yaml config)
    n_channels = X.shape[2]
    n_timepoints = X.shape[3]
    model = BCNN_LSTM(n_channels=n_channels, n_timepoints=n_timepoints).to(device)

    # Load trained weights
    model_path = "results_A/final_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Model loaded successfully.")

    # Forward pass to get predictions
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        outputs = model(X_tensor)
        probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        y_pred = (probs > 0.5).astype(int)

    # --- Metrics ---
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    reco = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print("\n📊 --- Evaluation Metrics ---")
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Reco (Accuracy): {reco:.4f}")
    print(f"confusion matrix:{cm}")
    # Optionally save metrics
    results = {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Recall": recall, "Precision": precision,
        "F1": f1, "Reco": reco ,"confusion_matrix":cm
    }
    os.makedirs("results_A/metrics", exist_ok=True)
    np.save(f"results_A/metrics/metrics_subject_{subject}.npy", results)
    print(f"💾 Metrics saved to results_A/metrics/metrics_subject_{subject}.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/lstm.yaml", help="Config file path")
    parser.add_argument("--subject", type=str, default="A", help="Subject ID (A/B)")
    args = parser.parse_args()

    # Dummy cfg placeholder (you can extend if needed)
    cfg = {}
    evaluate(cfg, args.subject)
