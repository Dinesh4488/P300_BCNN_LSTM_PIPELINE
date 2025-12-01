"""
results_report.py

Generates aggregated metrics, tables, and figures for Subject A and Subject B
without modifying any existing files. The script searches the repository for
existing prediction/probability files and previously-saved metrics, then
produces:

- CSV summary files with TP/TN/FP/FN, precision, recall, F1, accuracy, specificity, NPV, MCC
- Confusion matrix heatmaps (PNG)
- Metric bar charts (PNG)
- Probability histograms for the positive class (PNG)
- Exports saved under `results/analysis/` and `results_B/analysis/`

Usage:
    Activate the project's virtualenv (if any) and run:

    "C:/Users/Dinesh karthik/Desktop/p300 bcnn/venv/Scripts/python.exe" results_report.py

Dependencies (install if missing): numpy, pandas, matplotlib, seaborn, scikit-learn

Notes:
- The repository already contains `metrics_subject_A.npy` and `metrics_subject_B.npy` that
  include confusion matrices and some metrics. This script will use those where available.
- If per-sample ground-truth labels for the test set are not present, ROC and PR curves
  cannot be computed; the script will explain that and still produce other visualizations.

This script intentionally does not modify any existing files in the repo.
"""

from pathlib import Path
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
try:
    import importlib
    sns = importlib.import_module("seaborn")
    _HAS_SEABORN = True
    try:
        sns.set(style="whitegrid")
    except Exception:
        # if seaborn is present but setting style fails, ignore and continue
        pass
except Exception:
    sns = None
    _HAS_SEABORN = False

from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score,
                             confusion_matrix, matthews_corrcoef)

BASE = Path(__file__).resolve().parent
SUBJECTS = {
    'A': {
        'root': BASE / 'results_A',
        'preds': BASE / 'results_A' / 'predictions' / 'preds.npy',
        'probs': BASE / 'results_A' / 'predictions' / 'probs.npy',
        'char_preds': BASE / 'results_A' / 'predictions' / 'char_preds_subject_A.npy',
        'metrics': BASE / 'results_A' / 'metrics' / 'metrics_subject_A.npy',
        'out': BASE / 'results_A' / 'analysis'
    },
    'B': {
        'root': BASE / 'results_B',
        'preds': BASE / 'results_B' / 'predictions' / 'preds.npy',
        'probs': BASE / 'results_B' / 'predictions' / 'probs.npy',
        'char_preds': BASE / 'results_B' / 'predictions' / 'char_preds_subject_B.npy',
        'metrics': BASE / 'results_B' / 'metrics' / 'metrics_subject_B.npy',
        'out': BASE / 'results_B' / 'analysis'
    }
}


def safe_load(path):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return np.load(p, allow_pickle=True)
    except Exception as e:
        print(f"Warning: failed to load {p}: {e}")
        return None


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def derive_metrics_from_confusion(cm):
    # cm expected as array([[TN, FP],[FN, TP]])
    TN, FP = int(cm[0, 0]), int(cm[0, 1])
    FN, TP = int(cm[1, 0]), int(cm[1, 1])
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total else np.nan
    precision = TP / (TP + FP) if (TP + FP) else np.nan
    recall = TP / (TP + FN) if (TP + FN) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else np.nan
    specificity = TN / (TN + FP) if (TN + FP) else np.nan
    npv = TN / (TN + FN) if (TN + FN) else np.nan
    mcc = matthews_corrcoef(
        np.concatenate([np.zeros(TN + FP), np.ones(FN + TP)]),
        np.concatenate([np.zeros(TN), np.ones(FP), np.zeros(FN), np.ones(TP)])
    ) if total else np.nan
    return {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
        'specificity': specificity, 'npv': npv, 'mcc': mcc,
        'total': total
    }


def plot_confusion(cm, out_path: Path, subj_label: str):
    plt.figure(figsize=(5, 4))
    if _HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
    else:
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(int(cm[i, j]), 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.xticks([0, 1], ["Neg", "Pos"])  
        plt.yticks([0, 1], ["Neg", "Pos"])  

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - Subject {subj_label}')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_metric_bars(metrics_dict, out_path: Path, subj_label: str):
    keys = ['precision', 'recall', 'f1', 'accuracy', 'specificity', 'mcc']
    vals = [metrics_dict.get(k, np.nan) for k in keys]
    plt.figure(figsize=(7, 4))
    x = range(len(keys))
    plt.bar(x, [0 if np.isnan(v) else v for v in vals], color='tab:blue')
    plt.xticks(x, keys)
    plt.ylim(0, 1)
    plt.title(f'Key Metrics - Subject {subj_label}')
    for i, v in enumerate(vals):
        lab = f"{v:.3f}" if not np.isnan(v) else 'NA'
        plt.text(i, (v if not np.isnan(v) else 0) + 0.02, lab, ha='center')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_probability_hist(probs, preds, out_path: Path, subj_label: str):
    # probs may be shape (n,2) or (n,) for positive-class prob
    if probs is None:
        return
    if probs.ndim == 2 and probs.shape[1] >= 2:
        pos = probs[:, 1]
    else:
        pos = probs.flatten()
    plt.figure(figsize=(6, 4))
    if _HAS_SEABORN:
        sns.histplot(pos, bins=50, kde=False, color='tab:blue')
    else:
        plt.hist(pos, bins=50, color='tab:blue')
    if preds is not None:
        plt.axvline(0.5, color='k', linestyle='--', label='0.5 thresh')
    plt.title(f'Positive-class probability distribution - Subject {subj_label}')
    plt.xlabel('P(positive)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def try_plot_roc_pr(y_true, probs, out_dir: Path, subj_label: str):
    # returns dict with auc and average precision or None if not possible
    results = {}
    if y_true is None or probs is None:
        print(f"Skipping ROC/PR for Subject {subj_label}: missing y_true or probs")
        return None
    # extract positive probabilities
    if probs.ndim == 2 and probs.shape[1] >=2:
        y_score = probs[:, 1]
    else:
        y_score = probs.flatten()
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Subject {subj_label}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(out_dir / f'roc_subject_{subj_label}.png', dpi=150)
        plt.close()
        results['roc_auc'] = roc_auc
    except Exception as e:
        print(f"Could not compute ROC for Subject {subj_label}: {e}")

    try:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        plt.figure(figsize=(5, 5))
        plt.plot(recall, precision, label=f'AP = {ap:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - Subject {subj_label}')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(out_dir / f'pr_subject_{subj_label}.png', dpi=150)
        plt.close()
        results['average_precision'] = ap
    except Exception as e:
        print(f"Could not compute PR curve for Subject {subj_label}: {e}")

    return results


def analyze_subject(sub_label, conf):
    print(f"\nAnalyzing Subject {sub_label}...")
    out_dir = conf['out']
    ensure_dir(out_dir)

    preds = safe_load(conf['preds'])
    probs = safe_load(conf['probs'])
    char_preds = safe_load(conf['char_preds'])
    metrics_np = safe_load(conf['metrics'])

    # Prepare a metrics dict
    metrics_dict = {}
    if metrics_np is not None:
        # metrics file appears to be a saved dict
        try:
            if isinstance(metrics_np, np.ndarray) and metrics_np.shape == ():
                metrics_dict = metrics_np.tolist()
            elif isinstance(metrics_np, dict):
                metrics_dict = metrics_np
            else:
                # try converting to python object
                metrics_dict = metrics_np.item() if hasattr(metrics_np, 'item') else dict(metrics_np)
        except Exception:
            metrics_dict = {}
    else:
        metrics_dict = {}

    # If confusion matrix available in metrics
    cm = None
    if 'confusion_matrix' in metrics_dict:
        try:
            cm = np.array(metrics_dict['confusion_matrix'])
        except Exception:
            cm = None

    # If no cm but preds and true labels present in metrics, try to build
    if cm is None and 'y_true' in metrics_dict and preds is not None:
        try:
            cm = confusion_matrix(metrics_dict['y_true'], preds)
        except Exception:
            cm = None

    # Derived metrics
    derived = {}
    if cm is not None:
        derived = derive_metrics_from_confusion(cm)
    else:
        # try to compute simple metrics if y_true exists
        if 'y_true' in metrics_dict and preds is not None:
            y_true = np.array(metrics_dict['y_true']).ravel()
            cm2 = confusion_matrix(y_true, preds)
            derived = derive_metrics_from_confusion(cm2)
            cm = cm2
        else:
            # fallback: try to use already-saved scalar metrics
            scalar_keys = ['TP','TN','FP','FN','Precision','Recall','F1','precision','recall','f1']
            for k in scalar_keys:
                if k in metrics_dict:
                    derived[k.lower()] = float(metrics_dict[k]) if not isinstance(metrics_dict[k], (np.integer, np.ndarray)) else int(metrics_dict[k])

    # Compute totals if possible
    if 'total' not in derived:
        try:
            total = int(derived.get('TP', 0) + derived.get('TN', 0) + derived.get('FP', 0) + derived.get('FN', 0))
            derived['total'] = total
        except Exception:
            pass

    # Try to get per-sample y_true if present anywhere in metrics file
    y_true = None
    if 'y_true' in metrics_dict:
        y_true = np.array(metrics_dict['y_true']).ravel()

    # Save CSV summary
    summary = derived.copy()
    # Also include saved scalar metrics with clear key names
    for k in ['Precision', 'Recall', 'F1', 'Reco']:
        if k in metrics_dict:
            summary[k.lower()] = float(metrics_dict[k])

    # Save summary CSV using csv module to avoid pandas dependency
    summary_csv = out_dir / f'summary_metrics_subject_{sub_label}.csv'
    try:
        with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            for k, v in summary.items():
                writer.writerow([k, v])
        print(f"Saved summary CSV to {summary_csv}")
    except Exception as e:
        print(f"Warning: could not write summary CSV: {e}")

    # Figures
    if cm is not None:
        plot_confusion(cm, out_dir / f'confusion_subject_{sub_label}.png', sub_label)
        print(f"Saved confusion matrix to {out_dir / f'confusion_subject_{sub_label}.png'}")
    else:
        print(f"No confusion matrix available for Subject {sub_label}")

    plot_metric_bars(derived, out_dir / f'metrics_bar_subject_{sub_label}.png', sub_label)
    print(f"Saved metric bar chart to {out_dir / f'metrics_bar_subject_{sub_label}.png'}")

    if probs is not None:
        plot_probability_hist(probs, preds, out_dir / f'prob_hist_subject_{sub_label}.png', sub_label)
        print(f"Saved probability histogram to {out_dir / f'prob_hist_subject_{sub_label}.png'}")

    # ROC/PR if possible
    rocpr = try_plot_roc_pr(y_true, probs, out_dir, sub_label)
    if rocpr is not None:
        # save ROC/PR metrics to summary
        # append ROC/PR metrics to CSV
        try:
            with open(summary_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for k, v in rocpr.items():
                    writer.writerow([k, v])
            print(f"Updated summary CSV with ROC/PR metrics: {summary_csv}")
        except Exception as e:
            print(f"Warning: could not append ROC/PR metrics to CSV: {e}")
    else:
        print(f"ROC/PR not produced for Subject {sub_label} (missing per-sample true labels).")

    # Save raw preds/probs/char_preds copies into out_dir for convenient bundling (read-only copy)
    try:
        if preds is not None:
            np.save(out_dir / f'preds_subject_{sub_label}.npy', preds)
        if probs is not None:
            np.save(out_dir / f'probs_subject_{sub_label}.npy', probs)
        if char_preds is not None:
            np.save(out_dir / f'char_preds_subject_{sub_label}.npy', char_preds)
    except Exception as e:
        print(f"Warning: could not save copies of raw arrays: {e}")

    print(f"Completed analysis for Subject {sub_label}. Outputs in: {out_dir}\n")


def main():
    print("Starting results aggregation for Subject A and B...")
    for s, conf in SUBJECTS.items():
        analyze_subject(s, conf)
    print("All done. You can find CSV summaries and figures under results/analysis and results_B/analysis.")


if __name__ == '__main__':
    main()
