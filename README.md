**P300 BCNN-LSTM (Bayesian)**

This repository contains code to reproduce a Bayesian BCNN + LSTM pipeline for P300-like EEG character recognition. It includes preprocessing, epoch extraction, model training, MC inference, and evaluation scripts. Dataset files and large result/model binaries are intentionally excluded from the repo (see "Excluded Files" below).

**Dataset**
- **Name**: P300 dataset (Subjects A & B) — BCI Competition III, 2004
- **Download**: https://www.bbci.de/competition/download/competition_iii/albany/BCI_Comp_III_Wads_2004.zip
- After downloading, extract the `.mat` files and place them into the `data/` directory (do NOT commit raw data to the repository).

**Quick Start**
- **Python env**: Create and activate a virtualenv, then install requirements: `python -m venv venv` then `venv\Scripts\Activate.ps1` and `pip install -r requirements.txt`.
- **Place data**: Download dataset externally and copy the `.mat` files into `data/` (do NOT commit them).
- **Run pipeline (order)**: `python preprocess.py` → `python epoch_extraction.py` → `python balance_epochs.py` → `python train.py` (or `python train_B.py`) → `python predict_test.py` → `python evaluate_metrics.py` → `python results_report.py`.

**Suggested Commands (PowerShell)**
- Create env & install: `python -m venv venv; .\venv\Scripts\Activate.ps1; pip install -r requirements.txt`
- Preprocess & epochs: `python preprocess.py; python epoch_extraction.py; python balance_epochs.py`
- Train (Subject A): `python train.py --config configs/lstm.yaml`
- Train (Subject B): `python train_B.py --config configs/lstm.yaml`
- Predict & evaluate: `python predict_test.py; python evaluate_metrics.py; python results_report.py`

**Dependencies**

Install all required packages:
```powershell
pip install -r requirements.txt
```

Core dependencies: `torch`, `numpy`, `scipy`, `scikit-learn`, `tqdm`, `PyYAML`, `matplotlib`, `seaborn`

**Included Files (Required for Pipeline)**
- **Training**: `train.py`, `train_B.py`
- **Data Pipeline**: `preprocess.py`, `epoch_extraction.py`, `balance_epochs.py`
- **Inference & Reporting**: `predict_test.py`, `evaluate_metrics.py`, `results_report.py`
- **Model Code**: `src/models/bcnn_lstm.py`, `src/models/bayesian_layers.py`, `src/utils/config.py`
- **Configuration**: `configs/lstm.yaml`
- **Documentation**: `README.md`, `requirements.txt`

**Excluded from Git (Not Uploaded)**
- `data/` — raw `.mat` files and `data/processed/` (download dataset externally)
- `venv/` — virtual environment (create locally)
- `.gitignore` configured to exclude large binaries automatically

**Stored Results (what we keep in this repo)**
- This repository keeps non-regenerable summary outputs that are small and useful for quick inspection: CSV summaries (`summary_metrics_subject_*.csv`), plots (`*.png`, `*.svg`) and other small analysis artifacts under `results_A/analysis/` and `results_B/analysis/`.

**What we do NOT commit**
- Regenerable outputs (checkpoints, large numpy arrays, and processed datasets) are intentionally ignored because they can be recreated by running the pipeline locally. Examples:
	- model checkpoints: `*.pt`
	- arrays: `*.npy`, `data/processed/**`
	- raw dataset files: `data/*.mat`

This keeps the repository small and avoids pushing large binaries. If you need full checkpoints or processed arrays, download them from the provided external links (or generate them by running the pipeline).

**Results Summary**

Trained on BCI Competition III P300 dataset (2004). Reference test-set metrics are included below; full predictions, analysis plots, and checkpoints are available in `results_A/` and `results_B/`.

**Subject A** (22,949 test samples)

| Metric       | Value   |
|--------------|---------|
| Accuracy     | 86.94%  |
| Precision    | 85.57%  |
| Recall       | 84.94%  |
| F1-Score     | 85.25%  |
| Specificity  | 88.54%  |
| MCC          | 0.7354  |

**Subject B** (22,946 test samples)

| Metric       | Value   |
|--------------|---------|
| Accuracy     | 74.73%  |
| Precision    | 66.50%  |
| Recall       | 86.90%  |
| F1-Score     | 75.34%  |
| Specificity  | 64.99%  |
| MCC          | 0.5225  |

Confusion matrices, prediction histograms, and raw prediction arrays are included under `results_A/analysis/` and `results_B/analysis/` in this repository. If you prefer smaller clones, use the instructions below to avoid downloading large artifacts.

**Notes & Reproducibility**
- The repo is configured to keep large binaries out of version control. Before initializing any new Git repo, ensure `README.md` and `.gitignore` are present so `data/`, `results/`, and `venv/` remain untracked.
 - If you want to push large model files to a remote, use Git LFS or external storage instead.

the project owner. For questions or help reproducing results, open an issue or contact the maintainer.
