import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample

def extract_epochs(mat_path, out_dir, fs=240, tmin=0.0, tmax=0.65, downsample_factor=2, has_labels=True):
    """Extract epochs from .mat EEG file (for both train and test)."""
    print(f"\nExtracting epochs from {os.path.basename(mat_path)} ...")
    mat = loadmat(mat_path)
    sig = np.asarray(mat["Signal"])
    stim_code = np.asarray(mat["StimulusCode"])
    flashing = np.asarray(mat["Flashing"])
    stim_type = np.asarray(mat.get("StimulusType", np.zeros_like(stim_code)))

    if sig.ndim == 3:
        n_trials, n_samp, n_ch = sig.shape
        sig = sig.transpose(0, 2, 1).reshape(n_trials * n_samp, n_ch).T
        flashing = flashing.reshape(-1)
        stim_code = stim_code.reshape(-1)
        stim_type = stim_type.reshape(-1)
    else:
        raise ValueError("Unexpected Signal shape")

    onsets = np.where(np.diff(flashing.squeeze()) == 1)[0] + 1
    epoch_samp = int(fs * (tmax - tmin))
    X, y = [], []
    for s in onsets:
        e = s + epoch_samp
        if e > sig.shape[1]:
            break
        epoch = sig[:, s:e]
        epoch = resample(epoch, epoch_samp // downsample_factor, axis=1)
        epoch = (epoch - epoch.mean(axis=1, keepdims=True)) / (epoch.std(axis=1, keepdims=True) + 1e-9)
        X.append(epoch)
        if has_labels:
            label = int(stim_type[s]) if s < len(stim_type) else 0
            y.append(label)

    X = np.stack(X)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), X)

    if has_labels:
        y = np.array(y, dtype=int)
        np.save(os.path.join(out_dir, "y.npy"), y)
        print(f"✅ Saved {X.shape}, Targets: {np.sum(y==1)}, Non-targets: {np.sum(y==0)}")
    else:
        print(f"✅ Saved test epochs {X.shape} (no labels)")

if __name__ == "__main__":
    subjects = ["A", "B"]
    for subj in subjects:
        # --- Training data ---
        train_path = f"data/Subject_{subj}_Train.mat"
        if os.path.exists(train_path):
            out_dir = f"data/processed/Subject_{subj}/train_epochs"
            extract_epochs(train_path, out_dir, has_labels=True)
        else:
            print(f"⚠️ Missing: {train_path}")

        # --- Test data ---
        test_path = f"data/Subject_{subj}_Test.mat"
        if os.path.exists(test_path):
            out_dir = f"data/processed/Subject_{subj}/test_epochs"
            extract_epochs(test_path, out_dir, has_labels=False)
        else:
            print(f"⚠️ Missing: {test_path}")

    print("\n🎯 Epoch extraction complete.")
