import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample, iirnotch

# ---------- Filters ----------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return b, a

def bandpass_filter(data, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data, axis=-1)

def notch_filter(data, notch_freq, fs, q=30):
    b, a = iirnotch(notch_freq, q, fs)
    return filtfilt(b, a, data, axis=-1)

# ---------- Main processing ----------
def process_subject(mat_path, out_dir, fs=240, downsample_factor=2, notch_freq=50):
    print(f"\nProcessing {os.path.basename(mat_path)} ...")
    mat = loadmat(mat_path)
    signal = np.asarray(mat["Signal"])  # (trials, channels, samples)

    has_labels = "StimulusType" in mat
    if has_labels:
        stimulus_type = np.asarray(mat["StimulusType"]).astype(int).flatten()
    else:
        stimulus_type = np.zeros(signal.shape[0], dtype=int)
        print("⚠️ No StimulusType found — likely a TEST file without labels.")

    n_trials, n_channels, n_samples = signal.shape
    filtered = np.zeros((n_trials, n_channels, n_samples // downsample_factor), dtype=np.float32)

    for i in range(n_trials):
        for ch in range(n_channels):
            x = signal[i, ch, :]
            x = bandpass_filter(x, 0.1, 20, fs)
            x = notch_filter(x, notch_freq, fs)
            filtered[i, ch, :] = resample(x, n_samples // downsample_factor)
        # Normalize per trial
        filtered[i] = (filtered[i] - filtered[i].mean(axis=1, keepdims=True)) / (filtered[i].std(axis=1, keepdims=True) + 1e-9)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), filtered)
    if has_labels:
        np.save(os.path.join(out_dir, "y.npy"), stimulus_type)
        print(f"✅ Saved {filtered.shape}, labels: {np.bincount(stimulus_type)}")
    else:
        print(f"✅ Saved {filtered.shape} (no labels)")

if __name__ == "__main__":
    subjects = ["A", "B"]
    for subj in subjects:
        for split in ["Train", "Test"]:
            fname = f"data/Subject_{subj}_{split}.mat"
            if not os.path.exists(fname):
                print(f"File missing: {fname}")
                continue
            out_dir = f"data/processed/Subject_{subj}/{split.lower()}"
            process_subject(fname, out_dir)
    print("\n🎯 Preprocessing complete.")
