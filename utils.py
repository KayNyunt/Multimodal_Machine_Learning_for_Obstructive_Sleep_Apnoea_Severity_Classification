import re
import numpy as np
import pandas as pd
import mne

import config

# CHANNEL RESOLUTION
def resolve_channel(raw, candidates):
    def norm(s):
        return s.lower().replace(" ", "").replace("-", "").replace("_", "")

    # Tier 1: exact
    for name in candidates:
        if name in raw.ch_names:
            return name

    # Tier 2: case-insensitive
    lower_map = {ch.lower(): ch for ch in raw.ch_names}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    # Tier 3: normalised substring
    for name in candidates:
        for ch in raw.ch_names:
            if norm(name) in norm(ch):
                return ch

    return None


def subject_id_from_path(path):
    """Extract the numeric participant ID from a filename.
    """
    m = re.search(r"(\d+)(?=\D*$)", str(path.stem))
    return int(m.group(1)) if m else None


def load_edf_channels(edf_path, candidate_map):
    """Open a recording, resolve the requested modalities, load only those.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

    resolved = {}
    for key, candidates in candidate_map.items():
        ch = resolve_channel(raw, candidates)
        if ch is not None:
            resolved[key] = ch

    if not resolved:
        raise ValueError(f"No requested channels found in {edf_path.name}")

    raw.pick(list(dict.fromkeys(resolved.values())))
    raw.load_data()
    return raw, resolved

# SIGNAL QUALITY CONTROL
def mask_invalid_spo2(spo2, low=50, high=100):
    """Replace physiologically impossible oximetry values with NaN.
    """
    return np.where((spo2 >= low) & (spo2 <= high), spo2, np.nan)

def find_valid_window(signal, sfreq, validity_fn, window_sec=60,
                      min_frac=0.5, max_gap_sec=600):
    """Find the longest stretch where the signal is mostly valid.
    """
    win = int(window_sec * sfreq)
    n_win = len(signal) // win
    if n_win == 0:
        return 0.0, len(signal) / sfreq

    good = np.array([
        validity_fn(signal[i * win:(i + 1) * win]).mean() > min_frac
        for i in range(n_win)
    ])

    if not good.any():
        return 0.0, len(signal) / sfreq

    max_gap_windows = int(max_gap_sec / window_sec)
    bridged = good.copy()
    gap_start = None
    for i, ok in enumerate(good):
        if not ok:
            if gap_start is None:
                gap_start = i
        else:
            if gap_start is not None:
                gap_len = i - gap_start
                if gap_len <= max_gap_windows:
                    bridged[gap_start:i] = True
                gap_start = None

    good = bridged

    # Longest contiguous run of valid windows
    best_start = best_len = cur_start = cur_len = 0
    for i, ok in enumerate(good):
        if ok:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
        else:
            cur_len = 0

    return (best_start * window_sec,
            min((best_start + best_len) * window_sec, len(signal) / sfreq))


def filter_rr_intervals(nn_ms, low=300, high=2000):
    """Remove implausible intervals between heartbeats.
       The percentage is retained as a data-quality indicator.
    """
    mask = (nn_ms >= low) & (nn_ms <= high)
    return nn_ms[mask], (1 - mask.mean()) * 100

# CLINICAL DATA
def classify_severity(ahi):
    """Convert a continuous index into a severity category.
    """
    if pd.isna(ahi):
        return None
    if ahi < config.AHI_THRESHOLDS[0]:
        return "Normal"
    if ahi < config.AHI_THRESHOLDS[1]:
        return "Mild"
    if ahi < config.AHI_THRESHOLDS[2]:
        return "Moderate"
    return "Severe"


def load_clinical():
    """Load the clinical dataset, rename columns, derive severity labels.
    The NSRR variable nsrr_ahi_hp4u_aasm15 is used, scored under the
    AASM recommended criterion (30% airflow reduction with at least 4%
    desaturation). Participants with no index are dropped, since they
    cannot be labelled.
    """
    df = pd.read_csv(config.CLINICAL_CSV).rename(columns=config.CLINICAL_RENAME)
    df["Severity"] = df["AHI"].apply(classify_severity)
    df = df.dropna(subset=["Severity"]).copy()
    df["Severity"] = df["Severity"].map(config.SEVERITY_ORDER)
    return df


def build_cohort(features_csv, cohort_name, verbose=True):
    """Merge clinical data with extracted features for one cohort.
    """
    clinical = load_clinical()
    signal = pd.read_csv(features_csv)
    merged = clinical.merge(signal, on="nsrrid", how="inner")

    feats = [c for c in signal.columns
             if c not in config.LEAKAGE_COLS
             and c not in config.QC_DIAGNOSTIC_COLS
             and c != "recording_setting"]
    feats = [c for c in feats if pd.api.types.is_numeric_dtype(merged[c])]

    if verbose:
        print(f"{cohort_name}: {len(merged)} participants, "
              f"{len(feats)} signal features")
        dist = merged["Severity"].value_counts().sort_index()
        named = {config.LABEL_NAMES[k]: v for k, v in dist.items()}
        print(f"  severity: {named}")

    return merged, feats