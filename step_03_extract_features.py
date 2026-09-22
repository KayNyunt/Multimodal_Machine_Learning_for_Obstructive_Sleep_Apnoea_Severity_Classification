import warnings

import numpy as np
import pandas as pd
import neurokit2 as nk
from tqdm import tqdm

import config
from utils import (load_edf_channels, subject_id_from_path,
                   mask_invalid_spo2, find_valid_window, filter_rr_intervals)

warnings.filterwarnings("ignore")

# OXIMETRY
def spo2_features(spo2, sfreq):
    valid = mask_invalid_spo2(spo2)
    n_valid = np.sum(~np.isnan(valid))
    hours = n_valid / sfreq / 3600

    if hours < 0.5:
        return {}

    f = {
        "spo2_mean": np.nanmean(valid),
        "spo2_std": np.nanstd(valid),
        "spo2_pct_below_90": np.nanmean(valid < 90) * 100,
        "spo2_pct_invalid": (1 - n_valid / len(spo2)) * 100,
    }

    s = pd.Series(valid)
    baseline = s.rolling(int(60 * sfreq), min_periods=1, center=True).max()
    min_samples = int(10 * sfreq)

    for pct in (3, 4):
        below = ((baseline - s) >= pct).fillna(False).values

        count = run = 0
        for is_below in below:
            if is_below:
                run += 1
            else:
                if run >= min_samples:
                    count += 1
                run = 0
        if run >= min_samples:
            count += 1

        f[f"odi_{pct}pct"] = count / hours

    deficit = (baseline - s).clip(lower=0).fillna(0)
    f["hypoxic_burden"] = (deficit.sum() / sfreq / 60) / hours

    return f


# ELECTROCARDIOGRAPHY
def ecg_features(ecg, sfreq):
    try:
        cleaned = nk.ecg_clean(ecg, sampling_rate=sfreq)
        _, info = nk.ecg_peaks(cleaned, sampling_rate=sfreq,
                               correct_artifacts=True)
        r_peaks = info["ECG_R_Peaks"]

        nn = np.diff(r_peaks) / sfreq * 1000
        nn, pct_bad = filter_rr_intervals(nn)

        if len(nn) < 30:
            return {}, None

        dnn = np.diff(nn)
        sdnn = np.std(nn, ddof=1)
        sd_dnn = np.std(dnn, ddof=1)


        sd1 = np.sqrt(sd_dnn ** 2 * 0.5)
        sd2 = np.sqrt(max(2 * sdnn ** 2 - 0.5 * sd_dnn ** 2, 0))

        f = {
            "ecg_rate_mean": 60000 / np.mean(nn),
            "hrv_sdnn": sdnn,
            "hrv_rmssd": np.sqrt(np.mean(dnn ** 2)),
            "hrv_pnn50": np.mean(np.abs(dnn) > 50) * 100,
            "hrv_sd1": sd1,
            "hrv_sd2": sd2,
            "hrv_sd1sd2": sd1 / sd2 if sd2 else np.nan,
            "ecg_pct_invalid_beats": pct_bad,
        }

        freq = nk.hrv_frequency(info, sampling_rate=sfreq).iloc[0].to_dict()
        for k in ("HRV_LF", "HRV_HF", "HRV_LFHF", "HRV_VLF"):
            if k in freq:
                f[k.lower()] = freq[k]

        return f, r_peaks

    except Exception as e:
        print(f"      ECG failed: {e}")
        return {}, None

# AIRFLOW
def airflow_features(airflow, sfreq, prefix="airflow"):
    win = int(10 * sfreq)
    n_win = len(airflow) // win
    if n_win < 30:
        return {}

    windows = [airflow[i * win:(i + 1) * win] for i in range(n_win)]
    amps = np.array([np.ptp(w) for w in windows])
    valid_idx = np.isfinite(amps) & (amps > 0)
    amps = amps[valid_idx]
    if len(amps) < 30:
        return {}

    baseline = np.median(amps)
    hours = (len(amps) * 10) / 3600

    f = {
        f"{prefix}_std": np.nanstd(airflow),
        f"{prefix}_cv": np.nanstd(amps) / np.nanmean(amps),
        f"{prefix}_baseline_amp": baseline,
    }

    for thresh, name in [(0.7, "hypop"), (0.1, "apnea")]:
        below = amps < (thresh * baseline)
        f[f"{prefix}_{name}_pct"] = below.mean() * 100
        f[f"{prefix}_{name}_rate"] = below.sum() / hours

    flat_ratios = []
    for w in np.array(windows, dtype=object)[valid_idx]:
        peak = np.abs(w).max()
        if peak > 0:
            flat_ratios.append(np.abs(w).mean() / peak)
    if flat_ratios:
        f[f"{prefix}_flattening_index"] = float(np.mean(flat_ratios))

    return f

# RESPIRATORY EFFORT
def effort_features(chest, abdomen, sfreq):
    n = min(len(chest), len(abdomen))
    chest, abdomen = chest[:n], abdomen[:n]

    f = {"effort_corr_overall": np.corrcoef(chest, abdomen)[0, 1]}

    win = int(30 * sfreq)
    n_win = n // win
    if n_win >= 10:
        corrs = []
        for i in range(n_win):
            c = chest[i * win:(i + 1) * win]
            a = abdomen[i * win:(i + 1) * win]
            if np.std(c) > 1e-12 and np.std(a) > 1e-12:
                corrs.append(np.corrcoef(c, a)[0, 1])
        if corrs:
            corrs = np.array(corrs)
            f["effort_corr_mean"] = corrs.mean()
            f["effort_corr_std"] = corrs.std()
            f["effort_paradox_pct"] = (corrs < 0).mean() * 100

    return f

# PLETHYSMOGRAPHY
def pleth_features(pleth, sfreq):
    win = int(30 * sfreq)
    n_win = len(pleth) // win
    f = {"pleth_std": np.nanstd(pleth)}

    if n_win >= 10:
        amps = np.array([np.ptp(pleth[i * win:(i + 1) * win])
                         for i in range(n_win)])
        amps = amps[np.isfinite(amps)]
        if len(amps) and np.nanmean(amps) > 0:
            f["pleth_amp_cv"] = np.nanstd(amps) / np.nanmean(amps)
            f["pleth_amp_drop_pct"] = (amps < 0.7 * np.median(amps)).mean() * 100

    return f

# PULSE TRANSIT TIME
def ptt_features(ecg_r_peaks, pleth, sfreq):
    try:
        cleaned_pleth = nk.ppg_clean(pleth, sampling_rate=sfreq)
        _, ppg_info = nk.ppg_peaks(cleaned_pleth, sampling_rate=sfreq)
        ppg_peaks = ppg_info["PPG_Peaks"]

        if len(ecg_r_peaks) < 10 or len(ppg_peaks) < 10:
            return {}

        ptts = []
        for r in ecg_r_peaks:
            subsequent = ppg_peaks[ppg_peaks > r]
            if len(subsequent) > 0:
                next_ppg = subsequent[0]
                diff_ms = (next_ppg - r) / sfreq * 1000
                if 100 <= diff_ms <= 500:
                    ptts.append(diff_ms)

        ptts = np.array(ptts)
        if len(ptts) < 30:
            return {}

        median_ptt = np.nanmedian(ptts)
        drops = (ptts < 0.85 * median_ptt).mean() * 100

        return {
            "ptt_mean": float(np.nanmean(ptts)),
            "ptt_sd": float(np.nanstd(ptts, ddof=1)),
            "ptt_drop_pct": float(drops),
        }
    except Exception as e:
        print(f"      PTT failed: {e}")
        return {}

# ONE PARTICIPANT
def process_participant(edf_path, setting):
    """Extract every feature for one recording."""
    nsrrid = subject_id_from_path(edf_path)
    out = {"nsrrid": nsrrid, "recording_setting": setting}

    raw, resolved = load_edf_channels(edf_path, config.MODALITIES)
    sfreq = raw.info["sfreq"]

    start_sec, end_sec = 0.0, raw.n_times / sfreq
    if "spo2" in resolved:
        spo2_full = raw.get_data(picks=[resolved["spo2"]])[0]
        start_sec, end_sec = find_valid_window(
            spo2_full, sfreq,
            validity_fn=lambda s: (s >= 50) & (s <= 100))

    out["valid_window_hours"] = (end_sec - start_sec) / 3600

    def get(key):
        """Return one channel, trimmed to the valid window."""
        if key not in resolved:
            return None
        sig = raw.get_data(picks=[resolved[key]])[0]
        return sig[int(start_sec * sfreq):int(end_sec * sfreq)]

    if (s := get("spo2")) is not None:
        out.update(spo2_features(s, sfreq))

    ecg_r_peaks = None
    if (s := get("ecg")) is not None:
        ecg_result, ecg_r_peaks = ecg_features(s, sfreq)
        out.update(ecg_result)

    if (s := get("airflow")) is not None:
        out.update(airflow_features(s, sfreq))
    if (c := get("chest")) is not None and (a := get("abdomen")) is not None:
        out.update(effort_features(c, a, sfreq))

    pleth_sig = get("pleth")
    if pleth_sig is not None:
        out.update(pleth_features(pleth_sig, sfreq))
        if ecg_r_peaks is not None:
            out.update(ptt_features(ecg_r_peaks, pleth_sig, sfreq))

    return out


# ONE COHORT
def main(cohort):
    if cohort not in ("lab", "home"):
        raise ValueError(f'cohort must be "lab" or "home", got {cohort!r}')

    edf_dir = config.EDF_LAB_FULL if cohort == "lab" else config.EDF_HOME
    out_csv = (config.LAB_FEATURES_CSV if cohort == "lab"
               else config.HOME_FEATURES_CSV)

    files = sorted(edf_dir.rglob("*.edf"))
    print(f"Cohort: {cohort}")
    print(f"Found {len(files)} recordings in {edf_dir}\n")

    rows, failed = [], []
    for i, path in enumerate(tqdm(files, desc=f"Extracting {cohort}")):
        try:
            rows.append(process_participant(path, cohort))
        except Exception as e:
            print(f"  FAILED {path.name}: {e}")
            failed.append(path.name)

        if (i + 1) % 20 == 0 and rows:
            pd.DataFrame(rows).to_csv(f"{out_csv}.partial", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # Report
    print(f"\n{len(rows)} succeeded, {len(failed)} failed")
    if failed:
        print(f"Failed: {failed}")
    print(f"Features extracted: {len(df.columns) - 2}")

    print(f"\nValid recording window (hours)")
    print(df["valid_window_hours"].describe().round(2).to_string())

    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        print(f"\nFeatures with missing values")
        for col, n in missing.items():
            print(f"  {col:28s} {n:4d} ({100*n/len(df):.1f}%)")
    else:
        print("\nNo missing values.")

    return df


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ("lab", "home"):
        main(sys.argv[1])