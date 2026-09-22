from collections import Counter, defaultdict
import warnings

import pandas as pd
import mne
from tqdm import tqdm

import config
from utils import resolve_channel, subject_id_from_path

warnings.filterwarnings("ignore")

def audit_cohort(folder, label):
    """Scan every recording in one folder and record what resolved."""
    files = sorted(folder.rglob("*.edf"))
    print(f"\n{'='*66}\n{label}: {len(files)} recordings\n{'='*66}")

    if not files:
        print("No files found")
        return None, None, None

    rows = []
    resolved_names = defaultdict(Counter)
    unresolved = defaultdict(list)

    for path in tqdm(files, desc=f"Scanning {label}"):
        try:
            raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        except Exception as e:
            print(f"  UNREADABLE {path.name}: {e}")
            continue

        row = {"nsrrid": subject_id_from_path(path), "file": path.name}

        for modality, candidates in config.MODALITIES.items():
            ch = resolve_channel(raw, candidates)
            row[modality] = ch
            if ch:
                resolved_names[modality][ch] += 1
            else:
                unresolved[modality].append((path.name, raw.ch_names))

        rows.append(row)

    df = pd.DataFrame(rows)

    # ---- Resolution rate ----
    print(f"\n--- Resolution rate ({len(df)} recordings) ---")
    summary = []
    for modality in config.MODALITIES:
        n = df[modality].notna().sum()
        pct = 100 * n / len(df)
        flag = "" if pct == 100 else "   <-- CHECK"
        print(f"  {modality:9s} {n:4d}/{len(df)}  ({pct:5.1f}%){flag}")
        summary.append({
            "Cohort": label, "Modality": modality,
            "Resolved": n, "Total": len(df), "Percent": round(pct, 1),
            "Variants": ", ".join(resolved_names[modality].keys()),
        })

    # Which name each modality actually matched
    print("\n--- Resolved to ---")
    for modality, counter in resolved_names.items():
        variants = ", ".join(f"{name} (n={n})" for name, n in counter.most_common())
        print(f"  {modality:9s} {variants}")

    # For anything unresolved
    print("\n--- Unresolved ---")
    any_missing = False
    for modality, items in unresolved.items():
        if not items:
            continue
        any_missing = True
        print(f"\n  {modality.upper()}: {len(items)} recordings")
        for fname, chans in items[:3]:
            print(f"    {fname}")
            print(f"      available channels: {chans}")
        if len(items) > 3:
            print(f"    ... and {len(items) - 3} more")
    if not any_missing:
        print("  none -- every modality resolved in every recording")

    return df, pd.DataFrame(summary), resolved_names


def main():
    lab_df, lab_sum, lab_names = audit_cohort(config.EDF_LAB_FULL, "Laboratory")
    home_df, home_sum, home_names = audit_cohort(config.EDF_HOME, "Home")

    if lab_df is not None:
        lab_df.to_csv(config.OUTPUT_DIR / "channel_audit_lab.csv", index=False)
    if home_df is not None:
        home_df.to_csv(config.OUTPUT_DIR / "channel_audit_home.csv", index=False)

    # Combined summary
    if lab_sum is not None and home_sum is not None:
        combined = pd.concat([lab_sum, home_sum], ignore_index=True)
        table = combined.pivot(index="Modality", columns="Cohort",
                               values=["Resolved", "Percent"])
        variants = (combined.groupby("Modality")["Variants"]
                    .apply(lambda s: ", ".join(sorted(
                        {v for row in s for v in row.split(", ") if v}))))

        out = pd.DataFrame({
            "Variants encountered": variants,
            "Laboratory resolved": table[("Resolved", "Laboratory")],
            "Laboratory %": table[("Percent", "Laboratory")],
            "Home resolved": table[("Resolved", "Home")],
            "Home %": table[("Percent", "Home")],
        })
        out.to_csv(config.TABLE_DIR / "table_4_3_channel_resolution.csv")

        print(f"\n{'='*66}\nRESULTS TABLE 4.3\n{'='*66}")
        print(out.to_string())
        print(f"\nSaved -> {config.TABLE_DIR / 'table_4_3_channel_resolution.csv'}")

if __name__ == "__main__":
    main()