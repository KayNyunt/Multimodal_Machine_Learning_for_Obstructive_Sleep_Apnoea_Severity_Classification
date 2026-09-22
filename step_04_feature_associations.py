import numpy as np
import pandas as pd

import config
from utils import build_cohort

MODALITY_OF = [
    ("Oximetry", ["spo2", "odi", "hypoxic"]),
    ("Airflow", ["airflow"]),
    ("Electrocardiography", ["ecg", "hrv"]),
    ("Respiratory effort", ["effort"]),
    ("Plethysmography", ["pleth"]),
    ("Pulse transit time", ["ptt"]),
    ("Clinical", ["age", "bmi", "sex", "race", "ethnicity", "_bp"]),
]


def modality_of(feature):
    low = feature.lower()
    for label, patterns in MODALITY_OF:
        if any(p in low for p in patterns):
            return label
    return "Other"


def correlations(df, features, label):
    """Spearman correlation of each feature with severity."""
    rows = []
    for f in features:
        valid = df[[f, "Severity"]].dropna()
        if len(valid) < 30:            # too few to be meaningful
            rho, n = np.nan, len(valid)
        else:
            rho = valid[f].corr(valid["Severity"], method="spearman")
            n = len(valid)
        rows.append({"Modality": modality_of(f), "Feature": f,
                     f"rho_{label}": round(rho, 3) if pd.notna(rho) else np.nan,
                     f"n_{label}": n})
    return pd.DataFrame(rows)


def main():
    print("Loading cohorts\n")
    lab, lab_feats = build_cohort(config.LAB_FEATURES_CSV, "Laboratory")

    home_available = config.HOME_FEATURES_CSV.exists()
    if home_available:
        home, home_feats = build_cohort(config.HOME_FEATURES_CSV, "Home")
    else:
        print(f"\n[!] {config.HOME_FEATURES_CSV} not found -- laboratory only\n")

    lab_corr = correlations(lab, lab_feats, "lab")

    if home_available:
        home_corr = correlations(home, home_feats, "home")
        table = lab_corr.merge(
            home_corr[["Feature", "rho_home", "n_home"]],
            on="Feature", how="outer")
    else:
        table = lab_corr

    table["_abs"] = table["rho_lab"].abs()
    modality_order = [m for m, _ in MODALITY_OF] + ["Other"]
    table["_mod"] = pd.Categorical(table["Modality"],
                                   categories=modality_order, ordered=True)
    table = (table.sort_values(["_mod", "_abs"], ascending=[True, False])
                  .drop(columns=["_abs", "_mod"]))

    table.to_csv(config.TABLE_DIR / "feature_correlations.csv",
                 index=False)

    print(f"\n{'='*72}")
    print(" FEATURE CORRELATIONS WITH SEVERITY (Spearman)")
    print(f"{'='*72}")
    print(table.to_string(index=False))

    pairs = [
        ("Airflow", "airflow_std", "airflow_apnea_rate"),
        ("Airflow", "airflow_std", "airflow_hypop_pct"),
        ("Respiratory effort", "effort_corr_overall", "effort_paradox_pct"),
    ]

    rows = []
    for modality, whole, event in pairs:
        if whole in lab.columns and event in lab.columns:
            w = lab[[whole, "Severity"]].dropna()
            e = lab[[event, "Severity"]].dropna()
            rows.append({
                "Modality": modality,
                "Whole-recording feature": whole,
                "rho": round(w[whole].corr(w["Severity"], method="spearman"), 3),
                "Event-level feature": event,
                "rho ": round(e[event].corr(e["Severity"], method="spearman"), 3),
            })

    if rows:
        form = pd.DataFrame(rows)
        form.to_csv(config.TABLE_DIR / "feature_formulation.csv",
                    index=False)
        print(f"\n{'='*72}")
        print("EFFECT OF FEATURE FORMULATION (laboratory)")
        print(f"{'='*72}")
        print(form.to_string(index=False))

    return table


if __name__ == "__main__":
    main()