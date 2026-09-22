import warnings, os

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score

import config
from utils import build_cohort
from step_05_modelling import MODELS, feature_sets, METRIC
from step_10_SHAP import (
    TREE_MODELS, _get_best_model_name, _fit_and_transform,
    _compute_shap_values, CLASSES,
)

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

RS = config.RANDOM_STATE

SIZE_STEP = 2
MIN_FEATURES = 3

MODALITY_PATTERNS = [
    ("Oximetry", ["spo2", "odi", "hypoxic"]),
    ("Airflow", ["airflow"]),
    ("Electrocardiography", ["ecg", "hrv"]),
    ("Respiratory effort", ["effort"]),
    ("Plethysmography", ["pleth"]),
    ("Pulse transit time", ["ptt"]),
    ("Clinical", ["age", "bmi", "sex", "race", "ethnicity", "_bp"]),
]


def _modality_of(feature):
    low = feature.lower()
    for label, patterns in MODALITY_PATTERNS:
        if any(p in low for p in patterns):
            return label
    return "Other"


def _rank_by_shap(shap_values, names):
    stacked = np.stack([np.abs(sv) for sv in shap_values])
    mean_abs = stacked.mean(axis=(0, 1))
    order = np.argsort(mean_abs)[::-1]
    return [names[i] for i in order], mean_abs[order]


def _sweep_sizes(X, y, ranked_names, name_to_col, model_name, cv):
    factory, _ = MODELS[model_name]
    max_k = len(ranked_names)

    sizes = list(range(MIN_FEATURES, max_k, SIZE_STEP))
    if max_k not in sizes:
        sizes.append(max_k)

    rows = []
    for k in sizes:
        cols = [name_to_col[n] for n in ranked_names[:k]]
        X_k = X[:, cols]
        scores = cross_val_score(factory(), X_k, y, cv=cv,
                                 scoring=METRIC, n_jobs=-1)
        rows.append({"n_features": k, METRIC: scores.mean(),
                     f"{METRIC}_std": scores.std()})
        print(f"      top-{k:3d} features: {METRIC}={scores.mean():.3f} "
              f"(SD {scores.std():.3f})")

    return pd.DataFrame(rows)


def run(cohort, feature_set):
    df, feats = build_cohort(
        config.LAB_FEATURES_CSV if cohort == "Lab" else config.HOME_FEATURES_CSV,
        cohort, verbose=False)

    model_name, baseline_score = _get_best_model_name(cohort, feature_set)
    print(f"{cohort} / {feature_set}: best model is {model_name} "
          f"({METRIC}={baseline_score:.3f})")

    if model_name not in TREE_MODELS:
        print(f"  '{model_name}' is not a tree ensemble -- SHAP selection "
              f"skipped for this cell (see step10 for the same limitation).")
        return None, None, None

    cols, cat, num = feature_sets(feats)[feature_set]

    clf, X_enc, names = _fit_and_transform(df, cols, cat, num, model_name)
    shap_values = _compute_shap_values(clf, X_enc)
    ranked_names, ranked_importance = _rank_by_shap(shap_values, names)

    name_to_col = {n: i for i, n in enumerate(names)}
    y = df["Severity"]
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=RS)

    print(f"  Sweeping feature counts (step={SIZE_STEP})...")
    sweep = _sweep_sizes(X_enc, y, ranked_names, name_to_col, model_name, cv)

    best_row = sweep.loc[sweep[METRIC].idxmax()]
    best_k = int(best_row["n_features"])
    selected = ranked_names[:best_k]

    print(f"\n  Best: top-{best_k} SHAP-ranked features, "
          f"{METRIC}={best_row[METRIC]:.3f} "
          f"(full baseline: {baseline_score:.3f}, "
          f"change: {best_row[METRIC]-baseline_score:+.3f})")

    summary_row = {
        "Cohort": cohort, "Feature_set": feature_set, "Model": model_name,
        "n_features_full": len(names), "n_features_selected": best_k,
        f"{METRIC}_full": baseline_score,
        f"{METRIC}_selected": best_row[METRIC],
        f"{METRIC}_selected_std": best_row[f"{METRIC}_std"],
        "change_vs_full": best_row[METRIC] - baseline_score,
    }

    selected_rows = [
        {"Cohort": cohort, "Feature_set": feature_set, "rank": r + 1,
         "feature": f, "modality": _modality_of(f),
         "mean_abs_shap": float(ranked_importance[r])}
        for r, f in enumerate(selected)
    ]

    return summary_row, selected_rows, sweep


def compare_with_rfecv(cohort, feature_set, shap_selected):
    path = config.OUTPUT_DIR / "selected_features.csv"
    if not path.exists():
        print("  (selected_features.csv from step06 not found -- run "
              "step06 first for the RFECV comparison)")
        return None

    rfecv = pd.read_csv(path)
    rfecv_selected = set(rfecv[(rfecv.Cohort == cohort) &
                               (rfecv.Feature_set == feature_set)]["feature"])
    shap_set = set(shap_selected)

    if not rfecv_selected:
        print("  (no matching RFECV selection found for this cohort/feature set)")
        return None

    intersection = rfecv_selected & shap_set
    union = rfecv_selected | shap_set
    jaccard = len(intersection) / len(union) if union else 0.0

    print(f"\n  RFECV selected : {len(rfecv_selected)} features")
    print(f"  SHAP selected  : {len(shap_set)} features")
    print(f"  In both        : {len(intersection)} features "
          f"(Jaccard = {jaccard:.3f})")
    print(f"  RFECV only     : {sorted(rfecv_selected - shap_set)}")
    print(f"  SHAP only      : {sorted(shap_set - rfecv_selected)}")

    return {"Cohort": cohort, "Feature_set": feature_set,
           "n_rfecv": len(rfecv_selected), "n_shap": len(shap_set),
           "n_both": len(intersection), "jaccard": round(jaccard, 3),
           "rfecv_only": ", ".join(sorted(rfecv_selected - shap_set)),
           "shap_only": ", ".join(sorted(shap_set - rfecv_selected))}


def main(cohort="Lab", feature_set="Combined"):
    summary_row, selected_rows, sweep = run(cohort, feature_set)

    if summary_row is None:
        return None

    t = config.TABLE_DIR


    summary_df = pd.DataFrame([summary_row])
    path = t / "shap_selection.csv"
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[~((existing.Cohort == cohort) &
                              (existing.Feature_set == feature_set))]
        summary_df = pd.concat([existing, summary_df], ignore_index=True)
    summary_df.to_csv(path, index=False)

    print(f"\n{'-'*72}")
    print("SHAP-BASED SELECTION VS FULL FEATURE SET")
    print(f"{'-'*72}")
    print(summary_df.round(3).to_string(index=False))

    selected_df = pd.DataFrame(selected_rows)
    sel_path = t / "shap_selected_by_modality.csv"
    by_mod = (selected_df.groupby(["Cohort", "modality"])
             .size().reset_index(name="retained"))
    if sel_path.exists():
        existing = pd.read_csv(sel_path)
        existing = existing[existing.Cohort != cohort]
        by_mod = pd.concat([existing, by_mod], ignore_index=True)
    by_mod.to_csv(sel_path, index=False)

    print(f"\n{'-'*72}")
    print("SHAP-SELECTED FEATURES BY MODALITY")
    print(f"{'-'*72}")
    print(by_mod.to_string(index=False))

    print(f"\n{'-'*72}")
    print("RFECV VS SHAP SELECTION OVERLAP")
    print(f"{'-'*72}")
    overlap_row = compare_with_rfecv(
        cohort, feature_set, [r["feature"] for r in selected_rows])

    if overlap_row:
        overlap_path = t / "rfecv_vs_shap_overlap.csv"
        overlap_df = pd.DataFrame([overlap_row])
        if overlap_path.exists():
            existing = pd.read_csv(overlap_path)
            existing = existing[~((existing.Cohort == cohort) &
                                  (existing.Feature_set == feature_set))]
            overlap_df = pd.concat([existing, overlap_df], ignore_index=True)
        overlap_df.to_csv(overlap_path, index=False)

    print(f"\nSaved tables to {t}/")
    return summary_row, selected_rows, sweep


if __name__ == "__main__":
    main()