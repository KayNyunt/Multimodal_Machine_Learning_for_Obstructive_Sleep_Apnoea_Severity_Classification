import warnings, os

import numpy as np
import pandas as pd

from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     RandomizedSearchCV, GridSearchCV)
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

import config
from utils import build_cohort
from step_05_modelling import MODELS, make_pipeline, feature_sets

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

RS = config.RANDOM_STATE
METRIC = config.PRIMARY_METRIC

# SCOPE
MODELS_TO_TUNE = ["RandomForest", "CatBoost", "XGBoost", "LightGBM", "Bagging"]

SEARCH_TYPE = "randomized"
N_ITER = 30

SEARCH_TYPE_OVERRIDE = {
    "Bagging": "grid",
}

FEATURE_SETS_TO_RUN = ["Combined"]

# Prefixed clf__ because the model sits inside a Pipeline step named "clf"
GRIDS = {
    "RandomForest": {
        "clf__n_estimators": [10, 50, 100],
        "clf__max_depth": [10, 30, 50, 70, 90, None],
        "clf__max_features": ["sqrt", "log2"],
    },

    "XGBoost": {
        "clf__n_estimators": [10, 50, 100],
        "clf__learning_rate": [0.001, 0.01, 0.05],
        "clf__reg_alpha": [0.5, 1],
    },

    "LightGBM": {
        "clf__num_iterations": [150, 200, 500],
        "clf__learning_rate": [0.05, 0.005],
        "clf__max_depth": [10, 15, 25],
        "clf__num_leaves": [7, 15, 31],
        "clf__bagging_freq": [100, 200, 400],
        "clf__bagging_fraction": [0.8],
    },

    "CatBoost": {
        "clf__depth": [4, 5, 6, 7, 8, 9, 10],
        "clf__learning_rate": [0.01, 0.02, 0.03, 0.04],
        "clf__iterations": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    },

    "Bagging": {
        "clf__n_estimators": [10, 50, 100, 500],
        "clf__max_features": [0.6, 0.7, 0.8, 0.9, 1.0],
        "clf__bootstrap": [True, False],
        "clf__oob_score": [True, False],
    },
}

MODALITY_PATTERNS = [
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
    for label, patterns in MODALITY_PATTERNS:
        if any(p in low for p in patterns):
            return label
    return "Other"


# FEATURE SELECTION
def run_feature_selection(cohorts):
    rows, selected_rows = [], []

    for cohort, (df, signal_feats) in cohorts.items():
        y = df["Severity"]
        cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True,
                             random_state=RS)

        for fs_name, (cols, cat, num) in feature_sets(signal_feats).items():
            if fs_name not in FEATURE_SETS_TO_RUN:
                continue

            print(f"\n  {cohort} / {fs_name}")
            X = df[cols]

            # Apply pre-processing once so RFECV sees a numeric matrix
            pipe = make_pipeline(RandomForestClassifier(random_state=RS),
                                 cat, num)
            pre = pipe.named_steps["pre"]
            X_enc = pre.fit_transform(X, y)

            try:
                names = list(pre.get_feature_names_out())
            except Exception:
                names = [f"feature_{i}" for i in range(X_enc.shape[1])]

            selector = RFECV(
                estimator=RandomForestClassifier(
                    random_state=RS, class_weight="balanced", n_estimators=100),
                step=1, cv=cv, scoring=METRIC,
                min_features_to_select=3, n_jobs=-1)
            selector.fit(X_enc, y)

            kept = [n for n, k in zip(names, selector.support_) if k]
            print(f"    retained {selector.n_features_} of {len(names)} features")

            for model_name in MODELS_TO_TUNE:
                factory, _ = MODELS[model_name]
                try:
                    scores = cross_val_score(
                        factory(), X_enc[:, selector.support_], y,
                        cv=cv, scoring=METRIC, n_jobs=-1)
                    rows.append({
                        "Cohort": cohort, "Feature_set": fs_name,
                        "Model": model_name,
                        "n_features_full": X_enc.shape[1],
                        "n_features_selected": len(kept),
                        f"{METRIC}_selected": scores.mean(),
                        f"{METRIC}_selected_std": scores.std(),
                    })
                    print(f"      {model_name:13s} {scores.mean():.3f} "
                          f"(SD {scores.std():.3f})")
                except Exception as e:
                    print(f"      {model_name:13s} FAILED: {e}")

            for rank, feat in enumerate(kept, 1):
                selected_rows.append({
                    "Cohort": cohort, "Feature_set": fs_name,
                    "rank": rank, "feature": feat,
                    "modality": modality_of(feat),
                })

    return pd.DataFrame(rows), pd.DataFrame(selected_rows)

# HYPERPARAMETER TUNING
def run_tuning(cohorts):
    rows = []

    for cohort, (df, signal_feats) in cohorts.items():
        y = df["Severity"]
        cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True,
                             random_state=RS)

        for fs_name, (cols, cat, num) in feature_sets(signal_feats).items():
            if fs_name not in FEATURE_SETS_TO_RUN:
                continue

            print(f"\n  {cohort} / {fs_name}")
            X = df[cols]

            for model_name in MODELS_TO_TUNE:
                if model_name not in GRIDS:
                    continue
                factory, scale = MODELS[model_name]
                pipe = make_pipeline(factory(), cat, num, scale=scale)

                search_type = SEARCH_TYPE_OVERRIDE.get(model_name, SEARCH_TYPE)

                try:
                    if search_type == "grid":
                        search = GridSearchCV(pipe, GRIDS[model_name], cv=cv,
                                              scoring=METRIC, n_jobs=-1)
                    else:
                        search = RandomizedSearchCV(
                            pipe, GRIDS[model_name], n_iter=N_ITER, cv=cv,
                            scoring=METRIC, n_jobs=-1, random_state=RS)

                    print(f"    {model_name} ({search_type} search, "
                          f"{np.prod([len(v) for v in GRIDS[model_name].values()])} "
                          f"combinations in grid)")

                    search.fit(X, y)

                    params = {k.replace("clf__", ""): v
                              for k, v in search.best_params_.items()}

                    rows.append({
                        "Cohort": cohort, "Feature_set": fs_name,
                        "Model": model_name,
                        f"{METRIC}_tuned": search.best_score_,
                        "best_params": str(params),
                    })
                    print(f"      {model_name:13s} {search.best_score_:.3f}")
                    print(f"        {params}")

                except Exception as e:
                    print(f"      {model_name:13s} FAILED: {e}")

    return pd.DataFrame(rows)

# MAIN
def main(do_selection=True, do_tuning=True):
    print("Loading cohorts\n")
    cohorts = {}
    lab, lab_feats = build_cohort(config.LAB_FEATURES_CSV, "Laboratory")
    cohorts["Lab"] = (lab, lab_feats)

    if config.HOME_FEATURES_CSV.exists():
        home, home_feats = build_cohort(config.HOME_FEATURES_CSV, "Home")
        cohorts["Home"] = (home, home_feats)

    baseline_path = config.OUTPUT_DIR / "results_summary.csv"
    baseline = pd.read_csv(baseline_path) if baseline_path.exists() else None
    if baseline is None:
        print("results_summary.csv not found.")

    out, tab = config.OUTPUT_DIR, config.TABLE_DIR

    if do_selection:
        print(f"\n{'='*72}")
        print("FEATURE SELECTION")
        print(f"{'='*72}")
        sel, selected = run_feature_selection(cohorts)
        selected.to_csv(out / "selected_features.csv", index=False)

        if baseline is not None and not sel.empty:
            merged = sel.merge(
                baseline[["Cohort", "Feature_set", "Model", METRIC]],
                on=["Cohort", "Feature_set", "Model"], how="left")
            merged["change"] = merged[f"{METRIC}_selected"] - merged[METRIC]

            table = merged[["Cohort", "Model", "n_features_full",
                            "n_features_selected", METRIC,
                            f"{METRIC}_selected", "change"]].round(3)
            table.columns = ["Cohort", "Model", "All features",
                             "Selected", "Baseline", "After selection", "Change"]
            table.to_csv(tab / "table_4_19_selection.csv", index=False)

            print(f"\n{'-'*72}")
            print("FULL VERSUS SELECTED FEATURE SET")
            print(f"{'-'*72}")
            print(table.to_string(index=False))

        if not selected.empty:
            by_mod = (selected.groupby(["Cohort", "modality"])
                      .size().reset_index(name="retained"))
            by_mod.to_csv(tab / "selected_by_modality.csv",
                          index=False)
            print(f"\n{'-'*72}")
            print("RETAINED FEATURES BY MODALITY")
            print(f"{'-'*72}")
            print(by_mod.to_string(index=False))

    if do_tuning:
        print(f"\n{'='*72}")
        print(f"HYPERPARAMETER TUNING  "
              f"(default: {SEARCH_TYPE}; overrides: {SEARCH_TYPE_OVERRIDE})")
        print(f"{'='*72}")
        tuned = run_tuning(cohorts)

        if baseline is not None and not tuned.empty:
            merged = tuned.merge(
                baseline[["Cohort", "Feature_set", "Model", METRIC]],
                on=["Cohort", "Feature_set", "Model"], how="left")
            merged["change"] = merged[f"{METRIC}_tuned"] - merged[METRIC]

            table = merged[["Cohort", "Model", METRIC, f"{METRIC}_tuned",
                            "change", "best_params"]].round(3)
            table.columns = ["Cohort", "Model", "Default", "Tuned",
                             "Change", "Selected parameters"]
            table.to_csv(tab / "table_4_18_tuning.csv", index=False)

            print(f"\n{'-'*72}")
            print("DEFAULT VERSUS TUNED PARAMETERS")
            print(f"{'-'*72}")
            print(table.drop(columns=["Selected parameters"]).to_string(index=False))

            mean_change = merged["change"].mean()
            print(f"\n  Mean change: {mean_change:+.3f}")
        else:
            tuned.to_csv(tab / "tuning.csv", index=False)

if __name__ == "__main__":
    main()