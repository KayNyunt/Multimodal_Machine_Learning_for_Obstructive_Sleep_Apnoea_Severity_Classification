import warnings, os

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

import config
from utils import build_cohort
from step_05_modelling import MODELS, make_pipeline, feature_sets, SCORING, METRIC

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

RS = config.RANDOM_STATE

FAMILIES = {
    "Tree ensemble": {"XGBoost", "LightGBM", "CatBoost", "RandomForest",
                      "HistGB", "GradBoost", "Bagging", "AdaBoost"},
    "Neural": {"MLP"},
    "Probabilistic": {"GaussianNB"},
}


def _family_of(model_name):
    for fam, members in FAMILIES.items():
        if model_name in members:
            return fam
    return "Other"


def _ranked_models(cohort, feature_set):
    path = config.OUTPUT_DIR / "results_summary.csv"
    if not path.exists():
        raise SystemExit("results_summary.csv not found -- run step05 first.")
    summary = pd.read_csv(path)
    sub = summary[(summary.Cohort == cohort) &
                 (summary.Feature_set == feature_set)]
    if sub.empty:
        raise SystemExit(f"No results for {cohort}/{feature_set}.")
    return sub.sort_values(METRIC, ascending=False)


def _top_n_selection(ranked, n):
    return ranked.head(n)["Model"].tolist()


def _diverse_selection(ranked, n):
    chosen, seen_families = [], set()
    for _, row in ranked.iterrows():
        fam = _family_of(row["Model"])
        if fam not in seen_families:
            chosen.append(row["Model"])
            seen_families.add(fam)
        if len(chosen) == n:
            break
    return chosen


def _build_estimators(model_names, cat, num):
    estimators = []
    for name in model_names:
        factory, scale = MODELS[name]
        pipe = make_pipeline(factory(), cat, num, scale=scale)
        estimators.append((name, pipe))
    return estimators


def _score_ensemble(ensemble, X, y, cv):
    res = cross_validate(ensemble, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    return ({m: res[f"test_{m}"].mean() for m in SCORING},
            {m: res[f"test_{m}"].std() for m in SCORING})


def run(cohort, feature_set, top_n):
    df, feats = build_cohort(
        config.LAB_FEATURES_CSV if cohort == "Lab" else config.HOME_FEATURES_CSV,
        cohort, verbose=False)
    cols, cat, num = feature_sets(feats)[feature_set]
    X, y = df[cols], df["Severity"]

    ranked = _ranked_models(cohort, feature_set)
    best_single = ranked.iloc[0]
    print(f"{cohort} / {feature_set}")
    print(f"  Best single model: {best_single['Model']} "
          f"({METRIC}={best_single[METRIC]:.3f})")

    top_n_models = _top_n_selection(ranked, top_n)
    diverse_models = _diverse_selection(ranked, top_n)

    print(f"\n  Top-{top_n}  : {top_n_models}")
    print(f"    families: {[_family_of(m) for m in top_n_models]}")
    print(f"  Diverse : {diverse_models}")
    print(f"    families: {[_family_of(m) for m in diverse_models]}")

    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=RS)
    rows = []

    configs = {
        f"Top-{top_n} stack": ("stack", top_n_models),
        f"Top-{top_n} vote": ("vote", top_n_models),
        "Diverse stack": ("stack", diverse_models),
        "Diverse vote": ("vote", diverse_models),
    }

    for label, (kind, model_names) in configs.items():
        if len(model_names) < 2:
            print(f"\n  {label}: skipped, fewer than 2 base models available")
            continue

        estimators = _build_estimators(model_names, cat, num)

        if kind == "stack":
            ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=2000),
                cv=config.N_SPLITS,           # internal CV for meta-features
                stack_method="predict_proba",
                n_jobs=-1)
        else:
            ensemble = VotingClassifier(
                estimators=estimators, voting="soft", n_jobs=-1)

        try:
            means, stds = _score_ensemble(ensemble, X, y, cv)
            row = {"Cohort": cohort, "Feature_set": feature_set,
                  "Ensemble": label, "n_base_models": len(model_names),
                  "base_models": ", ".join(model_names)}
            row.update(means)
            row["change_vs_best_single"] = means[METRIC] - best_single[METRIC]
            rows.append(row)
            print(f"\n  {label:16s} {METRIC}={means[METRIC]:.3f} "
                  f"(+/-{stds[METRIC]:.3f})  "
                  f"vs best single: {row['change_vs_best_single']:+.3f}")
        except Exception as e:
            print(f"\n  {label}: FAILED -- {e}")

    return pd.DataFrame(rows), best_single


def main(cohort="Lab", feature_set="Combined", top_n=4):
    ens_df, best_single = run(cohort, feature_set, top_n)

    if ens_df.empty:
        print("\nNo ensembles could be built.")
        return ens_df

    current_run = ens_df.copy()

    table_path = config.TABLE_DIR / "table_4_ensembles.csv"
    if table_path.exists():
        existing = pd.read_csv(table_path)
        existing = existing[~((existing.Cohort == cohort) &
                              (existing.Feature_set == feature_set))]
        ens_df = pd.concat([existing, current_run], ignore_index=True)
    ens_df.to_csv(table_path, index=False)

    print(f"\n{'='*72}")
    print(f"SUMMARY  --  {cohort} / {feature_set}")
    print(f"{'='*72}")
    best_ensemble = current_run.loc[current_run[METRIC].idxmax()]
    print(f"  Best single model : {best_single['Model']:15s} "
          f"{METRIC}={best_single[METRIC]:.3f}")
    print(f"  Best ensemble     : {best_ensemble['Ensemble']:15s} "
          f"{METRIC}={best_ensemble[METRIC]:.3f}  "
          f"({best_ensemble['change_vs_best_single']:+.3f})")

if __name__ == "__main__":
    main()