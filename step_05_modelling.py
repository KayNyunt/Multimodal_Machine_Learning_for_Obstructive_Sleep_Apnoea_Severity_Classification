import warnings, os

import numpy as np
import pandas as pd

from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     cross_val_predict)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (RandomForestClassifier, BaggingClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix,
                             make_scorer)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.metrics import geometric_mean_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

import config
from utils import build_cohort

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

RS = config.RANDOM_STATE
METRIC = config.PRIMARY_METRIC
N_CLASSES = len(config.SEVERITY_ORDER)

MODELS = {
    "XGBoost": (lambda: xgb.XGBClassifier(
        objective="multi:softprob", num_class=N_CLASSES,
        random_state=RS, eval_metric="mlogloss", verbosity=0), False),

    "LightGBM": (lambda: lgb.LGBMClassifier(
        objective="multiclassova", num_class=N_CLASSES,
        random_state=RS, class_weight="balanced", verbose=-1), False),

    "CatBoost": (lambda: CatBoostClassifier(
        random_state=RS, verbose=0, auto_class_weights="Balanced"), False),

    "RandomForest": (lambda: RandomForestClassifier(
        random_state=RS, class_weight="balanced", n_estimators=100), False),

    "HistGB": (lambda: HistGradientBoostingClassifier(
        random_state=RS, class_weight="balanced"), False),

    "GradBoost": (lambda: GradientBoostingClassifier(random_state=RS), False),

    "Bagging": (lambda: BaggingClassifier(random_state=RS, n_estimators=50), False),

    "AdaBoost": (lambda: AdaBoostClassifier(random_state=RS, n_estimators=50), False),

    "MLP": (lambda: MLPClassifier(
        random_state=RS, max_iter=1000, hidden_layer_sizes=(50, 30)), True),

    "GaussianNB": (lambda: GaussianNB(), True),
}

# PIPELINE
def make_pipeline(model, categorical, numeric, scale=False, resampler=None):
    steps = []

    if categorical:
        steps.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]), categorical))

    if numeric:
        num_steps = [("impute", SimpleImputer(strategy="median"))]
        if scale:
            num_steps.append(("scale", StandardScaler()))
        steps.append(("num", Pipeline(num_steps), numeric))

    pre = ColumnTransformer(steps, remainder="drop")

    if resampler is not None:
        return ImbPipeline([("pre", pre), ("resample", resampler),
                            ("clf", model)])

    return Pipeline([("pre", pre), ("clf", model)])

# METRICS
SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
    "g_mean": make_scorer(geometric_mean_score, average="macro"),
}


def score_predictions(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro",
                                           zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro",
                                     zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "g_mean": geometric_mean_score(y_true, y_pred, average="macro"),
    }


def per_class_rows(y_true, y_pred, **meta):
    names = [config.LABEL_NAMES[i] for i in sorted(config.LABEL_NAMES)]
    rep = classification_report(y_true, y_pred, target_names=names,
                                output_dict=True, zero_division=0)
    return [{**meta, "Class": c,
             "precision": rep[c]["precision"],
             "recall": rep[c]["recall"],
             "f1": rep[c]["f1-score"],
             "support": rep[c]["support"]}
            for c in names if c in rep]


def confusion_rows(y_true, y_pred, **meta):
    labels = sorted(config.LABEL_NAMES)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return [{**meta,
             "True": config.LABEL_NAMES[t],
             "Predicted": config.LABEL_NAMES[p],
             "count": int(cm[i, j])}
            for i, t in enumerate(labels) for j, p in enumerate(labels)]


def error_distance(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    d = np.abs(y_true - y_pred)
    return {
        "correct_pct": 100 * np.mean(d == 0),
        "adjacent_error_pct": 100 * np.mean(d == 1),
        "distant_error_pct": 100 * np.mean(d >= 2),
        "within_one_class_pct": 100 * np.mean(d <= 1),
    }


# FEATURE SETS
def feature_sets(signal_features):
    clin = config.CLINICAL_FEATURES
    cat = config.CLINICAL_CATEGORICAL
    num = config.CLINICAL_NUMERIC
    return {
        "Clinical": (clin, cat, num),
        "Signal": (signal_features, [], signal_features),
        "Combined": (clin + signal_features, cat, num + signal_features),
    }


# WITHIN-COHORT CROSS-VALIDATION
def run_within_cohort(df, signal_features, cohort, resampler=None):
    y = df["Severity"]
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True,
                         random_state=RS)

    summary, per_fold, per_class, confusions = [], [], [], []

    for fs_name, (cols, cat, num) in feature_sets(signal_features).items():
        X = df[cols]
        print(f"\n  {cohort} / {fs_name}  ({len(cols)} features)")

        for model_name, (factory, scale) in MODELS.items():
            try:
                pipe = make_pipeline(factory(), cat, num,
                                     scale=scale, resampler=resampler)

                res = cross_validate(pipe, X, y, cv=cv,
                                     scoring=SCORING, n_jobs=-1)

                meta = {"Cohort": cohort, "Feature_set": fs_name,
                        "Model": model_name}

                row = dict(meta)
                for m in SCORING:
                    row[m] = res[f"test_{m}"].mean()
                    row[f"{m}_std"] = res[f"test_{m}"].std()

                # Out-of-fold predictions
                oof = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
                row.update(error_distance(y, oof))
                summary.append(row)

                for i in range(config.N_SPLITS):
                    per_fold.append({**meta, "fold": i,
                                     **{m: res[f"test_{m}"][i] for m in SCORING}})

                per_class.extend(per_class_rows(y, oof, **meta))
                confusions.extend(confusion_rows(y, oof, **meta))

                print(f"    {model_name:13s} {METRIC} {row[METRIC]:.3f} "
                      f"(SD {row[f'{METRIC}_std']:.3f})   "
                      f"F1 {row['f1_macro']:.3f}   "
                      f"within-1 {row['within_one_class_pct']:.0f}%")

            except Exception as e:
                print(f"    {model_name:13s} FAILED: {e}")

    return summary, per_fold, per_class, confusions

# CROSS-COHORT VALIDATION
def run_external(lab, home, lab_features, home_features, resampler=None):
    if config.OVERLAP_CSV.exists():
        overlap = pd.read_csv(config.OVERLAP_CSV)["nsrrid"].tolist()

        if config.OVERLAP_STRATEGY == "exclude_from_lab":
            n0 = len(lab)
            lab = lab[~lab["nsrrid"].isin(overlap)]
            print(f"  Removed {n0 - len(lab)} overlapping participants "
                  f"from TRAINING (n={len(lab)})")
            print(f"  Test cohort intact (n={len(home)})")
        else:
            n0 = len(home)
            home = home[~home["nsrrid"].isin(overlap)]
            print(f"  Removed {n0 - len(home)} overlapping participants "
                  f"from TEST (n={len(home)})")
            print(f"  Training cohort intact (n={len(lab)})")
    else:
        print(" No overlap file ")

    dist = home["Severity"].value_counts(normalize=True).sort_index()
    print(f"  Test severity: "
          f"{ {config.LABEL_NAMES[k]: round(v,2) for k,v in dist.items()} }")

    # Only features present in both cohorts
    common = sorted(set(lab_features) & set(home_features))
    print(f"  Features common to both cohorts: {len(common)}")
    not_shared = sorted(set(lab_features) ^ set(home_features))
    if not_shared:
        print(f"  Excluded (not in both): {not_shared}")

    summary, per_class, confusions = [], [], []

    for fs_name, (cols, cat, num) in feature_sets(common).items():
        print(f"\n  Lab -> Home / {fs_name}  ({len(cols)} features)")

        for model_name, (factory, scale) in MODELS.items():
            try:
                pipe = make_pipeline(factory(), cat, num,
                                     scale=scale, resampler=resampler)

                pipe.fit(lab[cols], lab["Severity"])
                y_pred = pipe.predict(home[cols])
                y_true = home["Severity"]

                meta = {"Cohort": "Lab->Home", "Feature_set": fs_name,
                        "Model": model_name}
                row = {**meta, **score_predictions(y_true, y_pred),
                       **error_distance(y_true, y_pred)}
                summary.append(row)
                per_class.extend(per_class_rows(y_true, y_pred, **meta))
                confusions.extend(confusion_rows(y_true, y_pred, **meta))

                print(f"    {model_name:13s} {METRIC} {row[METRIC]:.3f}   "
                      f"F1 {row['f1_macro']:.3f}   "
                      f"within-1 {row['within_one_class_pct']:.0f}%")

            except Exception as e:
                print(f"    {model_name:13s} FAILED: {e}")

    return summary, per_class, confusions

# MAIN
def main(resampler=None):
    print("Loading cohorts\n")
    lab, lab_feats = build_cohort(config.LAB_FEATURES_CSV, "Laboratory")

    home_ready = config.HOME_FEATURES_CSV.exists()
    if home_ready:
        home, home_feats = build_cohort(config.HOME_FEATURES_CSV, "Home")
    else:
        print(f"\n[!] {config.HOME_FEATURES_CSV} not found.")

    all_summary, all_fold, all_class, all_conf = [], [], [], []

    print(f"\n{'='*72}")
    print("WITHIN-COHORT CROSS-VALIDATION")
    print(f"{'='*72}")
    s, f, c, cm = run_within_cohort(lab, lab_feats, "Lab", resampler)
    all_summary += s; all_fold += f; all_class += c; all_conf += cm

    if home_ready:
        s, f, c, cm = run_within_cohort(home, home_feats, "Home", resampler)
        all_summary += s; all_fold += f; all_class += c; all_conf += cm

        print(f"\n{'='*72}")
        print("CROSS-COHORT VALIDATION")
        print(f"{'='*72}")
        s, c, cm = run_external(lab, home, lab_feats, home_feats, resampler)
        all_summary += s; all_class += c; all_conf += cm

if __name__ == "__main__":
    main()