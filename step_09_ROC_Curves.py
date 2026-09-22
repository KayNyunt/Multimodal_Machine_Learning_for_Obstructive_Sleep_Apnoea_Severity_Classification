import warnings, os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

import config
from utils import build_cohort
from step_05_modelling import MODELS, make_pipeline, feature_sets, METRIC

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

FIG = config.OUTPUT_DIR / "figures"
FIG.mkdir(exist_ok=True, parents=True)

CLASSES = [config.LABEL_NAMES[i] for i in sorted(config.LABEL_NAMES)]
N_CLASSES = len(CLASSES)
COLOURS = ["#1D9E75", "#5DA9E9", "#E8875A", "#B4826B"]


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3, linestyle=":")


def _get_best_model_name(cohort, feature_set):
    path = config.OUTPUT_DIR / "results_summary.csv"
    if not path.exists():
        raise SystemExit("results_summary.csv not found")
    summary = pd.read_csv(path)
    row = summary[(summary.Cohort == cohort) &
                  (summary.Feature_set == feature_set)]
    if row.empty:
        raise SystemExit(f"No results for {cohort}/{feature_set} in "
                         f"results_summary.csv")
    best = row.loc[row[METRIC].idxmax()]
    return best["Model"], best[METRIC]


def _oof_probabilities(df, cols, cat, num, model_name, y):
    factory, scale = MODELS[model_name]
    pipe = make_pipeline(factory(), cat, num, scale=scale)
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    return cross_val_predict(pipe, df[cols], y, cv=cv,
                             method="predict_proba", n_jobs=-1)


def _external_probabilities(lab_df, home_df, cols, cat, num, model_name):
    factory, scale = MODELS[model_name]
    pipe = make_pipeline(factory(), cat, num, scale=scale)
    pipe.fit(lab_df[cols], lab_df["Severity"])
    return pipe.predict_proba(home_df[cols])

# ROC curves, one-vs-rest
def plot_roc(y_true, y_proba, title, filename):
    y_bin = label_binarize(y_true, classes=sorted(config.LABEL_NAMES))

    fig, ax = plt.subplots(figsize=(7, 6.5))
    aucs = {}

    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        aucs[cls] = roc_auc
        ax.plot(fpr, tpr, color=COLOURS[i], linewidth=2,
                label=f"{cls} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, alpha=0.6,
            label="Chance")
    ax.set_xlabel("False positive rate", fontsize=12)
    ax.set_ylabel("True positive rate", fontsize=12)
    ax.set_title(f"One-vs-rest ROC curves\n{title}", fontsize=13,
                fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    _style(ax)

    plt.tight_layout()
    plt.savefig(FIG / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {filename}")
    print(f"    AUC by class: "
          f"{', '.join(f'{c}={a:.3f}' for c, a in aucs.items())}")
    return aucs


# Precision-recall curves, one-vs-rest
def plot_pr(y_true, y_proba, title, filename):
    y_bin = label_binarize(y_true, classes=sorted(config.LABEL_NAMES))

    fig, ax = plt.subplots(figsize=(7, 6.5))
    aps = {}

    for i, cls in enumerate(CLASSES):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_bin[:, i], y_proba[:, i])
        aps[cls] = ap

        # Baseline: the prevalence of this class -- a model with no
        # skill would score at this level, and it differs per class
        # under imbalance, unlike the fixed 0.5 chance line for ROC
        prevalence = y_bin[:, i].mean()

        ax.plot(recall, precision, color=COLOURS[i], linewidth=2,
                label=f"{cls} (AP = {ap:.3f}, prevalence = {prevalence:.2f})")
        ax.axhline(prevalence, color=COLOURS[i], linewidth=0.8,
                  linestyle=":", alpha=0.5)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"One-vs-rest precision-recall curves\n{title}",
                fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(0, 1.02)
    _style(ax)

    plt.tight_layout()
    plt.savefig(FIG / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {filename}")
    print(f"    Average precision by class: "
          f"{', '.join(f'{c}={a:.3f}' for c, a in aps.items())}")
    return aps


# ============================================================
def main(cohort="Lab", feature_set="Combined", external=False):
    if external:
        model_name, score = _get_best_model_name("Lab->Home", feature_set)
        print(f"Cross-cohort validation, {feature_set}: best model is "
              f"{model_name} ({config.PRIMARY_METRIC}={score:.3f})")

        lab_df, lab_feats = build_cohort(config.LAB_FEATURES_CSV, "Lab",
                                         verbose=False)
        home_df, home_feats = build_cohort(config.HOME_FEATURES_CSV, "Home",
                                           verbose=False)

        overlap = pd.read_csv(config.OVERLAP_CSV)["nsrrid"].tolist() \
            if config.OVERLAP_CSV.exists() else []
        if config.OVERLAP_STRATEGY == "exclude_from_lab":
            lab_df = lab_df[~lab_df["nsrrid"].isin(overlap)]

        common = sorted(set(lab_feats) & set(home_feats))
        cols, cat, num = feature_sets(common)[feature_set]

        y_proba = _external_probabilities(lab_df, home_df, cols, cat, num,
                                          model_name)
        y_true = home_df["Severity"]
        title = f"Lab -> Home, {feature_set} ({model_name})"
        tag = f"external_{feature_set}"

    else:
        features_csv = (config.LAB_FEATURES_CSV if cohort == "Lab"
                        else config.HOME_FEATURES_CSV)
        df, feats = build_cohort(features_csv, cohort, verbose=False)

        model_name, score = _get_best_model_name(cohort, feature_set)
        print(f"{cohort}, {feature_set}: best model is {model_name} "
              f"({config.PRIMARY_METRIC}={score:.3f})")

        cols, cat, num = feature_sets(feats)[feature_set]
        y_true = df["Severity"]
        y_proba = _oof_probabilities(df, cols, cat, num, model_name, y_true)
        title = f"{cohort}, {feature_set} ({model_name})"
        tag = f"{cohort}_{feature_set}"

    print()
    roc_aucs = plot_roc(y_true, y_proba, title, f"fig_roc_{tag}.png")
    print()
    pr_aps = plot_pr(y_true, y_proba, title, f"fig_pr_{tag}.png")

    return {"roc_auc": roc_aucs, "average_precision": pr_aps}


if __name__ == "__main__":
    main()