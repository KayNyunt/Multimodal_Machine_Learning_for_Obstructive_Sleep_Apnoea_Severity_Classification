import warnings, os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from utils import build_cohort
from step_05_modelling import MODELS, make_pipeline, feature_sets, METRIC

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import shap

FIG = config.OUTPUT_DIR / "figures"
FIG.mkdir(exist_ok=True, parents=True)

CLASSES = [config.LABEL_NAMES[i] for i in sorted(config.LABEL_NAMES)]
COLOURS = ["#1D9E75", "#5DA9E9", "#E8875A", "#B4826B"]

# Tree-based models only
TREE_MODELS = {"RandomForest", "Bagging", "GradBoost", "HistGB",
               "LightGBM", "XGBoost", "CatBoost"}


def _get_best_model_name(cohort, feature_set):
    path = config.OUTPUT_DIR / "results_summary.csv"
    if not path.exists():
        raise SystemExit("results_summary.csv not found")
    summary = pd.read_csv(path)
    row = summary[(summary.Cohort == cohort) &
                  (summary.Feature_set == feature_set)]
    if row.empty:
        raise SystemExit(f"No results for {cohort}/{feature_set}.")
    best = row.loc[row[METRIC].idxmax()]
    return best["Model"], best[METRIC]


def _fit_and_transform(df, cols, cat, num, model_name):
    factory, scale = MODELS[model_name]
    pipe = make_pipeline(factory(), cat, num, scale=scale)

    X = df[cols]
    y = df["Severity"]
    pipe.fit(X, y)

    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    X_transformed = pre.transform(X)

    try:
        names = list(pre.get_feature_names_out())
        names = [n.split("__", 1)[-1] for n in names]
    except Exception:
        names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    return clf, X_transformed, names


def _compute_shap_values(clf, X):
    explainer = shap.TreeExplainer(clf)
    raw = explainer.shap_values(X)

    if isinstance(raw, list):
        return raw
    if raw.ndim == 3:
        return [raw[:, :, c] for c in range(raw.shape[2])]
    return [raw]


# Global bar charts
def plot_bar_overall(shap_values, names, filename, top_n=15):
    stacked = np.stack([np.abs(sv) for sv in shap_values])   # classes, n, features
    mean_abs = stacked.mean(axis=(0, 1))

    order = np.argsort(mean_abs)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(order))))
    y = np.arange(len(order))
    ax.barh(y, mean_abs[order][::-1], color="#1D9E75")
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order][::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value| (average across classes)", fontsize=11)
    ax.set_title("Overall feature importance", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle=":")

    plt.tight_layout()
    plt.savefig(FIG / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {filename}")


def plot_bar_perclass(shap_values, names, filename, top_n=10):
    stacked = np.stack([np.abs(sv).mean(axis=0) for sv in shap_values])  # classes, features
    overall_rank = stacked.mean(axis=0)
    order = np.argsort(overall_rank)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.5 * len(order))))
    y = np.arange(len(order))
    width = 0.8 / len(CLASSES)

    for c, cls in enumerate(CLASSES):
        vals = stacked[c, order][::-1]
        ax.barh(y + (c - 1.5) * width, vals, width,
               label=cls, color=COLOURS[c])

    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order][::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Feature importance by severity class",
                fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle=":")

    plt.tight_layout()
    plt.savefig(FIG / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {filename}")


# Beeswarm plots
def plot_beeswarm(shap_values_for_class, X, names, class_name, filename,
                  top_n=15):
    fig = plt.figure(figsize=(8, max(4.5, 0.4 * top_n)))
    shap.summary_plot(
        shap_values_for_class, X, feature_names=names,
        max_display=top_n, show=False, plot_size=None)
    plt.title(f"SHAP contributions to '{class_name}' predictions",
             fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {filename}")

def main(cohort="Lab", feature_set="Combined", top_n=15):
    df_path = (config.LAB_FEATURES_CSV if cohort == "Lab"
              else config.HOME_FEATURES_CSV)
    df, feats = build_cohort(df_path, cohort, verbose=False)

    model_name, score = _get_best_model_name(cohort, feature_set)
    print(f"{cohort}, {feature_set}: best model is {model_name} "
          f"({config.PRIMARY_METRIC}={score:.3f})")

    if model_name not in TREE_MODELS:
        return None

    cols, cat, num = feature_sets(feats)[feature_set]
    clf, X, names = _fit_and_transform(df, cols, cat, num, model_name)

    print(f"Computing SHAP values for {X.shape[0]} participants x "
          f"{X.shape[1]} features...")
    shap_values = _compute_shap_values(clf, X)
    print(f"  {len(shap_values)} class arrays returned "
          f"(expected {len(CLASSES)})")

    print()
    plot_bar_overall(shap_values, names, "fig_shap_bar_overall.png", top_n)
    plot_bar_perclass(shap_values, names, "fig_shap_bar_perclass.png",
                      min(top_n, 10))

    print()
    severe_idx = CLASSES.index("Severe")
    plot_beeswarm(shap_values[severe_idx], X, names, "Severe",
                 "fig_shap_beeswarm_Severe.png", top_n)

    for i, cls in enumerate(CLASSES):
        if cls == "Severe":
            continue
        plot_beeswarm(shap_values[i], X, names, cls,
                     f"fig_shap_beeswarm_{cls}.png", top_n)

    return shap_values, names


if __name__ == "__main__":
    main()