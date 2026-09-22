import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config

warnings.filterwarnings("ignore")

FIG = config.FIGURE_DIR
METRIC = config.PRIMARY_METRIC
METRIC_LABEL = "Balanced accuracy"

COLOURS = {"Clinical": "#B4826B", "Signal": "#5DA9E9", "Combined": "#1D9E75"}
MARKERS = {"Clinical": "o", "Signal": "s", "Combined": "^"}
CLASSES = [config.LABEL_NAMES[i] for i in sorted(config.LABEL_NAMES)]
SHORT = {"Normal": "Norm", "Mild": "Mild", "Moderate": "Mod", "Severe": "Sev"}


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle=":")


def _load(name):
    p = config.OUTPUT_DIR / name
    return pd.read_csv(p) if p.exists() else None


def _safe(name):
    return name.replace("->", "_to_").replace(" ", "_")


# Cohort summary
def fig_cohort_summary(summary):
    best = summary.groupby(["Cohort", "Feature_set"])[METRIC].max().reset_index()
    cohorts = best["Cohort"].unique()
    fsets = ["Clinical", "Signal", "Combined"]

    x = np.arange(len(cohorts))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, fs in enumerate(fsets):
        vals = [best[(best.Cohort == c) & (best.Feature_set == fs)][METRIC].max()
                for c in cohorts]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=fs, color=COLOURS[fs])
        for b, v in zip(bars, vals):
            if pd.notna(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                        ha="center", fontsize=9, fontweight="bold")


    ax.set_xticks(x)
    ax.set_xticklabels(cohorts)
    ax.set_ylabel(f"{METRIC_LABEL} (best model)", fontsize=12)
    ax.set_title("Performance by cohort and feature set",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    _style(ax)

    plt.tight_layout()
    plt.savefig(FIG / "fig_4_1_cohort_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("cohort_summary.png")



# Model comparison, per cohort
def fig_model_comparison(summary):
    for cohort in summary["Cohort"].unique():
        sub = summary[summary.Cohort == cohort]

        order = (sub[sub.Feature_set == "Combined"]
                 .sort_values(METRIC, ascending=False)["Model"].tolist())
        if not order:
            order = sorted(sub["Model"].unique())

        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(order))

        for fs in ["Clinical", "Signal", "Combined"]:
            d = sub[sub.Feature_set == fs].set_index("Model").reindex(order)
            if d[METRIC].isna().all():
                continue
            yerr = d.get(f"{METRIC}_std")
            ax.errorbar(x, d[METRIC], yerr=yerr, marker=MARKERS[fs],
                        color=COLOURS[fs], linewidth=2, markersize=8,
                        capsize=3, label=fs)
            for xi, v in zip(x, d[METRIC]):
                if pd.notna(v):
                    ax.text(xi, v + 0.014, f"{v:.3f}", ha="center",
                            fontsize=7.5, color="#444")

        ax.axhline(color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=30, ha="right")
        ax.set_ylabel(METRIC_LABEL, fontsize=12)
        ax.set_xlabel("Algorithm", fontsize=12)
        ax.set_title(f"{cohort}: feature set comparison across algorithms",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        _style(ax)

        plt.tight_layout()
        plt.savefig(FIG / f"models_{_safe(cohort)}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"models_{_safe(cohort)}.png")



# Per-fold box plots
def fig_boxplots(per_fold):
    if per_fold is None:
        return

    for cohort in per_fold["Cohort"].unique():
        sub = per_fold[per_fold.Cohort == cohort]

        combined = sub[sub.Feature_set == "Combined"]
        order = (combined.groupby("Model")[METRIC].median()
                 .sort_values(ascending=False).index.tolist())
        if not order:
            order = sorted(sub["Model"].unique())

        fig, ax = plt.subplots(figsize=(max(10, len(order) * 1.3), 6))
        width = 0.25

        for i, model in enumerate(order):
            for j, fs in enumerate(["Clinical", "Signal", "Combined"]):
                vals = sub[(sub.Model == model) &
                           (sub.Feature_set == fs)][METRIC].dropna().values
                if len(vals) == 0:
                    continue
                pos = i + (j - 1) * width
                bp = ax.boxplot([vals], positions=[pos], widths=width * 0.85,
                                patch_artist=True, showfliers=False,
                                medianprops=dict(color="black", linewidth=1.4))
                bp["boxes"][0].set_facecolor(COLOURS[fs])
                bp["boxes"][0].set_alpha(0.85)
                ax.scatter(np.full(len(vals), pos), vals, s=12,
                           color="black", alpha=0.5, zorder=3)

        ax.axhline(color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=30, ha="right")
        ax.set_ylabel(METRIC_LABEL, fontsize=12)
        ax.set_title(f"{cohort}: per-fold score distribution\n"
                     f"boxes = interquartile range across "
                     f"{config.N_SPLITS} folds; points = individual folds",
                     fontsize=13, fontweight="bold")

        handles = [plt.Rectangle((0, 0), 1, 1, fc=COLOURS[fs], alpha=0.85)
                   for fs in ["Clinical", "Signal", "Combined"]]
        ax.legend(handles, ["Clinical", "Signal", "Combined"], fontsize=10)
        _style(ax)

        plt.tight_layout()
        plt.savefig(FIG / f"boxplot_{_safe(cohort)}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"boxplot_{_safe(cohort)}.png")



# Internal versus Cross-cohort
def fig_internal_external(summary):
    if "Lab->Home" not in summary["Cohort"].values:
        return

    fsets = ["Clinical", "Signal", "Combined"]
    internal = [summary[(summary.Cohort == "Lab") &
                        (summary.Feature_set == fs)][METRIC].max() for fs in fsets]
    external = [summary[(summary.Cohort == "Lab->Home") &
                        (summary.Feature_set == fs)][METRIC].max() for fs in fsets]

    x = np.arange(len(fsets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5.5))

    b1 = ax.bar(x - width / 2, internal, width, color="#1D9E75",
                label=f"Internal (laboratory, {config.N_SPLITS}-fold CV)")
    b2 = ax.bar(x + width / 2, external, width, color="#E8875A",
                label="Cross-cohort (lab to home)")

    for bars, vals in [(b1, internal), (b2, external)]:
        for b, v in zip(bars, vals):
            if pd.notna(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                        ha="center", fontsize=9, fontweight="bold")

    ax.axhline(color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(fsets)
    ax.set_ylabel(METRIC_LABEL, fontsize=12)
    ax.set_title("Internal versus Cross-cohort validation",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    _style(ax)

    plt.tight_layout()
    plt.savefig(FIG / "fig_4_7_internal_vs_external.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig_4_7_internal_vs_external.png")

# Transfer scatter
def fig_transfer_scatter(summary):
    if "Lab->Home" not in summary["Cohort"].values:
        print("  (skipping 4.9 -- no external results)")
        return

    merged = (summary[summary.Cohort == "Lab"]
              .merge(summary[summary.Cohort == "Lab->Home"],
                     on=["Feature_set", "Model"], suffixes=("_int", "_ext")))
    if merged.empty:
        return

    fig, ax = plt.subplots(figsize=(7.5, 7))
    lims = [0.2, 0.85]
    ax.plot(lims, lims, "--", color="gray", alpha=0.6, linewidth=1)
    ax.text(0.78, 0.80, "perfect transfer", fontsize=9, color="gray",
            rotation=45, ha="center")

    for fs in ["Clinical", "Signal", "Combined"]:
        d = merged[merged.Feature_set == fs]
        if d.empty:
            continue
        ax.scatter(d[f"{METRIC}_int"], d[f"{METRIC}_ext"], s=90,
                   color=COLOURS[fs], label=fs,
                   edgecolor="white", linewidth=1, zorder=3)
        for _, r in d.iterrows():
            ax.annotate(r["Model"],
                        (r[f"{METRIC}_int"], r[f"{METRIC}_ext"]),
                        fontsize=7, alpha=0.75,
                        xytext=(4, 4), textcoords="offset points")

    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(f"Internal {METRIC_LABEL.lower()}", fontsize=12)
    ax.set_ylabel(f""
                  f"Crpss-cohort {METRIC_LABEL.lower()}", fontsize=12)
    ax.set_title("Transfer from laboratory to home recordings\n"
                 "points below the diagonal indicate a transfer penalty",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIG / "fig_4_9_transfer_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig_4_9_transfer_scatter.png")

# Hyperparameter Tuning gain
def fig_tuning(summary):
    path = config.TABLE_DIR / "table_4_18_tuning.csv"
    if not path.exists():
        print("  (skipping 4.11 -- run step06 first)")
        return

    df = pd.read_csv(path)
    if "Default" not in df.columns or "Tuned" not in df.columns:
        return

    df = df.dropna(subset=["Default", "Tuned"]).sort_values("Tuned",
                                                            ascending=False)
    if df.empty:
        return

    labels = df["Model"] + "\n" + df["Cohort"]
    x = np.arange(len(df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(9, len(df) * 1.3), 5.5))
    b1 = ax.bar(x - width / 2, df["Default"], width,
                label="Default parameters", color="#B0B0B0")
    b2 = ax.bar(x + width / 2, df["Tuned"], width,
                label="Tuned", color="#1D9E75")

    for bars, vals in [(b1, df["Default"]), (b2, df["Tuned"])]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}",
                    ha="center", fontsize=8)

    for xi, (d, t) in enumerate(zip(df["Default"], df["Tuned"])):
        chg = t - d
        ax.text(xi, max(d, t) + 0.028, f"{chg:+.3f}", ha="center",
                fontsize=8.5, fontweight="bold",
                color="#1D9E75" if chg >= 0 else "#E8875A")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(METRIC_LABEL, fontsize=12)
    ax.set_title("Effect of hyperparameter tuning",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    _style(ax)

    plt.tight_layout()
    plt.savefig(FIG / "fig_4_11_tuning.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig_4_11_tuning.png")

# Selected features by modality
def fig_selected_features():
    path = config.OUTPUT_DIR / "selected_features.csv"
    if not path.exists():
        return

    sel = pd.read_csv(path)
    if sel.empty:
        return

    for (cohort, fs), grp in sel.groupby(["Cohort", "Feature_set"]):
        counts = grp["modality"].value_counts()

        fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(grp) * 0.28)),
                                 gridspec_kw={"width_ratios": [1, 2]})

        axes[0].barh(counts.index[::-1], counts.values[::-1], color="#1D9E75")
        for i, v in enumerate(counts.values[::-1]):
            axes[0].text(v + 0.1, i, str(v), va="center", fontsize=9)
        axes[0].set_xlabel("Features retained", fontsize=11)
        axes[0].set_title("By modality", fontsize=12, fontweight="bold")
        _style(axes[0])

        palette = {m: c for m, c in zip(
            counts.index,
            ["#1D9E75", "#5DA9E9", "#B4826B", "#E8875A",
             "#9B7EBD", "#57B8A9", "#C9A227", "#999999"])}

        g = grp.sort_values("rank")
        colours = [palette.get(m, "#999999") for m in g["modality"]]
        y = np.arange(len(g))
        axes[1].barh(y, [1] * len(g), color=colours)
        axes[1].set_yticks(y)
        axes[1].set_yticklabels(g["feature"], fontsize=8)
        axes[1].invert_yaxis()
        axes[1].set_xticks([])
        axes[1].set_title(f"Retained features ({len(g)})",
                          fontsize=12, fontweight="bold")
        for s in axes[1].spines.values():
            s.set_visible(False)

        fig.suptitle(f"{cohort} / {fs}: features retained by recursive elimination",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIG / f"elected_{_safe(cohort)}_{_safe(fs)}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"selected_{_safe(cohort)}_{_safe(fs)}.png")

# Per-class recall
def fig_per_class(per_class, summary):
    if per_class is None:
        print("  (skipping 4.14 -- results_per_class.csv not found)")
        return

    for cohort in per_class["Cohort"].unique():
        sub_sum = summary[summary.Cohort == cohort]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        x = np.arange(len(CLASSES))
        width = 0.26

        for i, fs in enumerate(["Clinical", "Signal", "Combined"]):
            rows = sub_sum[sub_sum.Feature_set == fs]
            if rows.empty:
                continue
            best_model = rows.loc[rows[METRIC].idxmax(), "Model"]
            d = per_class[(per_class.Cohort == cohort) &
                          (per_class.Feature_set == fs) &
                          (per_class.Model == best_model)].set_index("Class")
            vals = [d.loc[c, "recall"] if c in d.index else np.nan
                    for c in CLASSES]
            ax.bar(x + (i - 1) * width, vals, width, color=COLOURS[fs],
                   label=f"{fs} ({best_model})")

        ax.set_xticks(x)
        ax.set_xticklabels(CLASSES)
        ax.set_ylabel("Recall", fontsize=12)
        ax.set_xlabel("Severity class", fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{cohort}: per-class recall (best model per feature set)",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        _style(ax)

        plt.tight_layout()
        plt.savefig(FIG / f"fig_4_14_per_class_{_safe(cohort)}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  fig_4_14_per_class_{_safe(cohort)}.png")


# Confusion matrices, best model per cohort
def fig_confusion(confusions, summary, normalise=True):
    if confusions is None:
        return

    for cohort in confusions["Cohort"].unique():
        sub = summary[summary.Cohort == cohort]
        if sub.empty:
            continue
        best = sub.loc[sub[METRIC].idxmax()]

        d = confusions[(confusions.Cohort == cohort) &
                       (confusions.Feature_set == best["Feature_set"]) &
                       (confusions.Model == best["Model"])]
        if d.empty:
            continue

        cm_counts = d.pivot(index="True", columns="Predicted",
                            values="count").reindex(index=CLASSES,
                                                    columns=CLASSES)
        counts = cm_counts.values.astype(float)

        if normalise:
            totals = counts.sum(axis=1, keepdims=True)
            totals[totals == 0] = 1
            display = counts / totals
            vmax, fmt = 1.0, "{:.0%}"
            cbar_label = "Proportion of true class"
        else:
            display = counts
            vmax, fmt = max(counts.max(), 1), "{:.0f}"
            cbar_label = "Count"

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(display, cmap="Blues", vmin=0, vmax=vmax)

        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                v = display[i, j]
                # Annotate WITH the underlying count so a normalised
                # cell still shows how many participants it represents
                label = fmt.format(v)
                if normalise:
                    label += f"\n(n={int(counts[i, j])})"
                ax.text(j, i, label, ha="center", va="center", fontsize=9,
                        color="white" if v > vmax * 0.55 else "black")

        ax.set_xticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASSES, rotation=20, ha="right")
        ax.set_yticks(range(len(CLASSES)))
        ax.set_yticklabels(CLASSES)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual (true class)", fontsize=11)
        kind = "row-normalised: diagonal = recall" if normalise else "raw counts"
        ax.set_title(f"{cohort}: {best['Model']} / {best['Feature_set']}\n"
                     f"{METRIC_LABEL} = {best[METRIC]:.3f}  ({kind})",
                     fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, label=cbar_label)

        plt.tight_layout()
        suffix = "_normalised" if normalise else "_counts"
        fname = f"fig_4_16_confusion_{_safe(cohort)}{suffix}.png"
        plt.savefig(FIG / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  {fname}")



# Confusion grids
def fig_confusion_grids(confusions, summary, normalise=False):
    if confusions is None:
        return

    combos = confusions[["Cohort", "Feature_set"]].drop_duplicates()

    for _, combo in combos.iterrows():
        cohort, fs = combo.Cohort, combo.Feature_set
        models = sorted(confusions[(confusions.Cohort == cohort) &
                                   (confusions.Feature_set == fs)]["Model"].unique())
        if not models:
            continue

        s = summary[(summary.Cohort == cohort) & (summary.Feature_set == fs)]
        if not s.empty:
            ranked = s.sort_values(METRIC, ascending=False)["Model"].tolist()
            models = [m for m in ranked if m in models] + \
                     [m for m in models if m not in ranked]

        ncols = 5
        nrows = int(np.ceil(len(models) / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(3.6 * ncols, 4.0 * nrows),
                                 constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()

        for ax, model in zip(axes, models):
            d = confusions[(confusions.Cohort == cohort) &
                           (confusions.Feature_set == fs) &
                           (confusions.Model == model)]
            cm = d.pivot(index="True", columns="Predicted",
                         values="count").reindex(index=CLASSES, columns=CLASSES)
            vals = cm.values.astype(float)

            if normalise:
                totals = vals.sum(axis=1, keepdims=True)
                totals[totals == 0] = 1
                vals = vals / totals
                vmax, fmt = 1.0, "{:.2f}"
            else:
                vmax, fmt = max(vals.max(), 1), "{:.0f}"

            ax.imshow(vals, cmap="Blues", vmin=0, vmax=vmax)
            for i in range(len(CLASSES)):
                for j in range(len(CLASSES)):
                    ax.text(j, i, fmt.format(vals[i, j]), ha="center",
                            va="center", fontsize=8,
                            color="white" if vals[i, j] > vmax * 0.55 else "black")

            score = ""
            row = summary[(summary.Cohort == cohort) &
                          (summary.Feature_set == fs) &
                          (summary.Model == model)]
            if not row.empty:
                score = f"\n{METRIC_LABEL} = {row.iloc[0][METRIC]:.3f}"

            labels = [SHORT[c] for c in CLASSES]
            ax.set_xticks(range(len(CLASSES)))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_yticks(range(len(CLASSES)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_title(f"{model}{score}", fontsize=10, fontweight="bold",
                        pad=8)

        for ax in axes[len(models):]:
            ax.axis("off")

        kind = "row-normalised (diagonal = recall)" if normalise else "counts"
        fig.suptitle(f"{cohort} — {fs} features — {kind}\n"
                     f"rows = actual, columns = predicted",
                     fontsize=13, fontweight="bold")

        suffix = "_normalised" if normalise else ""
        fname = f"_grid_{_safe(cohort)}_{_safe(fs)}{suffix}.png"
        fig.savefig(FIG / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {fname}")

def main():
    summary = _load("results_summary.csv")
    if summary is None:
        print("results_summary.csv not found")
        return

    per_fold = _load("results_per_fold.csv")
    per_class = _load("results_per_class.csv")
    confusions = _load("confusion_matrices.csv")

if __name__ == "__main__":
    main()