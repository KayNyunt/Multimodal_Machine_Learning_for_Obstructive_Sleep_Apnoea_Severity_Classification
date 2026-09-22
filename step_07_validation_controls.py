import warnings, os

import numpy as np
import pandas as pd

import config
from utils import build_cohort
from step_05_modelling import (MODELS, make_pipeline, feature_sets,
                              run_within_cohort, score_predictions)

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

RS = config.RANDOM_STATE
METRIC = config.PRIMARY_METRIC

MATCH_TRAINING_SIZE = True

def run_size_control(lab, lab_feats, overlap):
    reduced = lab[~lab["nsrrid"].isin(overlap)].copy()

    print(f"  Full laboratory cohort    : {len(lab)}")
    print(f"  Reduced (overlap removed) : {len(reduced)}")
    print(f"  Removed                   : {len(lab) - len(reduced)}")

    def dist(df):
        d = df["Severity"].value_counts(normalize=True).sort_index()
        return {config.LABEL_NAMES[k]: round(v, 2) for k, v in d.items()}

    print(f"\n  Severity, full    : {dist(lab)}")
    print(f"  Severity, reduced : {dist(reduced)}")

    print(f"\n  Running cross-validation on the reduced cohort...")
    summary, _, _, _ = run_within_cohort(reduced, lab_feats, "Lab-reduced")
    return pd.DataFrame(summary)


def decompose(baseline, reduced):
    full = baseline[baseline.Cohort == "Lab"][["Feature_set", "Model", METRIC]]
    ext = baseline[baseline.Cohort == "Lab->Home"][["Feature_set", "Model", METRIC]]
    red = reduced[["Feature_set", "Model", METRIC]]

    comp = (full.merge(red, on=["Feature_set", "Model"],
                       suffixes=("_full", "_reduced"))
                .merge(ext.rename(columns={METRIC: f"{METRIC}_external"}),
                       on=["Feature_set", "Model"], how="left"))

    comp["size_effect"] = comp[f"{METRIC}_reduced"] - comp[f"{METRIC}_full"]
    comp["modality_effect"] = comp[f"{METRIC}_external"] - comp[f"{METRIC}_reduced"]
    comp["total_drop"] = comp[f"{METRIC}_external"] - comp[f"{METRIC}_full"]
    return comp


# BIDIRECTIONAL TRANSFER
def stratified_subsample(df, n, random_state=RS):
    if len(df) <= n:
        return df
    frac = n / len(df)
    out = (df.groupby("Severity", group_keys=False)
             .apply(lambda g: g.sample(max(1, round(len(g) * frac)),
                                       random_state=random_state)))
    if len(out) > n:
        out = out.sample(n, random_state=random_state)
    elif len(out) < n:
        remainder = df[~df.index.isin(out.index)]
        out = pd.concat([out, remainder.sample(n - len(out),
                                               random_state=random_state)])
    return out


def transfer(train_df, test_df, common_feats, direction):
    rows = []
    for fs_name, (cols, cat, num) in feature_sets(common_feats).items():
        print(f"\n    {direction} / {fs_name}")
        for model_name, (factory, scale) in MODELS.items():
            try:
                pipe = make_pipeline(factory(), cat, num, scale=scale)
                pipe.fit(train_df[cols], train_df["Severity"])
                y_pred = pipe.predict(test_df[cols])

                row = {"Direction": direction, "Feature_set": fs_name,
                       "Model": model_name,
                       "n_train": len(train_df), "n_test": len(test_df)}
                row.update(score_predictions(test_df["Severity"], y_pred))
                rows.append(row)
                print(f"      {model_name:13s} {row[METRIC]:.3f}")
            except Exception as e:
                print(f"      {model_name:13s} FAILED: {e}")
    return rows


def run_bidirectional(lab, home, lab_feats, home_feats, overlap):
    common = sorted(set(lab_feats) & set(home_feats))
    print(f"  Features common to both cohorts: {len(common)}")

    lab_train = lab[~lab["nsrrid"].isin(overlap)]
    home_train = home[~home["nsrrid"].isin(overlap)]

    print(f"\n  Before size matching:")
    print(f"    Lab -> Home : train {len(lab_train)}, test {len(home)}")
    print(f"    Home -> Lab : train {len(home_train)}, test {len(lab)}")

    if MATCH_TRAINING_SIZE:
        n = min(len(lab_train), len(home_train))
        lab_train = stratified_subsample(lab_train, n)
        home_train = stratified_subsample(home_train, n)
        print(f"\n  Training sets matched to n = {n}")

    rows = transfer(lab_train, home, common, "Lab->Home")
    rows += transfer(home_train, lab, common, "Home->Lab")
    return pd.DataFrame(rows)


# MAIN
def main(do_size_control=True, do_bidirectional=True):
    print("Loading cohorts\n")
    lab, lab_feats = build_cohort(config.LAB_FEATURES_CSV, "Laboratory")

    if not config.HOME_FEATURES_CSV.exists():
        raise SystemExit("Home features not found")
    home, home_feats = build_cohort(config.HOME_FEATURES_CSV, "Home")

    if not config.OVERLAP_CSV.exists():
        raise SystemExit("cohort_overlap.csv not found")
    overlap = pd.read_csv(config.OVERLAP_CSV)["nsrrid"].tolist()

    baseline_path = config.OUTPUT_DIR / "results_summary.csv"
    baseline = pd.read_csv(baseline_path) if baseline_path.exists() else None

    tab = config.TABLE_DIR

    if do_size_control:
        print(f"\n{'='*72}")
        print("TRAINING-SIZE CONTROL")
        print(f"{'='*72}")
        reduced = run_size_control(lab, lab_feats, overlap)

        if baseline is not None and "Lab->Home" in baseline.Cohort.values:
            comp = decompose(baseline, reduced)

            table = comp[["Feature_set", "Model",
                          f"{METRIC}_full", f"{METRIC}_reduced",
                          f"{METRIC}_external",
                          "size_effect", "modality_effect", "total_drop"]].round(3)
            table.columns = ["Feature set", "Model", "Lab (full)",
                             "Lab (reduced)", "Lab -> Home",
                             "Size effect", "Modality effect", "Total drop"]
            table.to_csv(tab / "table_4_15_size_control.csv", index=False)

            print(f"\n{'-'*72}")
            print("DECOMPOSING THE CROSS-COHORT VALIDATION DROP")
            print(f"{'-'*72}")
            print(table.to_string(index=False))

            print(f"\n  Mean effect by feature set")
            agg = comp.groupby("Feature_set")[
                ["size_effect", "modality_effect", "total_drop"]].mean().round(3)
            print(agg.to_string())

            size = comp["size_effect"].mean()
            mod = comp["modality_effect"].mean()
            print(f"""
  INTERPRETATION
    Mean size effect     {size:+.3f}   (fewer training participants)
    Mean modality effect {mod:+.3f}   (different recording equipment)

    Whichever dominates determines how the transfer result should be
    reported. If the size effect is substantial, state the external
    result against the REDUCED-cohort baseline rather than the full one.
""")
        else:
            reduced.to_csv(tab / "table_4_15_size_control.csv", index=False)
            print("\n  [!] No external results in baseline; saved raw output only.")

    if do_bidirectional:
        print(f"\n{'='*72}")
        print("BIDIRECTIONAL TRANSFER")
        print(f"{'='*72}")
        bi = run_bidirectional(lab, home, lab_feats, home_feats, overlap)

        pivot = bi.pivot_table(index=["Feature_set", "Model"],
                               columns="Direction", values=METRIC)
        if {"Lab->Home", "Home->Lab"}.issubset(pivot.columns):
            pivot["Difference"] = pivot["Lab->Home"] - pivot["Home->Lab"]

        pivot.round(3).to_csv(tab / "bidirectional.csv")

        print(f"\n{'-'*72}")
        print("TRANSFER IN BOTH DIRECTIONS")
        print(f"{'-'*72}")
        print(pivot.round(3).to_string())

        if {"Lab->Home", "Home->Lab"}.issubset(set(bi.Direction)):
            fwd = bi[bi.Direction == "Lab->Home"][METRIC].mean()
            rev = bi[bi.Direction == "Home->Lab"][METRIC].mean()
            print(f"""
  Lab -> Home mean : {fwd:.3f}
  Home -> Lab mean : {rev:.3f}
  Difference       : {fwd - rev:+.3f}
""")


if __name__ == "__main__":
    main()