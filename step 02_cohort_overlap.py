import config
import pandas as pd
from utils import subject_id_from_path, load_clinical

def get_ids(folder, label):
    """Collect participant IDs from the filenames in a folder."""
    if not folder.exists():
        print(f"Not found: {folder}")
        return set()
    ids = {subject_id_from_path(f) for f in folder.rglob("*.edf")}
    ids = {i for i in ids if i is not None}
    print(f"  {label:20s} {len(ids):4d} participants")
    return ids


def main():
    print("Recordings available by modality\n")
    lab_full = get_ids(config.EDF_LAB_FULL, "Laboratory, full-night")
    lab_split = get_ids(config.EDF_LAB_SPLIT, "Laboratory, split-night")
    home = get_ids(config.EDF_HOME, "Home")

    overlap = lab_full & home
    home_only = home - lab_full - lab_split
    lab_only = lab_full - home

    print(f"\nOverlap")
    print(f"  In both laboratory-full and home : {len(overlap)}")
    print(f"  Laboratory only                  : {len(lab_only)}")
    print(f"  Home only                        : {len(home_only)}")
    print(f"  Unique participants overall      : "
          f"{len(lab_full | lab_split | home)}")

    pd.DataFrame({"nsrrid": sorted(overlap)}).to_csv(
        config.OVERLAP_CSV, index=False)
    print(f"\nSaved overlap IDs -> {config.OVERLAP_CSV}")

    if not overlap:
        print("\nNo overlap. config.OVERLAP_STRATEGY has no effect.")
        return

    # What would excluding them do?
    clinical = load_clinical()

    def distribution(ids, label):
        sub = clinical[clinical.nsrrid.isin(ids)]
        counts = sub["Severity"].value_counts().sort_index()
        props = sub["Severity"].value_counts(normalize=True).sort_index()
        return {
            "Set": label,
            "n": len(sub),
            **{config.LABEL_NAMES[k]: f"{counts.get(k,0)} ({props.get(k,0):.2f})"
               for k in sorted(config.LABEL_NAMES)},
            "_normal_mild": props.get(0, 0) + props.get(1, 0),
        }

    rows = [
        distribution(home, "All home"),
        distribution(home - overlap, "Home, overlap excluded"),
        distribution(lab_full, "All laboratory"),
        distribution(lab_full - overlap, "Laboratory, overlap excluded"),
    ]

    table = pd.DataFrame(rows)
    display = table.drop(columns=["_normal_mild"])

    print(f"\n{'='*70}")
    print("EFFECT OF EXCLUDING OVERLAPPING PARTICIPANTS")
    print(f"{'='*70}")
    print(display.to_string(index=False))

    display.to_csv(config.TABLE_DIR / "table_4_16_overlap_effect.csv",
                   index=False)

    home_all = table.loc[0, "_normal_mild"]
    home_cut = table.loc[1, "_normal_mild"]
    lab_all = table.loc[2, "_normal_mild"]
    lab_cut = table.loc[3, "_normal_mild"]

    print(f"\nProportion of Normal + Mild cases")
    print(f"  Home, all               : {home_all:.2f}")
    print(f"  Home, overlap excluded  : {home_cut:.2f}   "
          f"(change {home_cut - home_all:+.2f})")
    print(f"  Laboratory, all         : {lab_all:.2f}")
    print(f"  Laboratory, excluded    : {lab_cut:.2f}   "
          f"(change {lab_cut - lab_all:+.2f})")

    print(f"""
INTERPRETATION
  Excluding overlapping participants from HOME shifts its Normal+Mild
  proportion by {home_cut - home_all:+.2f}. Excluding them from LABORATORY shifts it
  by {lab_cut - lab_all:+.2f}.
  Current setting: config.OVERLAP_STRATEGY = "{config.OVERLAP_STRATEGY}"
""")

    return table


if __name__ == "__main__":
    main()
