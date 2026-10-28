"""Import the author's blind A7 annotation into results/E4/human_check.csv.

Input: results/human_check_blind1.csv -- the author's labels (Excel export: cp1251, ';'), rows in the shuffled order of
results/E4/human_check_blind.csv, which was given without any automatic or earlier labels. Text columns of the export
lose Kazakh letters (cp1251), so rows are matched by position to human_check_blind.csv and then by (id, paraphrase)
to the original template order of human_check.csv (ids repeat across A7 styles).
The author stated (24.10.2026) that the annotation was done blind, without seeing samples.csv or any earlier labels.
"""
import os
import shutil

import pandas as pd

E4 = "results/E4"


def main():
    lab = pd.read_csv("results/human_check_blind1.csv", sep=";", encoding="cp1251", dtype=str)
    blind = pd.read_csv(f"{E4}/human_check_blind.csv", encoding="utf-8-sig", dtype=str)
    assert list(lab.id) == list(blind.id), "row order differs from human_check_blind.csv"
    blind["meaning_kept"], blind["readable"] = lab.meaning_kept.str.strip().values, lab.readable.str.strip().values
    assert blind[["meaning_kept", "readable"]].isin(["0", "1"]).all().all(), "labels must be 0/1"

    tpl_path = f"{E4}/human_check_template.csv"
    if not os.path.exists(tpl_path):  # keep the original blank template once
        shutil.copy(f"{E4}/human_check.csv", tpl_path)
    tpl = pd.read_csv(tpl_path, encoding="utf-8-sig", dtype=str)
    m = blind.set_index(["id", "paraphrase"])
    key = list(zip(tpl.id, tpl.attacked))
    assert all(k in m.index for k in key), "a template pair is missing from the blind file"
    tpl["my_meaning_kept"] = [m.loc[k, "meaning_kept"] for k in key]
    tpl["my_readable"] = [m.loc[k, "readable"] for k in key]
    tpl.to_csv(f"{E4}/human_check.csv", index=False, encoding="utf-8-sig")
    print(tpl[["id", "attack", "my_meaning_kept", "my_readable"]].to_string())


if __name__ == "__main__":
    main()
