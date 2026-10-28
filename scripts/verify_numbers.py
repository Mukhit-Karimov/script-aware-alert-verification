"""Check that every number in paper_numbers.json (and RESULTS.md, if present) is backed by a value in results/ or tables/.

A number matches if some source value, rounded to the same number of decimals (also as a percentage), equals it.
Integers <= 31 and years/dates/seeds/section-like tokens are ignored (counts of sections, seeds, budgets, dates).
Exit code 1 if anything is unmatched; the list is printed.
"""
import bisect
import glob
import json
import os
import re
import sys

import pandas as pd

IGNORE = {"13", "42", "2026", "2027", "2025", "2024", "2021", "2018", "2017", "100", "128", "300", "500", "1000", "0.80", "0.85"}


def source_values():
    vals = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            vals.append(float(x))
    for f in glob.glob("results/**/*.json", recursive=True):
        try:
            walk(json.load(open(f)))
        except Exception:
            pass
    for f in glob.glob("tables/*.csv") + glob.glob("results/**/*.csv", recursive=True):
        try:
            d = pd.read_csv(f)
            vals += [float(v) for v in d.select_dtypes("number").values.ravel() if pd.notna(v)]
        except Exception:
            pass
    return vals


def keyset(vals):
    ks = set()
    for v in vals:
        for x in (v, v * 100):
            for dec in range(0, 5):
                ks.add(f"{round(x, dec):.{dec}f}")
    return ks


def numbers_in(text):
    text = re.sub(r"\d{4}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{4}|\d{1,2}:\d{2}", " ", text)  # dates/times
    text = re.sub(r"`[^`]*`", " ", text)  # code spans / file names
    out = []
    for m in re.finditer(r"(?<![\w.])-?\d+(?:[.,]\d+)?", text):
        s = m.group().replace(",", ".")
        if s in IGNORE or ("." not in s and abs(int(float(s))) <= 31):
            continue
        out.append(s)
    return out


def main():
    vals = source_values()
    ks = keyset(vals)
    exact = sorted(vals + [v * 100 for v in vals])  # paper_numbers.json keeps full precision

    def has_exact(v):  # a source value within 1e-9 (relative) of v
        i = bisect.bisect_left(exact, v - 1e-9 * max(1.0, abs(v)))
        return i < len(exact) and abs(exact[i] - v) <= 1e-9 * max(1.0, abs(v))
    bad = []
    md = open("RESULTS.md", encoding="utf-8").read() if os.path.exists("RESULTS.md") else ""
    for s in numbers_in(md):
        if s.lstrip("-") not in ks:
            bad.append(("RESULTS.md", s))
    pn = json.load(open("paper_numbers.json"))
    for k, v in pn.items():
        dec = len(str(v).split(".")[1]) if "." in str(v) else 0
        if not has_exact(v) and f"{round(v, dec):.{dec}f}" not in ks:
            bad.append(("paper_numbers.json", f"{k}={v}"))
    print(f"checked RESULTS.md ({len(numbers_in(md))} numbers) and paper_numbers.json ({len(pn)} numbers)")
    for b in bad:
        print("UNMATCHED", *b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
