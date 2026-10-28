"""Recompute metrics_<split>.json slices from saved predictions (e.g. after adding the audited `real_verified` slice).
Keeps non-metric keys (history, args, hp, ...) of the existing json.
  python -m src.eval.refresh_metrics results/E1 results/E5
"""
import glob
import json
import os
import sys

import pandas as pd

from src.eval.io import metrics_with_slices

SLICE_KEYS = {"all", "real_only", "real_verified"}


def main():
    for root in sys.argv[1:]:
        for p in glob.glob(f"{root}/*/*/predictions_*.parquet"):
            tag = os.path.basename(p)[len("predictions_"):-len(".parquet")]
            mf = os.path.join(os.path.dirname(p), f"metrics_{tag}.json")
            old = json.load(open(mf)) if os.path.exists(mf) else {}
            keep = {k: v for k, v in old.items() if k not in SLICE_KEYS and not k.startswith("lang_")}
            new = metrics_with_slices(pd.read_parquet(p))
            json.dump({**new, **keep}, open(mf, "w"), ensure_ascii=False, indent=1)
            print("refreshed", mf)


if __name__ == "__main__":
    main()
