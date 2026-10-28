"""E5 clean-data evaluation of defenses on val/test (+ normalizer report features for E6).

  python -m src.eval.eval_defense --exp E1 --model M6 --seed 13 --defense aware --name M6+aware
Writes results/E5/<name>/<seed>/predictions_{val,test}.parquet, metrics_*.json, and norm_report_{split}.parquet.
"""
import argparse
import os

import pandas as pd

from src.attacks.run_attacks import Counted, load_model
from src.eval.io import save_predictions
from src.normalize import normalize

SPLITS = os.environ.get("SPLITS", "data/splits")  # E11: data/splits_strip, data/splits_src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--defense", default="none", choices=["none", "naive", "aware"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-exp", default="E5")
    a = ap.parse_args()
    model = Counted(load_model(a.exp, a.model, a.seed), a.defense)
    for split in ["val", "test", "test_ood_event"]:
        if not os.path.exists(f"{SPLITS}/{split}.parquet"):
            continue
        df = pd.read_parquet(f"{SPLITS}/{split}.parquet")
        if not len(df):
            continue
        save_predictions(a.out_exp, a.name, a.seed, df, model(df.text.tolist()), tag=split, extra={"defense": a.defense, "base": f"{a.exp}/{a.model}"})
        rep = pd.DataFrame([normalize(t)[1] for t in df.text])
        rep.insert(0, "id", df.id.values)
        rep["anomaly_classes"] = rep.anomaly_classes.map(lambda x: ",".join(x))
        rep.to_parquet(f"results/{a.out_exp}/{a.name}/{a.seed}/norm_report_{split}.parquet", index=False)


if __name__ == "__main__":
    main()
