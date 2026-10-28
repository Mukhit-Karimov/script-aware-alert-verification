"""E8: deployability on the CPU node. One subprocess per (model, threads, batch) so peak RSS is per configuration.

  python -m src.eval.efficiency                 # all configs -> results/E8/efficiency.json, tables/T7_efficiency.*
  python -m src.eval.efficiency --one M6 1 1    # internal: measure one config, print json
Must run when the node is otherwise idle (load average is logged with every measurement).
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd

MODELS = {"M1": "results/E1/M1/13", "M4": "results/E1/M4/13", "M5": "results/E1/M5/13", "M6": "results/E1/M6/13",
          "M7": "results/E1/M7/13", "M8": "results/E1/M8/13", "M6_D4": "results/E5/M6_D3/13"}
N_MSG = 256


def dir_size_mb(p):
    if os.path.isfile(p):
        return os.path.getsize(p) / 2 ** 20
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs) / 2 ** 20


def measure_one(name, threads, batch):
    os.environ["OMP_NUM_THREADS"] = os.environ["NTHREADS"] = str(threads)  # load_model reads NTHREADS
    import torch
    torch.set_num_threads(threads)
    te = pd.read_parquet("data/splits/test.parquet").sample(N_MSG, random_state=0)
    texts = te.text.tolist()
    if name == "L1":  # Layer 1 alone
        from src.normalize import normalize
        fn, size = (lambda b: [normalize(t) for t in b]), dir_size_mb("src/normalize.py")
    else:
        d = MODELS[name]
        from src.attacks.run_attacks import load_model
        exp, mdl, seed = d.split("/")[1:4]
        f = load_model(exp, mdl, int(seed))
        if name == "M6_D4":
            from src.normalize import normalize
            fn = lambda b: f([normalize(t)[0] for t in b])
        else:
            fn = f
        size = dir_size_mb(f"{d}/model") if os.path.isdir(f"{d}/model") else dir_size_mb(f"{d}/model.pkl")
    fn(texts[:batch])  # warm-up
    lat = []
    t0 = time.perf_counter()
    for i in range(0, len(texts), batch):
        s = time.perf_counter()
        fn(texts[i:i + batch])
        lat.append((time.perf_counter() - s) / len(texts[i:i + batch]) * 1000)
    total = time.perf_counter() - t0
    return {"model": name, "threads": threads, "batch": batch, "p50_ms_per_msg": float(np.percentile(lat, 50)),
            "p95_ms_per_msg": float(np.percentile(lat, 95)), "throughput_msg_s": len(texts) / total,
            "size_mb": size, "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "loadavg": os.getloadavg()[0], "n_msg": len(texts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", nargs=3)
    ap.add_argument("--threads-all", type=int, default=28)
    a = ap.parse_args()
    if a.one:
        print(json.dumps(measure_one(a.one[0], int(a.one[1]), int(a.one[2])))); return
    rows = []
    for name in ["L1"] + [m for m, d in MODELS.items() if os.path.exists(d)]:
        for th in (1, a.threads_all):
            for b in (1, 32):
                cmd = ["numactl", "--cpunodebind=0", "--membind=0", sys.executable, "-m", "src.eval.efficiency", "--one", name, str(th), str(b)]
                out = subprocess.run(cmd, capture_output=True, text=True)
                line = [l for l in out.stdout.splitlines() if l.startswith("{")]
                if line:
                    rows.append(json.loads(line[-1])); print(line[-1], flush=True)
                else:
                    print("FAIL", name, th, b, out.stderr[-500:], flush=True)
    os.makedirs("results/E8", exist_ok=True)
    json.dump(rows, open("results/E8/efficiency.json", "w"), indent=1)
    t = pd.DataFrame(rows)
    t.to_csv("tables/T7_efficiency.csv", index=False)
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Model & Threads & Batch & p50 (ms/msg) & p95 (ms/msg) & Msg/s & Size (MB) \\", r"\midrule"]
    for _, r in t.iterrows():
        lines.append(f"{r.model} & {r.threads} & {r.batch} & {r.p50_ms_per_msg:.2f} & {r.p95_ms_per_msg:.2f} & {r.throughput_msg_s:.1f} & {r.size_mb:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open("tables/T7_efficiency.tex", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
