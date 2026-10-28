"""Remaining outputs: T3/F4 (E2/E3), T6/F10/F11/F12 (E6), F13/F14 (E7), F15 (E8), F16/F17 (E9), F18 (E10).
Every function reads only files under results/ and silently skips when its inputs do not exist yet.
  python -m src.viz.other_figs
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from src.eval.stats import friedman_nemenyi
from src.viz.style import DEF_STYLE, FULL_W, HALF_W, LABEL_COLOR, LABEL_NAME, MODEL_NAME, MODEL_STYLE, OI, save, setup

LANGS = ["kk", "ru", "mixed"]


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


# ---------------- E2 / E3 ----------------
def t3_f4():
    rows = []
    for m in ["M1", "M4", "M6", "M8", "M5"]:
        for f in glob.glob(f"results/E1/{m}/*/metrics_test.json"):
            j = json.load(open(f))
            o = jl(f.replace("metrics_test", "metrics_test_ood_event"))
            r = {"model": m, "seed": int(f.split("/")[3]), "test_all": j["all"]["macro_f1"]}
            for g in LANGS:
                r[g] = j.get(f"lang_{g}", {}).get("macro_f1")
            r["ood_event"] = o["all"]["macro_f1"] if o else None
            r["ood_n"] = o["all"]["n"] if o else None
            rows.append(r)
    for f in glob.glob("results/E1/M9_*/13/metrics_test.json"):
        j = json.load(open(f)); o = jl(f.replace("metrics_test", "metrics_test_ood_event"))
        rows.append({"model": "M9", "seed": 13, "test_all": j["all"]["macro_f1"], **{g: j.get(f"lang_{g}", {}).get("macro_f1") for g in LANGS},
                     "ood_event": o["all"]["macro_f1"] if o else None, "ood_n": o["all"]["n"] if o else None})
    if not rows:
        return
    # note: the `mixed` slice contains only credible texts, so its macro-F1 reduces to a one-class score
    t = pd.DataFrame(rows).drop(columns="seed").groupby("model").agg(["mean", "std"]).round(4)
    t.columns = ["_".join(c) for c in t.columns]
    t.to_csv("tables/T3_lang_ood.csv")
    # cross-lingual transfer matrix for M6: rows = train language, cols = test language
    M = pd.DataFrame(index=["ru", "kk", "all"], columns=LANGS, dtype=float)
    for src in ["ru", "kk"]:
        vals = {g: [] for g in LANGS}
        for f in glob.glob(f"results/E2/M6_train_{src}/*/metrics_test.json"):
            j = json.load(open(f))
            for g in LANGS:
                if f"lang_{g}" in j:
                    vals[g].append(j[f"lang_{g}"]["macro_f1"])
        for g in LANGS:
            M.loc[src, g] = np.mean(vals[g]) if vals[g] else np.nan
    for g in LANGS:
        v = [json.load(open(f)).get(f"lang_{g}", {}).get("macro_f1") for f in glob.glob("results/E1/M6/*/metrics_test.json")]
        v = [x for x in v if x is not None]
        M.loc["all", g] = np.mean(v) if v else np.nan
    M.to_csv("tables/T3_transfer_M6.csv")
    if M.isna().all().all():
        return
    setup()
    fig, ax = plt.subplots(figsize=(HALF_W, 2.0))
    ax.imshow(M.values.astype(float), cmap="Blues", vmin=0, vmax=1)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if (not np.isnan(v) and v > 0.6) else "#222222")
    ax.set_xticks(range(3), ["test kk", "test ru", "test mixed"])
    ax.set_yticks(range(3), ["train ru", "train kk", "train all"])
    ax.grid(False)
    save(fig, "F4_crosslingual", "Heatmap of XLM-R macro-F1 for each training language (Russian only, Kazakh only, all) "
         "and test language (Kazakh, Russian, mixed), showing cross-lingual transfer between Russian and Kazakh.")


# ---------------- E6 ----------------
def e6():
    ms = [jl(f) for f in sorted(glob.glob("results/E6/*/metrics.json"))]
    ms = [m for m in ms if m]
    if not ms:
        return
    rows = []
    for key in [k for k in ms[0] if "." in k]:
        for part in ("test", "test_real_only"):
            vals = pd.DataFrame([m[key][part] for m in ms])
            rows.append({"model": key, "slice": part, **{f"{c}_mean": vals[c].mean() for c in vals},
                         **{f"{c}_std": vals[c].std(ddof=1) for c in vals}, "n_seeds": len(ms)})
    t = pd.DataFrame(rows)
    # task 4: bootstrap 95% CIs (results/E6/fusion_ci.csv) and the val value (target: missed forgery <= 1% on val)
    if os.path.exists("results/E6/fusion_ci.csv"):
        ci = pd.read_csv("results/E6/fusion_ci.csv")
        for c in ("false_block_rate", "review_load", "missed_forgery_rate"):
            x = ci[ci.metric == c][["model", "slice", "ci_lo", "ci_hi", "val_value"]].rename(
                columns={"ci_lo": f"{c}_ci_lo", "ci_hi": f"{c}_ci_hi", "val_value": f"{c}_val"})
            t = t.merge(x, on=["model", "slice"], how="left")
    t.to_csv("tables/T6_fusion.csv", index=False)

    def ci_(r, c):
        return f" [{r[c + '_ci_lo']:.4f}, {r[c + '_ci_hi']:.4f}]" if f"{c}_ci_lo" in r and pd.notna(r[f"{c}_ci_lo"]) else ""
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Model & Features & False block [95\% CI] & Review load [95\% CI] & Missed forgery [95\% CI] & Missed forgery (val) \\",
             r"\midrule"]
    for _, r in t[t.slice == "test"].iterrows():
        k, fs = r.model.split(".")
        mv = f"{r['missed_forgery_rate_val']:.4f}" if "missed_forgery_rate_val" in r and pd.notna(r["missed_forgery_rate_val"]) else "--"
        lines.append(f"{'LR' if k == 'lr' else 'LightGBM'} & {fs} & {r.false_block_rate_mean:.4f}{ci_(r, 'false_block_rate')} & "
                     f"{r.review_load_mean:.4f}{ci_(r, 'review_load')} & {r.missed_forgery_rate_mean:.4f}{ci_(r, 'missed_forgery_rate')} & {mv} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open("tables/T6_fusion.tex", "w").write("\n".join(lines) + "\n")
    setup()
    seed = sorted(glob.glob("results/E6/*/"))[0]
    col = {"content": OI["blue"], "source": OI["orange"], "fusion": OI["vermillion"]}
    lsty = {"content": "-", "source": "--", "fusion": "-."}
    # F10: ROC + PR for forged_alert vs rest (LR)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W * 0.75, 2.3))
    for fs in ("content", "source", "fusion"):
        d = pd.read_parquet(f"{seed}/decisions_lr.{fs}.parquet")
        y = (d.y_true == "forged_alert").astype(int)
        fpr, tpr, _ = roc_curve(y, d.risk); pr, rc, _ = precision_recall_curve(y, d.risk)
        a1.plot(fpr, tpr, color=col[fs], ls=lsty[fs], label=f"{fs} (AUC {auc(fpr, tpr):.3f})")
        a2.plot(rc, pr, color=col[fs], ls=lsty[fs], label=f"{fs} (AUC {auc(rc, pr):.3f})")
    a1.set_xlabel("False positive rate"); a1.set_ylabel("True positive rate"); a1.set_title("ROC, forged alert", loc="left")
    a2.set_xlabel("Recall"); a2.set_ylabel("Precision"); a2.set_title("PR, forged alert", loc="left")
    for a in (a1, a2):
        a.legend(frameon=False, fontsize=8)
    save(fig, "F10_fusion_roc_pr", "ROC and precision-recall curves for detecting forged alerts with content-only, "
         "source-only (simulated metadata) and fused risk scores (logistic regression, one seed).")
    # F11: review load vs missed forgery when sweeping t_review (t_block fixed at the tuned value)
    fig, ax = plt.subplots(figsize=(HALF_W * 1.1, 2.3))
    m0 = jl(f"{seed}/metrics.json")
    for fs in ("content", "fusion"):
        d = pd.read_parquet(f"{seed}/decisions_lr.{fs}.parquet")
        tb = m0[f"lr.{fs}"]["t_block"]
        pts = []
        for tr in np.quantile(d.risk, np.linspace(0, 0.995, 120)):
            dec = np.where(d.risk >= max(tb, tr), "block", np.where(d.risk >= tr, "human_review", "publish"))
            f = d.y_true == "forged_alert"
            pts.append(((dec == "human_review").mean(), (dec[f] == "publish").mean()))
        pts = np.array(pts)
        ax.plot(pts[:, 0] * 100, pts[:, 1] * 100, color=col[fs], ls=lsty[fs], label=fs)
        tr0 = m0[f"lr.{fs}"]["test"]
        ax.scatter(tr0["review_load"] * 100, tr0["missed_forgery_rate"] * 100, color=col[fs], s=20, zorder=3, edgecolor="white")
    ax.set_xlabel("Review load (% of all messages)"); ax.set_ylabel("Missed forgeries (%)")
    ax.legend(frameon=False)
    save(fig, "F11_review_tradeoff", "Trade-off curve between the share of messages sent to human review and the share of "
         "forged alerts that would be published, for content-only and fused risk; dots mark the thresholds tuned on validation.")
    # F12: decision flow true class -> decision (fusion LR), grouped bars
    d = pd.read_parquet(f"{seed}/decisions_lr.fusion.parquet")
    ct = pd.crosstab(d.y_true, d.decision, normalize="index").reindex(columns=["publish", "human_review", "block"], fill_value=0)
    fig, ax = plt.subplots(figsize=(HALF_W * 1.2, 2.0))
    dcol = {"publish": OI["blue"], "human_review": OI["yellow"], "block": OI["vermillion"]}
    x = np.arange(len(ct))
    for k, dc in enumerate(ct.columns):
        ax.bar(x + (k - 1) * 0.26, ct[dc].values * 100, 0.24, color=dcol[dc], edgecolor="#555555" if dc == "human_review" else "white",
               linewidth=0.6, label=dc.replace("_", " "))
    ax.set_xticks(x, [LABEL_NAME[l] for l in ct.index]); ax.set_ylabel("Share of class (%)")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save(fig, "F12_decision_flow", "Grouped bars showing, for each true class, the share of messages that the fused Layer-3 "
         "policy publishes, sends to human review, or blocks (test set, one seed).")


# ---------------- E7 ----------------
def e7():
    names = [n for n in ["M1", "M6", "M6_D4"] if glob.glob(f"results/E7/{n}/*/calibration.json")]
    if not names:
        return
    setup()
    nm = {"M1": "TF-IDF+LR", "M6": "XLM-R", "M6_D4": "XLM-R + D4"}
    cl = {"M1": MODEL_STYLE["M1"][0], "M6": MODEL_STYLE["M6"][0], "M6_D4": DEF_STYLE["D4"][0]}
    fig, axes = plt.subplots(2, len(names), figsize=(FULL_W, 3.2), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    axes = np.atleast_2d(axes).reshape(2, -1)
    for j, n in enumerate(names):
        s = sorted(glob.glob(f"results/E7/{n}/*/"))[0]
        cj = jl(f"{s}/calibration.json")
        for tag, ls in (("before", ":"), ("after", "-")):
            r = pd.read_parquet(f"{s}/reliability_{tag}.parquet").dropna()
            axes[0, j].plot(r.conf, r.acc, marker="o", ms=3, ls=ls, color=cl[n],
                            label=f"{tag} (ECE {cj[tag]['ece']:.3f})")
        axes[0, j].plot([0, 1], [0, 1], color="#999999", lw=0.8)
        axes[0, j].set_title(nm[n], loc="left"); axes[0, j].legend(frameon=False, fontsize=8)
        r = pd.read_parquet(f"{s}/reliability_before.parquet")
        axes[1, j].bar((r.lo + r.hi) / 2, r.n, width=1 / 15 * 0.9, color=cl[n])
        axes[1, j].set_yscale("log"); axes[1, j].set_xlabel("Confidence")
    axes[0, 0].set_ylabel("Accuracy"); axes[1, 0].set_ylabel("Count")
    save(fig, "F13_reliability", "Reliability diagrams before and after temperature scaling for TF-IDF+LR, XLM-R and XLM-R "
         "with the full defense (seed 13), with confidence histograms (log scale) underneath.")
    fig, ax = plt.subplots(figsize=(HALF_W, 2.2))
    for n in names:
        s = sorted(glob.glob(f"results/E7/{n}/*/"))[0]
        cr = pd.read_parquet(f"{s}/coverage_risk_after.parquet")
        ax.plot(cr.coverage * 100, cr.risk * 100, color=cl[n], label=nm[n], ls={"M1": "--", "M6": "-", "M6_D4": ":"}[n])
    ax.set_xlabel("Coverage: automatic decisions (%)"); ax.set_ylabel("Error rate (%)")
    ax.legend(frameon=False)
    save(fig, "F14_coverage_risk", "Coverage-risk curves: error rate of automatic decisions as a function of the share of "
         "messages decided automatically when the least confident ones go to a human (seed 13, after temperature scaling).")


# ---------------- E8 ----------------
def f15():
    e = jl("results/E8/efficiency.json")
    if not e:
        return
    e = pd.DataFrame(e)
    e = e[(e.batch == 1) & (e.threads == e.threads.max())]
    setup()
    fig, ax = plt.subplots(figsize=(HALF_W * 1.2, 2.3))
    for _, r in e.iterrows():
        mdl = r.model.split("_")[0]
        if mdl == "L1":
            continue
        j = jl(f"results/E1/{mdl}/13/metrics_test.json") if r.model != "M6_D4" else jl("results/E5/M6_D3+aware/13/metrics_test.json")
        if not j:
            continue
        c, mk, _ = MODEL_STYLE.get(mdl, (OI["gray"], "o", "-"))
        if r.model == "M6_D4":
            c, mk = DEF_STYLE["D4"]
        ax.scatter(r.p95_ms_per_msg, j["all"]["macro_f1"], s=12 + r.size_mb / 15, color=c, marker=mk, edgecolor="white", linewidth=0.5)
        ax.annotate(MODEL_NAME.get(mdl, mdl) + (" +D4" if r.model == "M6_D4" else ""), (r.p95_ms_per_msg, j["all"]["macro_f1"]),
                    fontsize=8, xytext=(4, 2), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("p95 latency per message (ms, log scale)"); ax.set_ylabel("Macro-F1 (test, seed 13)")
    from matplotlib.ticker import FixedLocator, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50])); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["2", "5", "10", "20", "50"]); ax.set_xlim(2, 50)
    save(fig, "F15_latency_quality", "Scatter of p95 per-message latency (log scale, batch 1, 28 CPU threads) against test "
         "macro-F1 (seed-13 model, the one benchmarked) for each model; point size is model size on disk.")


# ---------------- E9 ----------------
def f17():
    p = "results/E9/m1_top_features.csv"
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    setup()
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.8))
    fig.subplots_adjust(wspace=0.75)
    lab = lambda f: f'"{f[2:].replace(" ", "␣")}"' if f.startswith("c:") else f[2:]
    for ax, cls in zip(axes, ["credible", "misinfo", "forged_alert"]):
        x = d[d["class"] == cls].head(12).iloc[::-1]
        ax.barh(range(len(x)), x.coef, color=LABEL_COLOR[cls], height=0.7)
        ax.set_yticks(range(len(x)), [lab(f) for f in x.feature], fontsize=8)
        ax.set_title(LABEL_NAME[cls], loc="left"); ax.set_xlabel("LR coefficient")
        ax.grid(axis="y", visible=False)
    save(fig, "F17_top_features", "Horizontal bar charts of the twelve largest TF-IDF logistic-regression coefficients for "
         "each class; word and character n-grams such as hearsay markers for misinfo and urgency or ID-number requests for forged alerts.")


def f16():
    s = jl("results/E9/shap.json")
    if not s:
        return
    pairs = [x for x in s if x["group"].startswith("attacked")][:4]
    if len(pairs) < 2:
        return
    setup()
    fig, axes = plt.subplots(len(pairs), 1, figsize=(FULL_W, 0.75 * len(pairs)))
    norm = TwoSlopeNorm(vcenter=0, vmin=-0.3, vmax=0.3)
    cmap = plt.get_cmap("RdBu_r")
    for ax, x in zip(np.atleast_1d(axes), pairs):
        cls = 0  # attribution toward `credible`
        vals = np.array(x["shap"])[:, cls]
        xpos = 0.0
        ax.axis("off")
        for tkn, v in list(zip(x["tokens"], vals))[:40]:
            t = ax.text(xpos, 0.5, tkn, fontsize=8, va="center", ha="left",
                        bbox=dict(boxstyle="square,pad=0.1", fc=cmap(norm(v)), ec="none"))
            fig.canvas.draw()
            bb = t.get_window_extent().transformed(ax.transData.inverted())
            xpos = bb.x1 + 0.004
            if xpos > 0.98:
                break
        ax.set_title(f"{x['group'].replace('_', ' ')}: true {x['label']}, predicted {x['pred']}", loc="left", fontsize=8)
    save(fig, "F16_shap", "Token-level SHAP attributions toward the credible class for the same messages before and after a "
         "homoglyph attack; red tokens push the XLM-R prediction toward credible, blue tokens away from it.")


# ---------------- E10 ----------------
def f18():
    """Friedman/Nemenyi over blocks = seeds x slices (all, kk, ru, real_verified) on macro-F1."""
    models = [m for m in ["M1", "M2", "M3", "M4", "M5", "M6", "M8"] if len(glob.glob(f"results/E1/{m}/*/metrics_test.json")) == 3]
    if len(models) < 4:
        return
    blocks = []
    for s in (13, 42, 2026):
        for sl in ("all", "lang_kk", "lang_ru", "real_verified"):
            blocks.append([json.load(open(f"results/E1/{m}/{s}/metrics_test.json"))[sl]["macro_f1"] for m in models])
    stat, p, ranks, cd = friedman_nemenyi(np.array(blocks))
    json.dump({"models": models, "friedman_chi2": stat, "p": p, "avg_ranks": dict(zip(models, ranks.tolist())), "cd": cd,
               "n_blocks": len(blocks), "blocks": "seeds x {all, kk, ru, real_verified}"}, open("results/E10_friedman.json", "w"), indent=1)
    setup()
    k = len(models)
    half = (k + 1) // 2
    fig, ax = plt.subplots(figsize=(FULL_W * 0.85, 0.9 + 0.22 * half))
    ax.set_xlim(-0.6, k + 1.6); ax.set_ylim(0.62 - 0.14 * half, 1.05); ax.invert_xaxis(); ax.axis("off")
    ax.hlines(0.75, 1, k, color="#333333", lw=0.8)
    for r in range(1, k + 1):
        ax.vlines(r, 0.72, 0.78, color="#333333", lw=0.8); ax.text(r, 0.84, str(r), ha="center", fontsize=8)
    order = np.argsort(ranks)
    # classic CD layout: better half labelled on the right, worse half on the left, each on its own row
    for i, j in enumerate(order):
        c, mk, _ = MODEL_STYLE[models[j]]
        row = i if i < half else k - 1 - i
        y = 0.64 - 0.14 * row
        xe, ha = (0.85, "left") if i < half else (k + 0.15, "right")
        ax.plot([ranks[j], ranks[j], xe], [0.75, y, y], color=c, lw=0.9)
        ax.text(xe - 0.05 if i < half else xe + 0.05, y, f"{MODEL_NAME[models[j]]} ({ranks[j]:.2f})", fontsize=8, ha=ha, va="center", color="#222222")
    ax.plot([1, 1 + cd], [0.95, 0.95], color="#000000", lw=1.5)
    ax.text(1 + cd / 2, 0.98, f"CD = {cd:.2f}", ha="center", fontsize=8)
    save(fig, "F18_critical_difference", f"Critical-difference diagram of average macro-F1 ranks (Friedman p = {p:.3g}, Nemenyi "
         f"CD = {cd:.2f}) over seeds and test slices; models closer than CD are not significantly different.")


def main():
    os.makedirs("tables", exist_ok=True)
    for f in (t3_f4, e6, e7, f15, f17, f16, f18):
        try:
            f(); print("ok", f.__name__)
        except Exception as e:
            print("skip", f.__name__, type(e).__name__, e)


if __name__ == "__main__":
    main()
