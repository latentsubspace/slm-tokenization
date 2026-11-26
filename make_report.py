#!/usr/bin/env python
"""Self-contained HTML report (matplotlib PNGs embedded) from the run.py outputs.

Usage: python make_report.py [--config config.yaml] [--runs results/runs] [--out results/report.html]

Works with partial results (missing cells are shown as '-'). Smoke-test runs (limit_test / max_steps)
are ignored. CIs are 95% percentile bootstrap over test sentences; for seed-averaged cells the same
resampled sentences are used for every seed, and gaps are paired (EN and HU on the same sentences).
Also writes results/summary_main.csv, summary_curve.csv, summary_corr.csv and prints a short summary.
"""
import argparse
import base64
import csv
import glob
import io
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import stats

EN_COLOR, HU_COLOR = "#2a78d6", "#eb6834"  # categorical slots 1-2 of the default validated palette
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "text.color": INK})


# ---------------------------------------------------------------- loading

def load(cfg, runs_dir):
    with open(os.path.join(cfg["data"]["out_dir"], "fpb_split.jsonl"), encoding="utf-8") as f:
        test = sorted((r for r in map(json.loads, f) if r["split"] == "test"), key=lambda r: r["id"])
    ids = np.array([r["id"] for r in test])
    runs = {}
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "metrics.json"))):
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        if m.get("limit_test") or m.get("max_steps"):
            continue
        t = m["task"]
        preds = {}
        with open(os.path.join(os.path.dirname(path), "preds.jsonl"), encoding="utf-8") as f:
            P = [json.loads(line) for line in f]
        for lang in {p["lang"] for p in P}:
            Q = sorted((p for p in P if p["lang"] == lang), key=lambda p: p["id"])
            if not np.array_equal([p["id"] for p in Q], ids):
                print(f"WARN skip {t['id']}: test ids differ")
                break
            preds[lang] = {k: np.array([p[k] for p in Q]) for k in ("pred", "pred_cal") if k in Q[0]}
        else:
            runs[t["id"]] = {"task": t, "res": m["results"], "preds": preds}
    return test, runs


def find(runs, **kw):
    return [r for r in runs.values() if all(str(r["task"].get(k)) == str(v) for k, v in kw.items())]


# ---------------------------------------------------------------- statistics

def f1_boot(gold, pred, idx):
    """Macro-F1 for each bootstrap row of idx (B x n), vectorized; same definition as sklearn zero_division=0."""
    g, p = gold[idx], pred[idx]
    out = []
    for c in range(3):
        tp = ((g == c) & (p == c)).sum(1)
        d = 2 * tp + ((g != c) & (p == c)).sum(1) + ((g == c) & (p != c)).sum(1)
        out.append(np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0))
    return np.mean(out, 0)


class Boot:
    def __init__(self, gold, aa, n_boot, seed=0):
        rng = np.random.default_rng(seed)
        self.sets = {"all": (gold, np.arange(len(gold))), "allagree": (gold[aa], np.flatnonzero(aa))}
        self.idx = {k: rng.integers(0, len(v[0]), (n_boot, len(v[0]))) for k, v in self.sets.items()}

    def cell(self, preds, subset="all"):
        """preds: list of full-test prediction arrays (one per seed) -> dict with mean, sd, ci, boot."""
        if not preds:
            return None
        gold, sel = self.sets[subset]
        full = np.arange(len(gold))[None]
        pts = [f1_boot(gold, p[sel], full)[0] for p in preds]
        boot = np.mean([f1_boot(gold, p[sel], self.idx[subset]) for p in preds], 0)
        return {"mean": float(np.mean(pts)), "median": float(np.median(pts)), "sd": float(np.std(pts, ddof=1)) if len(pts) > 1 else 0.0,
                "ci": np.percentile(boot, [2.5, 97.5]), "boot": boot, "n": len(pts)}


def gap(a, b):
    if not a or not b:
        return None
    d = a["boot"] - b["boot"]
    return {"mean": a["mean"] - b["mean"], "ci": np.percentile(d, [2.5, 97.5])}


def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    Z = np.c_[np.ones(len(rz)), rz]
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    r = float(np.corrcoef(ex, ey)[0, 1])
    df = len(x) - 3
    p = float(2 * stats.t.sf(abs(r) * math.sqrt(df / max(1e-12, 1 - r * r)), df))
    return r, p


def needed(xs, ys, target):
    """Smallest training size where the curve reaches target (log-linear interpolation)."""
    if ys[0] >= target:
        return f"<={xs[0]}"
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if y1 >= target:
            t = (target - y0) / (y1 - y0)
            return str(round(math.exp(math.log(x0) + t * (math.log(x1) - math.log(x0)))))
    return f">{xs[-1]}"


# ---------------------------------------------------------------- formatting

def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if isinstance(v, dict):
        s = f"{v['mean']:.{nd}f}"
        if "ci" in v:
            s += f" [{v['ci'][0]:.{nd}f}, {v['ci'][1]:.{nd}f}]"
        return s
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def html_table(rows, cols):
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{fmt(r.get(c))}</td>" for c in cols) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def write_csv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([fmt(r.get(c)) for c in cols])


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f'<img alt="" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


# ---------------------------------------------------------------- figures

def fig_dots(names, panels):
    """panels: list of (title, {name: (en_cell, hu_cell)}); paired dot plot with CI whiskers."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 0.38 * len(names) + 1), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(names))[::-1]
    for ax, (title, data) in zip(axes, panels):
        for yi, n in zip(y, names):
            en, hu = data.get(n, (None, None))
            if en and hu:
                ax.plot([en["mean"], hu["mean"]], [yi + 0.12, yi - 0.12], color=GRID, lw=2, zorder=1)
            for c, col, lab, dy in ((en, EN_COLOR, "EN", 0.12), (hu, HU_COLOR, "HU", -0.12)):
                if c:
                    ax.errorbar(c["mean"], yi + dy, xerr=[[c["mean"] - c["ci"][0]], [c["ci"][1] - c["mean"]]],
                                fmt="o", ms=6, color=col, ecolor=col, elinewidth=1.2, capsize=0,
                                mec="white", mew=1.5, zorder=2, label=lab if yi == y[0] or n == names[0] else None)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("macro-F1")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, names)
    h, lab = axes[0].get_legend_handles_labels()
    uniq = dict(zip(lab, h))
    fig.legend(uniq.values(), uniq.keys(), loc="upper right", frameon=False, ncol=2)
    return fig


def fig_curves(names, curves):
    """curves: {name: {lang: (xs, means, sds)}}; small multiples with shared y."""
    cols = 5
    rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.1 * rows), sharey=True, squeeze=False)
    for ax, n in zip(axes.flat, names):
        for lang, col in (("en", EN_COLOR), ("hu", HU_COLOR)):
            if lang in curves.get(n, {}):
                xs, ms, sds = curves[n][lang]
                ax.fill_between(xs, np.array(ms) - sds, np.array(ms) + sds, color=col, alpha=0.15, lw=0)
                ax.plot(xs, ms, "-o", color=col, lw=1.5, ms=4, mec="white", mew=1, label=lang.upper())
        ax.set_xscale("log")
        if curves.get(n):
            xs = max((v[0] for v in curves[n].values()), key=len)
            ax.set_xticks(xs, [str(x) for x in xs], fontsize=7)
            ax.minorticks_off()
        ax.set_title(n, fontsize=8.5, color=INK, loc="left")
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("train size (log)")
    for ax in axes[:, 0]:
        ax.set_ylabel("macro-F1")
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, ["train EN, test EN", "train HU, test HU"][:len(h)], loc="upper right", frameon=False, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def fig_scatter(rows, xkeys, ykey):
    fig, axes = plt.subplots(1, len(xkeys), figsize=(3.6 * len(xkeys), 3.0), sharey=True)
    for ax, xk in zip(np.atleast_1d(axes), xkeys):
        pts = [(r[xk], r[ykey], r["model"]) for r in rows if r.get(xk) is not None and r.get(ykey) is not None]
        if pts:
            xs, ys, ns = zip(*pts)
            ax.scatter(xs, ys, s=40, color=HU_COLOR, edgecolor="white", linewidth=1.5, zorder=2)
            for x, yv, n in pts:
                ax.annotate(n, (x, yv), xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK2)
        ax.margins(x=0.2, y=0.1)
        ax.set_xlabel(xk)
    np.atleast_1d(axes)[0].set_ylabel(f"{ykey} (EN - HU macro-F1)")
    return fig


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="results/report.html")
    ap.add_argument("--tokstats", default="results/tokstats.csv")
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    test, runs = load(cfg, args.runs)
    gold = np.array([r["label"] for r in test])
    aa = np.array([r["allagree"] for r in test])
    B = Boot(gold, aa, args.n_boot)
    models = cfg["models"]
    names = [m["name"] for m in models]
    dec = [m["name"] for m in models if m["kind"] == "decoder"]
    sizes = [str(s) for s in cfg["data"]["curve_sizes"]] + ["full"]
    n_exp = {"ft": len(models) * 2 * len(sizes) * len(cfg["seeds"]), "c0": 3 * len(dec), "bpc": len(dec)}
    n_got = {k: sum(r["task"]["type"] == k for r in runs.values()) for k in n_exp}

    def ft_cell(model, train, test_lang, size="full", subset="all"):
        P = [r["preds"][test_lang]["pred"] for r in find(runs, type="ft", model=model, train_lang=train, size=size)]
        return B.cell(P, subset)

    def c0_cell(model, test_lang, prompt_lang, key="pred"):
        P = [r["preds"][test_lang][key] for r in find(runs, type="c0", model=model, test_lang=test_lang,
                                                       prompt_lang=prompt_lang)]
        return B.cell(P)

    # H1: main table
    main_rows, dots_ft, dots_c0 = [], {}, {}
    for m in models:
        n = m["name"]
        ee, hh = ft_cell(n, "en", "en"), ft_cell(n, "hu", "hu")
        r = {"model": n, "kind": m["kind"], "EN>EN": ee, "HU>HU": hh, "gap": gap(ee, hh),
             # sensitivity check: median over seeds is robust to a single diverged training run
             "gap (median)": ee["median"] - hh["median"] if ee and hh else None,
             "gap_allagree": gap(ft_cell(n, "en", "en", subset="allagree"), ft_cell(n, "hu", "hu", subset="allagree")),
             "EN>HU": ft_cell(n, "en", "hu"), "HU>EN": ft_cell(n, "hu", "en"), "seeds": ee["n"] if ee else 0}
        if m["kind"] == "decoder":
            c_en, c_hu = c0_cell(n, "en", "en"), c0_cell(n, "hu", "hu")
            r.update({"C0 EN": c_en, "C0 HU": c_hu, "C0 gap": gap(c_en, c_hu), "C0 HU(EN prompt)": c0_cell(n, "hu", "en"),
                      "C0cal EN": c0_cell(n, "en", "en", "pred_cal"), "C0cal HU": c0_cell(n, "hu", "hu", "pred_cal")})
            dots_c0[n] = (c_en, c_hu)
        dots_ft[n] = (ee, hh)
        main_rows.append(r)
    main_cols = ["model", "kind", "seeds", "EN>EN", "HU>HU", "gap", "gap (median)", "gap_allagree", "EN>HU", "HU>EN",
                 "C0 EN", "C0 HU", "C0 gap", "C0 HU(EN prompt)", "C0cal EN", "C0cal HU"]

    # Calibration and per-class F1 (full train, in-language), mean over seeds
    cal_rows = []
    for n in names:
        r = {"model": n}
        for tr in ("en", "hu"):
            R = [x["res"]["test"][tr]["all"] for x in find(runs, type="ft", model=n, train_lang=tr, size="full")]
            if R:
                r[f"ECE {tr.upper()}>{tr.upper()}"] = float(np.mean([x["ece"] for x in R]))
                for c, lab in enumerate(("neg", "neu", "pos")):
                    r[f"F1 {lab} {tr.upper()}"] = float(np.mean([x["f1_per_class"][c] for x in R]))
        cal_rows.append(r)
    cal_cols = ["model", "ECE EN>EN", "ECE HU>HU", "F1 neg EN", "F1 neg HU", "F1 neu EN", "F1 neu HU",
                "F1 pos EN", "F1 pos HU"]

    # H3: learning curves and data efficiency
    curve_rows, eff_rows, curves = [], [], {}
    for n in names:
        for lang in ("en", "hu"):
            xs, ms, sds, r = [], [], [], {"model": n, "lang": f"{lang.upper()}>{lang.upper()}"}
            for s in sizes:
                R = find(runs, type="ft", model=n, train_lang=lang, size=s)
                if not R:
                    continue
                f1s = [x["res"]["test"][lang]["all"]["macro_f1"] for x in R]
                xs.append(R[0]["res"]["train_info"]["n_train"])
                ms.append(float(np.mean(f1s)))
                sds.append(float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0)
                r[s] = f"{ms[-1]:.3f} ±{sds[-1]:.3f}"
            if xs:
                curves.setdefault(n, {})[lang] = (xs, ms, sds)
            curve_rows.append(r)
        c = curves.get(n, {})
        if "en" in c and "hu" in c:
            r = {"model": n}
            for s in ("250", "1000"):
                if int(s) in c["en"][0]:
                    target = c["en"][1][c["en"][0].index(int(s))]
                    r[f"EN F1 @{s}"] = target
                    r[f"HU n for it (EN n={s})"] = needed(c["hu"][0], c["hu"][1], target)
            eff_rows.append(r)
    eff_cols = ["model", "EN F1 @250", "HU n for it (EN n=250)", "EN F1 @1000", "HU n for it (EN n=1000)"]

    # H2: correlations over decoders
    tok = {}
    if os.path.exists(args.tokstats):
        with open(args.tokstats, encoding="utf-8") as f:
            tok = {r["model"]: r for r in csv.DictReader(f)}
    h2_rows = []
    for n in dec:
        r = {"model": n}
        mr = next(x for x in main_rows if x["model"] == n)
        r["gap"] = mr["gap"]["mean"] if mr["gap"] else None
        r["gap (median)"] = mr["gap (median)"]
        r["C0 gap"] = mr["C0 gap"]["mean"] if mr.get("C0 gap") else None
        if n in tok:
            for k in ("hu_fertility", "hu_split_rate", "hu_en_tokens"):
                r[k] = float(tok[n][k])
        b = find(runs, type="bpc", model=n)
        if b:
            r["hu_bpc"], r["hu_en_bpc"] = b[0]["res"]["hu"]["bpc"], b[0]["res"]["hu_en_bpc"]
            if "hu_native" in b[0]["res"]:
                r["hu_native_bpc"] = b[0]["res"]["hu_native"]["bpc"]
        ft = find(runs, type="ft", model=n)
        if ft:
            ti = ft[0]["res"]["train_info"]
            r["nonemb_params_M"] = (ti["n_params"] - ti["n_embed"]) / 1e6
        h2_rows.append(r)
    h2_cols = ["model", "gap", "gap (median)", "C0 gap", "hu_fertility", "hu_split_rate", "hu_en_tokens", "hu_bpc", "hu_en_bpc",
               "hu_native_bpc", "nonemb_params_M"]
    corr_rows = []
    for yk in ("gap", "gap (median)", "C0 gap"):
        for xk in ("hu_fertility", "hu_split_rate", "hu_en_tokens", "hu_bpc", "hu_en_bpc", "hu_native_bpc"):
            pts = [(r[xk], r[yk], r["nonemb_params_M"]) for r in h2_rows
                   if all(r.get(k) is not None for k in (xk, yk, "nonemb_params_M"))]
            row = {"y": yk, "x": xk, "n": len(pts)}
            if len(pts) >= 4:
                x, y, z = map(np.array, zip(*pts))
                rho, p = stats.spearmanr(x, y)
                row.update({"spearman": float(rho), "p": float(p)})
                row["partial (size)"], row["p partial"] = partial_spearman(x, y, z)
            corr_rows.append(row)
    corr_cols = ["y", "x", "n", "spearman", "p", "partial (size)", "p partial"]

    # CSV outputs
    write_csv(os.path.join(out_dir, "summary_main.csv"), main_rows, main_cols)
    write_csv(os.path.join(out_dir, "summary_curve.csv"), curve_rows, ["model", "lang"] + sizes)
    write_csv(os.path.join(out_dir, "summary_corr.csv"), corr_rows, corr_cols)

    # Numeric data for the paper figures and tables (paper_figs.py); bootstrap arrays are dropped
    def num(v):
        if isinstance(v, dict):
            return {k: num(x) for k, x in v.items() if k != "boot"}
        return v.tolist() if isinstance(v, np.ndarray) else v
    with open(os.path.join(out_dir, "paper_data.json"), "w", encoding="utf-8") as f:
        json.dump({"main": [num(r) for r in main_rows], "h2": h2_rows, "corr": corr_rows, "curves": curves,
                   "eff": eff_rows, "n_test": int(len(gold)), "n_allagree": int(aa.sum())}, f, indent=1)

    # HTML
    status = [{"type": k, "done": n_got[k], "expected": n_exp[k]} for k in n_exp]
    parts = [
        "<h1>ICNLP 2027: FPB EN vs HU</h1>",
        f"<p class=note>Test set n={len(gold)} (AllAgree n={int(aa.sum())}). macro-F1, mean over seeds; "
        f"[95% bootstrap CI, {args.n_boot} resamples of test sentences, paired for gaps]. "
        "A&gt;B = train language A, test language B. Smoke-test runs are excluded.</p>",
        "<h2>Status</h2>", html_table(status, ["type", "done", "expected"]),
        "<h2>H1: EN vs HU</h2>",
        png(fig_dots(names, [("Fine-tuned, full train, in-language", dots_ft), ("Zero-shot C0 (raw)", dots_c0)])),
        html_table(main_rows, main_cols),
        "<h3>Calibration and per-class F1 (full train, in-language)</h3>", html_table(cal_rows, cal_cols),
        "<h2>H2: gap vs tokenizer quality and BPC (decoders)</h2>",
        png(fig_scatter(h2_rows, ["hu_fertility", "hu_bpc", "hu_native_bpc"], "gap")),
        html_table(h2_rows, h2_cols), html_table(corr_rows, corr_cols),
        f"<p class=note>n={len(dec)} decoders; Qwen3 x3 and Llama 3.2 x2 share a tokenizer, so tokenizer metrics have "
        "only 5 distinct values. Partial Spearman controls for non-embedding parameter count. Low power: treat "
        "p-values as descriptive.</p>",
        "<h2>H3: learning curves and data efficiency</h2>", png(fig_curves(names, curves)),
        html_table(curve_rows, ["model", "lang"] + sizes),
        "<h3>HU training size needed to match the EN F1 at a given EN size (log-linear interpolation)</h3>",
        html_table(eff_rows, eff_cols),
    ]
    css = ("body{font-family:system-ui,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;color:#0b0b0b;"
           "background:#fcfcfb}table{border-collapse:collapse;font-size:12px;margin:8px 0 20px;display:block;"
           "overflow-x:auto}th,td{border-bottom:1px solid #e5e4e0;padding:4px 8px;text-align:right;white-space:nowrap}"
           "th:first-child,td:first-child{text-align:left}th{color:#52514e;font-weight:600}img{max-width:100%}"
           ".note{color:#52514e;font-size:13px}")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"<!doctype html><html><head><meta charset=utf-8><title>FPB EN-HU report</title>"
                f"<style>{css}</style></head><body>{''.join(parts)}</body></html>")

    # Short summary that can be typed back by hand
    print("REPORT " + " ".join(f"{k}={n_got[k]}/{n_exp[k]}" for k in n_exp))
    for r in main_rows:
        if r["gap"]:
            print(f"H1 {r['model']} en={r['EN>EN']['mean']:.3f} hu={r['HU>HU']['mean']:.3f} gap={fmt(r['gap'])}")
    for r in corr_rows:
        if "spearman" in r:
            print(f"H2 {r['y']}~{r['x']} n={r['n']} rho={r['spearman']:.2f} p={r['p']:.2f} "
                  f"partial={r['partial (size)']:.2f}")
    for r in eff_rows:
        print(f"H3 {r['model']} " + " ".join(f"{k}={fmt(v)}" for k, v in r.items() if k != "model"))
    print(f"REPORT saved {args.out}")


if __name__ == "__main__":
    main()
