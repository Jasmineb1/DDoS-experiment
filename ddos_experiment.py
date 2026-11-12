#!/usr/bin/env python3
"""
ML-based DDoS detection experiment on CIC-DDoS2019
===================================================
Advanced Network Engineering (12094 PG) - Research Paper experiment

What this script does
---------------------
1. Loads the two capture days of CIC-DDoS2019 (CSV files). It keeps ALL
   benign flows and a random sample of attack flows from every file, so the
   data fits in memory.
2. Trains four detectors:
     - Threshold : a simple packets-per-second rule (non-ML baseline)
     - LR        : logistic regression
     - DT        : decision tree
     - RF        : random forest
3. Tests them in two ways:
     - "Random split"  : 70/30 random split of day 1 (how most papers test)
     - "Next day"      : train on day 1, test on day 2 (closer to real use;
                         day 2 also contains attack types never seen on day 1)
4. Groups the day-2 predictions into time windows (1 s ... 60 s) and measures
   false alarms per attack-free day and detection delay.
5. Saves everything to the results folder:
     results/fig_split_comparison.png   -> Fig. 2 in the paper
     results/fig_per_attack_recall.png  -> Fig. 3 in the paper
     results/fig_window_tradeoff.png    -> Fig. 4 in the paper
     results/fig_feature_importance.png -> Fig. 5 (Appendix A)
     results/results_table.tex          -> rows of Table III (auto-included)
     results/paper_values.tex           -> numbers used in the text (auto-included)
     results/*.csv                      -> raw numbers for your own checking
     results/results_summary.txt        -> plain-text summary to read

How to run
----------
Kaggle download (zip or unzipped folder):
    python ddos_experiment.py --data archive.zip
Original UNB CSVs:
    python ddos_experiment.py --day1 CSV-01-12 --day2 CSV-03-11
See README.md for full steps.

Runtime: roughly 20-60 minutes in Colab, most of it reading the large CSVs.
Use --attack-frac 0.02 to make it faster (smaller attack sample).
"""

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

# ---------------------------------------------------------------------------
# Settings (can also be changed with command-line options)
# ---------------------------------------------------------------------------
SEED = 42
CHUNK_ROWS = 250_000
WINDOWS_S = [1, 2, 5, 10, 30, 60]       # window lengths for the window analysis
ALARM_MIN_FLOWS = [1, 5]                 # a window raises an alarm if >= k flows are flagged
SECONDS_PER_DAY = 86_400

# Identifier columns are removed so the model cannot "cheat" by learning
# specific IP addresses, ports or row numbers from the lab test bed.
DROP_COLS = {
    "Unnamed: 0", "Flow ID", "Source IP", "Source Port", "Destination IP",
    "Destination Port", "Timestamp", "SimillarHTTP", "Inbound", "Label",
}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 150, "savefig.dpi": 300, "font.family": "serif",
})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def clean_label(raw: str) -> str:
    """Make labels from both days comparable, e.g. 'DrDoS_DNS' -> 'DNS'."""
    s = str(raw).strip()
    if s.upper() == "BENIGN":
        return "BENIGN"
    s = s.replace("DrDoS_", "").replace("DRDOS_", "")
    s = s.replace("-", "").replace("_", "").upper()
    return s


def _split_chunk(chunk, rng, attack_frac):
    chunk.columns = [str(c).strip() for c in chunk.columns]
    chunk = chunk.loc[:, ~pd.Index(chunk.columns).duplicated()]
    is_benign = chunk["Label"].astype(str).str.strip().str.upper() == "BENIGN"
    att = chunk[~is_benign]
    if len(att):
        att = att[rng.random(len(att)) < attack_frac]
    return chunk[is_benign], att


def load_day(folder: str, attack_frac: float, attack_cap: int, tag: str) -> pd.DataFrame:
    """Reads every .csv (original UNB files) or .parquet (Kaggle copy) file in a folder."""
    files = sorted(glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True) +
                   glob.glob(os.path.join(folder, "**", "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"No .csv or .parquet files found under {folder}. Check the --{tag} path.")
    rng = np.random.default_rng(SEED)
    parts = []
    for path in files:
        t0 = time.time()
        benign_parts, attack_parts, n_rows = [], [], 0
        if path.endswith(".parquet"):
            chunks = [pd.read_parquet(path)]
        else:
            chunks = pd.read_csv(path, chunksize=CHUNK_ROWS, low_memory=False,
                                 encoding_errors="ignore")
        for chunk in chunks:
            n_rows += len(chunk)
            b, a = _split_chunk(chunk, rng, attack_frac)
            benign_parts.append(b)
            attack_parts.append(a)
        benign = pd.concat(benign_parts) if benign_parts else pd.DataFrame()
        attack = pd.concat(attack_parts) if attack_parts else pd.DataFrame()
        if len(attack) > attack_cap:
            attack = attack.sample(attack_cap, random_state=SEED)
        parts.append(pd.concat([benign, attack], ignore_index=True))
        print(f"  [{tag}] {os.path.basename(path):<26} rows={n_rows:>10,}  "
              f"kept benign={len(benign):>7,} attack={len(attack):>7,}  "
              f"({time.time() - t0:.0f}s)")
    data = pd.concat(parts, ignore_index=True)
    data["label_clean"] = data["Label"].map(clean_label)
    data["y"] = (data["label_clean"] != "BENIGN").astype(int)
    if "Timestamp" in data.columns:
        data["ts"] = pd.to_datetime(data["Timestamp"], errors="coerce")
    else:
        data["ts"] = pd.NaT
        print(f"  [{tag}] NOTE: no Timestamp column, so the window analysis will be skipped.")
    return data


def prepare_kaggle(path: str):
    """Accepts the Kaggle zip or folder; sorts *-training files into day 1 and
    *-testing files into day 2 (using links, so nothing large is copied)."""
    import zipfile
    path = os.path.abspath(os.path.expanduser(path))
    if path.lower().endswith(".zip"):
        folder = os.path.splitext(path)[0] + "_unzipped"
        if not os.path.isdir(folder):
            print(f"Unzipping {path} ...")
            with zipfile.ZipFile(path) as z:
                z.extractall(folder)
        path = folder
    if not os.path.isdir(path):
        raise SystemExit(f"Cannot find {path}")
    files = glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True) + \
        glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    split_root = os.path.join(os.path.dirname(path.rstrip(os.sep)), "ddos_split")
    d1, d2 = os.path.join(split_root, "day1"), os.path.join(split_root, "day2")
    for d in (d1, d2):
        os.makedirs(d, exist_ok=True)
        for old in glob.glob(os.path.join(d, "*")):
            os.remove(old)
    n1 = n2 = 0
    for f in files:
        name = os.path.basename(f).lower()
        if "training" in name:
            dest, n1 = d1, n1 + 1
        elif "testing" in name:
            dest, n2 = d2, n2 + 1
        else:
            print(f"  skipping {os.path.basename(f)} (name has neither 'training' nor 'testing')")
            continue
        target = os.path.join(dest, os.path.basename(f))
        try:
            os.symlink(f, target)
        except OSError:  # e.g. Windows without symlink permission
            import shutil
            shutil.copy(f, target)
    if not n1 or not n2:
        raise SystemExit(f"Found {n1} training and {n2} testing files in {path}. "
                         "Check that you downloaded the full Kaggle dataset.")
    print(f"Kaggle data: {n1} training files -> day 1, {n2} testing files -> day 2")
    return d1, d2


def build_features(train: pd.DataFrame, others: list):
    """Select numeric flow features, clean infinities/NaNs, drop constant columns."""
    cols = [c for c in train.columns
            if c not in DROP_COLS and c not in ("label_clean", "y", "ts")]
    X_tr = train[cols].apply(pd.to_numeric, errors="coerce")
    X_tr = X_tr.replace([np.inf, -np.inf], np.nan)
    keep = [c for c in X_tr.columns if X_tr[c].notna().any() and X_tr[c].nunique() > 1]
    X_tr = X_tr[keep]
    med = X_tr.median()
    X_tr = X_tr.fillna(med)
    outs = []
    for df in others:
        X = df.reindex(columns=keep).apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(med)
        outs.append(X)
    return X_tr, outs, keep


# ---------------------------------------------------------------------------
# Models and metrics
# ---------------------------------------------------------------------------
class ThresholdDetector:
    """Non-ML baseline: flag a flow if its packet rate is above the 99th
    percentile of benign packet rates seen in training."""

    def __init__(self, feature="Flow Packets/s", q=0.99):
        self.feature, self.q, self.tau = feature, q, None

    def fit(self, X, y):
        self.tau = float(np.quantile(X.loc[y == 0, self.feature], self.q))
        return self

    def predict(self, X):
        return (X[self.feature].values > self.tau).astype(int)


def make_models(feature_names):
    models = {
        "LR": make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced")),
        "DT": DecisionTreeClassifier(max_depth=12, class_weight="balanced",
                                     random_state=SEED),
        "RF": RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                     class_weight="balanced_subsample",
                                     random_state=SEED),
    }
    if "Flow Packets/s" in feature_names:
        models = {"Threshold": ThresholdDetector(), **models}
    return models


def metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, fpr=fpr,
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


# ---------------------------------------------------------------------------
# Window analysis
# ---------------------------------------------------------------------------
def window_analysis(test_df: pd.DataFrame, y_pred: np.ndarray):
    d = test_df[["ts", "y", "label_clean"]].copy()
    d["pred"] = y_pred
    d = d.dropna(subset=["ts"])
    if d.empty:
        return pd.DataFrame()
    t0 = d["ts"].min()
    d["sec"] = (d["ts"] - t0).dt.total_seconds()
    rows = []
    for dt in WINDOWS_S:
        d["win"] = np.floor(d["sec"] / dt).astype(int)
        g = d.groupby("win").agg(n=("y", "size"), n_att=("y", "sum"), n_pred=("pred", "sum"))
        benign_w = g[g["n_att"] == 0]
        for k in ALARM_MIN_FLOWS:
            alarm = g["n_pred"] >= k
            fpr_w = float((benign_w["n_pred"] >= k).mean()) if len(benign_w) else np.nan
            fa_day = SECONDS_PER_DAY / dt * fpr_w if not np.isnan(fpr_w) else np.nan
            # detection delay: for each attack type, time from its first flow
            # to the end of the first alarmed window at or after that time
            delays = []
            for lab, grp in d[d["y"] == 1].groupby("label_clean"):
                start = grp["sec"].min()
                w0 = int(np.floor(start / dt))
                hit = g.index[(g.index >= w0) & alarm.values]
                if len(hit):
                    delays.append((hit[0] + 1) * dt - start)
            rows.append(dict(window_s=dt, k=k, benign_windows=len(benign_w),
                             window_fpr=fpr_w, false_alarms_per_day=fa_day,
                             median_delay_s=float(np.median(delays)) if delays else np.nan,
                             max_delay_s=float(np.max(delays)) if delays else np.nan,
                             attacks_detected=len(delays),
                             attacks_total=d.loc[d["y"] == 1, "label_clean"].nunique()))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_split_comparison(res: pd.DataFrame, out):
    models = list(res["model"].unique())
    x = np.arange(len(models))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.4))
    for ax, metric, title in [(axes[0], "f1", "F1-score (higher = better)"),
                              (axes[1], "fpr", "FPR (lower = better)")]:
        for i, (scen, hatch) in enumerate([("Random split", ""), ("Next day", "///")]):
            vals = [res[(res.model == m) & (res.scenario == scen)][metric].values[0] for m in models]
            ax.bar(x + (i - 0.5) * w, vals, w, label=scen, color="white",
                   edgecolor="black", hatch=hatch, linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0, 1.05)
    axes[1].set_yscale("symlog", linthresh=1e-3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(out, "fig_split_comparison.png"), bbox_inches="tight")
    plt.close(fig)


def fig_per_attack(per_att: pd.DataFrame, model: str, out):
    d = per_att[per_att.model == model].sort_values("recall")
    fig, ax = plt.subplots(figsize=(3.5, 0.9 + 0.22 * len(d)))
    colors = ["lightgray" if s else "black" for s in d["seen_in_training"]]
    ax.barh(d["attack"], d["recall"], color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(f"Recall on day 2 ({model})")
    ax.grid(axis="x", alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="lightgray", edgecolor="black", label="Seen in training"),
                       Patch(facecolor="black", edgecolor="black", label="Unseen (new on day 2)")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_per_attack_recall.png"), bbox_inches="tight")
    plt.close(fig)


def fig_window(win: pd.DataFrame, out):
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.3))
    styles = {1: "-o", 5: "--s"}
    for k in ALARM_MIN_FLOWS:
        d = win[win.k == k]
        axes[0].plot(d.window_s, d.false_alarms_per_day.clip(lower=1e-2), styles.get(k, "-"),
                     color="black", ms=3, label=f"alarm if $\\geq${k} flow(s)")
        axes[1].plot(d.window_s, d.median_delay_s, styles.get(k, "-"), color="black", ms=3,
                     label=f"$\\geq${k}")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("Window $\\Delta t$ (s)"); axes[0].set_ylabel("False alarms per day")
    axes[0].axhline(1, color="gray", lw=0.6, ls=":")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Window $\\Delta t$ (s)"); axes[1].set_ylabel("Median delay (s)")
    for ax in axes:
        ax.grid(alpha=0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(out, "fig_window_tradeoff.png"), bbox_inches="tight")
    plt.close(fig)


def fig_importance(rf, names, out, top=15):
    imp = pd.Series(rf.feature_importances_, index=names).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.barh(imp.index, imp.values, color="gray", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Random forest feature importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_feature_importance.png"), bbox_inches="tight")
    plt.close(fig)
    return imp.sort_values(ascending=False)


# ---------------------------------------------------------------------------
# LaTeX outputs
# ---------------------------------------------------------------------------
def f3(v):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def write_latex(res, per_att, win, info, out):
    # Table III rows
    lines = []
    for scen in ["Random split", "Next day"]:
        for _, r in res[res.scenario == scen].iterrows():
            lines.append(f"{scen} & {r.model} & {f3(r.precision)} & {f3(r.recall)} & "
                         f"{f3(r.f1)} & {r.fpr:.4f} \\\\")
        if scen == "Random split":
            lines.append("\\midrule")
    with open(os.path.join(out, "results_table.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    def get(scen, model, m):
        d = res[(res.scenario == scen) & (res.model == model)]
        return d[m].values[0] if len(d) else np.nan

    def win_get(dt, k, col):
        if not len(win):
            return np.nan
        d = win[(win.window_s == dt) & (win.k == k)]
        return d[col].values[0] if len(d) else np.nan

    rf_att = per_att[per_att.model == "RF"]
    unseen = rf_att[~rf_att.seen_in_training]
    vals = {
        "NTrain": f"{info['n_train']:,}", "NTest": f"{info['n_test']:,}",
        "NFeatures": str(info["n_features"]),
        "NBenignTrain": f"{info['n_benign_train']:,}", "NBenignTest": f"{info['n_benign_test']:,}",
        "RFRandFone": f3(get("Random split", "RF", "f1")),
        "RFNextFone": f3(get("Next day", "RF", "f1")),
        "RFRandFPR": f"{get('Random split', 'RF', 'fpr'):.4f}",
        "RFNextFPR": f"{get('Next day', 'RF', 'fpr'):.4f}",
        "LRNextFone": f3(get("Next day", "LR", "f1")),
        "DTNextFone": f3(get("Next day", "DT", "f1")),
        "THNextFone": f3(get("Next day", "Threshold", "f1")),
        "THNextFPR": f"{get('Next day', 'Threshold', 'fpr'):.4f}",
        "UnseenNames": ", ".join(unseen.attack) if len(unseen) else "none",
        "UnseenRecall": f3(unseen.recall.mean()) if len(unseen) else "--",
        "SeenRecall": f3(rf_att[rf_att.seen_in_training].recall.mean()),
        "FAdayOne": f"{win_get(1, 1, 'false_alarms_per_day'):,.0f}",
        "FAdayTen": f"{win_get(10, 1, 'false_alarms_per_day'):,.0f}",
        "FAdaySixty": f"{win_get(60, 1, 'false_alarms_per_day'):,.0f}",
        "FAdayOneKfive": f"{win_get(1, 5, 'false_alarms_per_day'):,.0f}",
        "DelayOne": f"{win_get(1, 1, 'median_delay_s'):.1f}",
        "DelayTen": f"{win_get(10, 1, 'median_delay_s'):.1f}",
        "DelaySixty": f"{win_get(60, 1, 'median_delay_s'):.1f}",
    }
    with open(os.path.join(out, "paper_values.tex"), "w") as fh:
        fh.write("% Auto-generated by ddos_experiment.py - do not edit by hand\n")
        for k, v in vals.items():
            fh.write(f"\\renewcommand{{\\{k}}}{{{v}}}\n")
        fh.write("\\HasWindowtrue\n" if len(win) else "\\HasWindowfalse\n")
    return vals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="Kaggle download: the zip file (e.g. archive.zip) or the "
                                   "unzipped folder of .parquet files. Day 1/day 2 are "
                                   "split automatically from the file names.")
    ap.add_argument("--day1", help="folder with day-1 files (original CSV-01-12)")
    ap.add_argument("--day2", help="folder with day-2 files (original CSV-03-11)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--attack-frac", type=float, default=None,
                    help="fraction of attack flows to sample per file "
                         "(default 1.0 for Kaggle parquet, 0.05 for original CSVs)")
    ap.add_argument("--attack-cap", type=int, default=40_000,
                    help="max attack flows kept per file (default 40000)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.data:
        args.day1, args.day2 = prepare_kaggle(args.data)
    elif not (args.day1 and args.day2):
        raise SystemExit("Give either --data (Kaggle download) or both --day1 and --day2.")
    if args.attack_frac is None:
        args.attack_frac = 1.0 if args.data else 0.05
    np.random.seed(SEED)

    print("Loading day 1 ...")
    day1 = load_day(args.day1, args.attack_frac, args.attack_cap, "day1")
    print("Loading day 2 ...")
    day2 = load_day(args.day2, args.attack_frac, args.attack_cap, "day2")
    print(f"Day 1: {len(day1):,} flows ({(day1.y == 0).sum():,} benign) | "
          f"Day 2: {len(day2):,} flows ({(day2.y == 0).sum():,} benign)")

    # ----- Scenario A: random split of day 1 ------------------------------
    tr_a, te_a = train_test_split(day1, test_size=0.3, stratify=day1["y"], random_state=SEED)
    Xa_tr, (Xa_te,), feats_a = build_features(tr_a, [te_a])
    # ----- Scenario B: train day 1, test day 2 ----------------------------
    Xb_tr, (Xb_te,), feats_b = build_features(day1, [day2])

    results, per_attack, fitted_b = [], [], {}
    for scen, Xtr, ytr, Xte, te_df, feats in [
        ("Random split", Xa_tr, tr_a.y.values, Xa_te, te_a, feats_a),
        ("Next day", Xb_tr, day1.y.values, Xb_te, day2, feats_b),
    ]:
        for name, model in make_models(feats).items():
            t0 = time.time()
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            m = metrics(te_df.y.values, pred)
            m.update(model=name, scenario=scen, seconds=time.time() - t0)
            results.append(m)
            print(f"  {scen:<12} {name:<9} F1={m['f1']:.3f}  FPR={m['fpr']:.4f}  "
                  f"recall={m['recall']:.3f}  ({m['seconds']:.0f}s)")
            if scen == "Next day":
                fitted_b[name] = (model, pred)
                seen = set(day1.loc[day1.y == 1, "label_clean"])
                for lab, grp in te_df.assign(pred=pred)[te_df.y == 1].groupby("label_clean"):
                    per_attack.append(dict(model=name, attack=lab, n=len(grp),
                                           recall=grp["pred"].mean(),
                                           seen_in_training=lab in seen))

    res = pd.DataFrame(results)
    per_att = pd.DataFrame(per_attack)
    res.to_csv(os.path.join(args.out, "table_main_results.csv"), index=False)
    per_att.to_csv(os.path.join(args.out, "table_per_attack_recall.csv"), index=False)

    # ----- Window analysis on day 2 (random forest) -----------------------
    win = window_analysis(day2, fitted_b["RF"][1])
    win.to_csv(os.path.join(args.out, "table_window_analysis.csv"), index=False)

    # ----- Figures --------------------------------------------------------
    fig_split_comparison(res, args.out)
    fig_per_attack(per_att, "RF", args.out)
    if len(win):
        fig_window(win, args.out)
    imp = fig_importance(fitted_b["RF"][0], feats_b, args.out)
    imp.to_csv(os.path.join(args.out, "table_feature_importance.csv"), header=["importance"])

    info = dict(n_train=len(day1), n_test=len(day2), n_features=len(feats_b),
                n_benign_train=int((day1.y == 0).sum()), n_benign_test=int((day2.y == 0).sum()))
    vals = write_latex(res, per_att, win, info, args.out)

    with open(os.path.join(args.out, "results_summary.txt"), "w") as fh:
        fh.write("MAIN RESULTS\n" + res[["scenario", "model", "precision", "recall", "f1",
                                         "fpr"]].to_string(index=False))
        fh.write("\n\nPER-ATTACK RECALL ON DAY 2\n" + per_att.to_string(index=False))
        fh.write("\n\nWINDOW ANALYSIS (RF, day 2)\n" + win.to_string(index=False))
        fh.write("\n\nTOP FEATURES (RF)\n" + imp.to_string())
        fh.write("\n\nVALUES WRITTEN TO paper_values.tex\n")
        for k, v in vals.items():
            fh.write(f"  {k:<14} {v}\n")
        if len(win) and win["benign_windows"].min() < 30:
            fh.write("\nWARNING: fewer than 30 attack-free windows for some window sizes. "
                     "Treat the false-alarm numbers for long windows with caution.\n")
    print(f"\nDone. Results are in '{args.out}/'. Read results_summary.txt first.")


if __name__ == "__main__":
    main()
