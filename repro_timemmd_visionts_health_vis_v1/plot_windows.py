#!/usr/bin/env python3
"""EXP-023 plot: 10 two-panel PNGs for the selected Health windows
(fixed H=12, best-5 + worst-5 by per-window MSE in the EXP-021 scaled
evaluation space).

Panel 1 (global locator): the full 1389-row Health OT series in
original units vs real dates, the test region shaded, and the current
window's history span / prediction span highlighted with vertical
lines at the prediction start/end.

Panel 2 (local detail): input history (blue), prediction-span truth
(orange, read directly from the CSV), model prediction (red dashed,
inverse-transformed from the frozen npz via orig = scaled*scale+mean),
and the post-horizon observation window (gray, rows [s+H, s+2H-1],
clipped at series end with an in-plot note).

Requires results/ from verify_and_select.py (gates must have passed).
"""
import csv
import datetime
import json
import os
import subprocess
import sys

import numpy as np

import anchors as A

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.dates as mdates

RESULTS = A.RESULTS_DIR
FIGS = A.FIG_DIR

LBL_ZH = {
    "lang": "zh",
    "global_title": "全局定位：Health 完整真实序列（原始单位）",
    "local_title": "局部细节：输入历史 / 预测区间真值 / 模型预测",
    "hist": "历史真值",
    "truth": "预测区间真值",
    "pred": "模型预测",
    "post": "后续观察区间",
    "post_clip": "后续观察区间（不足一个预测长度，已截断）",
    "post_none": "后续观察区间（序列末端，无）",
    "test_region": "测试区",
    "hist_span": "本窗历史段",
    "pred_span": "本窗预测段",
    "ylabel": "OT（原始单位）",
    "suptitle": "{kind}#{rank}/{total}  {wid}  |  预测期 {d0} .. {d1}  |  "
                "MSE={mse:.4f}  MAE={mae:.4f}  |  P={p}  L={l}  H={h}",
    "note_font": "注：中文字体",
}
LBL_EN = {
    "lang": "en",
    "global_title": "Global locator: Health full ground-truth series (original units)",
    "local_title": "Local detail: input history / prediction-span truth / model forecast",
    "hist": "History truth",
    "truth": "Prediction-span truth",
    "pred": "Model prediction",
    "post": "Post-horizon observations",
    "post_clip": "Post-horizon observations (truncated, < 1 horizon left)",
    "post_none": "Post-horizon observations (none, series end)",
    "test_region": "Test region",
    "hist_span": "This window: history span",
    "pred_span": "This window: prediction span",
    "ylabel": "OT (original units)",
    "suptitle": "{kind}#{rank}/{total}  {wid}  |  prediction {d0} .. {d1}  |  "
                "MSE={mse:.4f}  MAE={mae:.4f}  |  P={p}  L={l}  H={h}",
    "note_font": "",
}

CJK_MARKERS = ("Noto Sans CJK", "Noto Serif CJK", "WenQuanYi", "AR PL",
               "Source Han", "SimHei", "Microsoft YaHei", "SimSun",
               "Uming", "Ukai")


def setup_cjk():
    """Register a CJK font if one exists; return label dict (zh or en)."""
    try:
        out = subprocess.run(["fc-list", ":lang=zh", "file"],
                             capture_output=True, text=True, timeout=20)
        cand = []
        for line in out.stdout.splitlines():
            p = line.strip().rstrip(":")
            if p.lower().endswith((".ttf", ".otf", ".ttc")):
                cand.append(p)
        fams = []
        for p in cand[:12]:
            try:
                font_manager.fontManager.addfont(p)
                fam = font_manager.FontProperties(fname=p).get_name()
                if any(m in fam for m in CJK_MARKERS) and fam not in fams:
                    fams.append(fam)
            except Exception:
                continue
        if fams:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = fams + [
                "DejaVu Sans", "Arial"]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[plot] CJK fonts: {fams}")
            return LBL_ZH, fams
    except Exception as e:
        print(f"[plot] CJK probe failed ({e!r}); English labels")
    print("[plot] no CJK font found — falling back to English labels "
          "(documented in README)")
    plt.rcParams["axes.unicode_minus"] = False
    return LBL_EN, []


def to_days(date_strs):
    return [datetime.date.fromisoformat(d) for d in date_strs]


def main():
    for p in ("gates.json", "selection.json", "scaler.json",
              "health_window_metrics.csv"):
        if not os.path.isfile(os.path.join(RESULTS, p)):
            sys.exit(f"[plot] missing {p} — run verify_and_select.py first")
    gates = json.load(open(os.path.join(RESULTS, "gates.json"),
                           encoding="utf-8"))
    if not all(g["ok"] for g in gates["gates"]):
        sys.exit("[plot] gates.json contains failures — aborting")
    selection = json.load(open(os.path.join(RESULTS, "selection.json"),
                               encoding="utf-8"))
    scaler = json.load(open(os.path.join(RESULTS, "scaler.json"),
                            encoding="utf-8"))
    mean_ot = scaler["mean_ot"]
    scale_ot = scaler["scale_ot"]

    dates, ot = [], []
    with open(A.CSV_PATH, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            dates.append(row["date"])
            ot.append(float(row["OT"]))
    if len(ot) != A.N_ROWS:
        sys.exit(f"[plot] CSV rows {len(ot)} != {A.N_ROWS}")
    all_x = to_days(dates)
    ot = np.asarray(ot)

    LBL, fams = setup_cjk()

    z12 = np.load(A.npz_path(A.SEL_H), allow_pickle=False)
    row_to_idx = {int(s): i for i, s in
                  enumerate(z12["row_pred_start"].tolist())}

    os.makedirs(FIGS, exist_ok=True)
    kind_order = [("best", i) for i in range(1, A.K + 1)] + \
                 [("worst", i) for i in range(1, A.K + 1)]
    by_key = {(w["kind"], w["rank"]): w for w in selection["windows"]}
    total = A.WINDOWS[A.SEL_H]
    manifest = []
    for kind, rank in kind_order:
        w = by_key[(kind, rank)]
        s = w["row_pred_start"]
        H = w["pred_len"]
        i = row_to_idx[s]
        d0, d1 = w["date_pred_start"], w["date_pred_end"]

        fig = plt.figure(figsize=(13.0, 9.0))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.6],
                              hspace=0.28)

        # ---- global locator ----
        ax = fig.add_subplot(gs[0])
        ax.plot(all_x, ot, color="#3a3a3a", lw=0.9)
        tx0, tx1 = to_days([A.DATES["test_first"], A.DATES["test_last"]])
        ax.axvspan(tx0, tx1, color=A.COLOR_TEST_BG, alpha=0.18,
                   label=LBL["test_region"])
        hx0, hx1 = to_days([dates[s - A.SEQ_LEN], dates[s - 1]])
        ax.axvspan(hx0, hx1, color=A.COLOR_HIST, alpha=0.28,
                   label=LBL["hist_span"])
        px0, px1 = to_days([d0, d1])
        ax.axvspan(px0, px1, color=A.COLOR_TRUTH, alpha=0.38,
                   label=LBL["pred_span"])
        for xv in (px0, px1):
            ax.axvline(xv, color="black", ls="--", lw=0.8)
        ax.set_title(LBL["global_title"], fontsize=10)
        ax.set_ylabel(LBL["ylabel"], fontsize=9)
        ax.legend(loc="upper left", fontsize=7, ncol=3, framealpha=0.85)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(labelsize=8)

        # ---- local detail ----
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(all_x[s - A.SEQ_LEN:s], ot[s - A.SEQ_LEN:s],
                 color=A.COLOR_HIST, lw=1.2, label=LBL["hist"])
        ax2.plot(all_x[s:s + H], ot[s:s + H], color=A.COLOR_TRUTH,
                 lw=1.7, marker="o", ms=3.0, label=LBL["truth"])
        pred_orig = z12["pred"][i, :, 0].astype(np.float64) \
            * scale_ot + mean_ot
        ax2.plot(all_x[s:s + H], pred_orig, color=A.COLOR_PRED,
                 lw=1.5, ls="--", marker="s", ms=3.0, label=LBL["pred"])
        p0, p1 = s + H, min(s + 2 * H, A.N_ROWS)
        if p0 >= A.N_ROWS:
            ax2.text(0.02, 0.04, LBL["post_none"], transform=ax2.transAxes,
                     fontsize=8, color=A.COLOR_POST)
        else:
            ax2.plot(all_x[p0:p1], ot[p0:p1], color=A.COLOR_POST,
                     lw=1.2, label=LBL["post"] if p1 - p0 == H
                     else LBL["post_clip"])
        for xv in (px0, px1):
            ax2.axvline(xv, color="black", ls="--", lw=0.8)
        ax2.axvline(px0, color=A.COLOR_PRED, ls=":", lw=0.8, alpha=0.6)
        ax2.set_title(LBL["local_title"], fontsize=10)
        ax2.set_ylabel(LBL["ylabel"], fontsize=9)
        ax2.set_xlabel("date", fontsize=9)
        ax2.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.85)
        ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0,
                                                          interval=8))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax2.tick_params(labelsize=8)

        sup = LBL["suptitle"].format(
            kind=kind, rank=rank, total=total, wid=w["window_id"],
            d0=d0, d1=d1, mse=w["mse"], mae=w["mae"], p=A.SELECTED_P,
            l=A.SEQ_LEN, h=H)
        fig.suptitle(sup, fontsize=11, y=0.98)
        fname = (f"Health_{kind}{rank}_h{H}_s{s:04d}_{d0}.png")
        out = os.path.join(FIGS, fname)
        fig.savefig(out, dpi=A.DPI)
        plt.close(fig)
        manifest.append({"figure": fname, "kind": kind, "rank": rank,
                         "window_id": w["window_id"], "row_pred_start": s,
                         "date_pred_start": d0, "date_pred_end": d1,
                         "mse": w["mse"], "mae": w["mae"]})
        print(f"[plot] {fname}")

    with open(os.path.join(FIGS, "plot_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump({"labels": LBL["lang"], "cjk_fonts": fams,
                   "figures": manifest}, f, indent=1, ensure_ascii=False)
    if len(manifest) != 10:
        sys.exit(f"[plot] expected 10 figures, wrote {len(manifest)}")
    print("PLOT OK")


if __name__ == "__main__":
    main()
