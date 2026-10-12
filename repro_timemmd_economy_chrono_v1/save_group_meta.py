#!/usr/bin/env python3
"""Assemble one group's outputs after the official entry finished.

Inputs : preds.npy/trues.npy written by the patched official exp_main.py
         (TMMD_PRED_DIR), plus the group's captured log.
Checks : window count == protocol pin; shapes/finite (NaN/Inf hard stop);
         official float32 metric replica of the saved arrays vs the
         metrics printed by the official entry (rtol=atol=1e-12 — same
         computation, so equality is expected; float64 is reported but
         carries no gate);
         trues are bitwise the scaler-transformed OT column:
         trues[i,t,0] == float32((OT_raw[border1+seq_len+i+t] - mean)/scale)
         with scaler stats from the rebuilt official dataset;
         window dates match the sorted CSV's date column row-for-row.
Outputs: predictions/{DS}__h{H}.npz (pred/true float64, row_pred_start,
         date_pred_start) + {DS}__h{H}.done.json bound to the protocol
         sha256, the actual batch size and the npz sha256;
         results/split_report.json (written once, then validated
         identically across groups).
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C
import protocol as P


def build_dataset(ds, pred_len):
    """Rebuild the official test Dataset (same CSV/args) for identity."""
    os.chdir(C.OFFICIAL_TMM)
    if C.OFFICIAL_TMM not in sys.path:
        sys.path.insert(0, C.OFFICIAL_TMM)
    from types import SimpleNamespace
    from data_provider.data_loader import Dataset_TimeMMD
    cfg = P.DATASETS[ds]
    args = SimpleNamespace(embed="timeF", few_shot_ratio=0.1)
    dset = Dataset_TimeMMD(
        args=args, root_path=C.DATASET_DIR, flag="test",
        size=[cfg["seq_len"], P.LABEL_LEN, pred_len],
        features=P.FEATURES, data_path=f"{ds}.csv", target=P.TARGET,
        scale=True, timeenc=1, freq="h")
    return dset


def read_ot_and_dates():
    di, oi = None, None
    dates, ot = [], []
    with open(P.csv_path("Economy"), newline="") as f:
        r = csv.reader(f)
        header = next(r)
        di, oi = header.index("date"), header.index("OT")
        for row in r:
            dates.append(row[di])
            ot.append(float(row[oi]))
    return dates, np.asarray(ot, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--pred-len", type=int, required=True)
    ap.add_argument("--stage-dir", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--protocol-token", required=True)
    ap.add_argument("--out-name", default=None,
                    help="write npz/done under this name instead of "
                         "{DS}__h{H} (sandbox use)")
    ap.add_argument("--pred-root", default=None,
                    help="override output dir (run_all passes its "
                         "--pred-root so sandbox runs stay in the sandbox)")
    a = ap.parse_args()
    out_dir = os.path.abspath(a.pred_root) if a.pred_root else C.PRED_DIR
    os.makedirs(out_dir, exist_ok=True)

    proto = P.verify_frozen(a.protocol_token)
    proto_sha = C.sha256_file(C.PROTO_PATH)
    cfg = proto["datasets"][a.ds]
    want_n = cfg["windows"][str(a.pred_len)]

    preds = np.load(os.path.join(a.stage_dir, "preds.npy"))
    trues = np.load(os.path.join(a.stage_dir, "trues.npy"))
    if preds.shape != trues.shape:
        sys.exit(f"[save-meta] shape mismatch preds {preds.shape} vs "
                 f"trues {trues.shape}")
    n = preds.shape[0]
    if n != want_n:
        sys.exit(f"[save-meta] window count {n} != expected {want_n} "
                 f"({a.ds} H={a.pred_len})")
    if preds.ndim != 3 or preds.shape[1] != a.pred_len or preds.shape[2] != 1:
        sys.exit(f"[save-meta] unexpected pred shape {preds.shape} for "
                 f"H={a.pred_len}")
    if not (np.isfinite(preds).all() and np.isfinite(trues).all()):
        sys.exit(f"[save-meta] NaN/Inf in outputs for {a.ds} "
                 f"H={a.pred_len} -> HARD STOP")

    printed = C.parse_printed_metrics(open(a.log, errors="replace").read())
    if printed is None:
        sys.exit(f"[save-meta] could not parse official metrics from {a.log}")
    mae32, mse32 = C.official_metric_replica(preds, trues)
    if not (C.close(mse32, printed["mse"]) and C.close(mae32, printed["mae"])):
        sys.exit(f"[save-meta] recompute gate FAIL: replica "
                 f"mse={mse32!r} mae={mae32!r} vs official "
                 f"mse={printed['mse']!r} mae={printed['mae']!r}")
    mae64, mse64 = C.metrics_float64(preds, trues)

    dset = build_dataset(a.ds, a.pred_len)
    if len(dset) != want_n:
        sys.exit(f"[save-meta] dataset len {len(dset)} != expected {want_n}")
    border1 = dset.border1
    seq_len = cfg["seq_len"]
    idx = np.arange(n, dtype=np.int64)
    row_pred_start = border1 + seq_len + idx
    dates = np.array([str(x[0]) for x in dset.date[seq_len:seq_len + n]])

    # trues must be bitwise the scaler-transformed OT rows the windows
    # target (loader: float64 transform -> batch_y.float() -> float32)
    csv_dates, ot_raw = read_ot_and_dates()
    rows2d = row_pred_start[:, None] + np.arange(a.pred_len, dtype=np.int64)[None, :]
    mean_ = float(dset.scaler.mean_[0])
    scale_ = float(dset.scaler.scale_[0])
    expected32 = ((ot_raw[rows2d] - mean_) / scale_).astype(np.float32)[..., None]
    if trues.dtype != np.float32:
        sys.exit(f"[save-meta] trues dtype {trues.dtype} != float32")
    if not np.array_equal(trues, expected32):
        bad = np.argwhere(trues != expected32)
        i0, t0 = (int(bad[0][0]), int(bad[0][1]))
        sys.exit(f"[save-meta] trues != scaler(OT) at {len(bad)} cells "
                 f"(first window {i0}, t={t0}: trues={trues[i0, t0, 0]!r} "
                 f"expected={expected32[i0, t0, 0]!r}) -> HARD STOP")
    # window dates must equal the sorted CSV's date column row-for-row
    for i in range(n):
        if dates[i] != csv_dates[row_pred_start[i]]:
            sys.exit(f"[save-meta] date identity FAIL window {i}: "
                     f"{dates[i]!r} != CSV row {row_pred_start[i]} "
                     f"{csv_dates[row_pred_start[i]]!r} -> HARD STOP")

    # split report: written once, validated identical by later groups
    sc = P.SPLIT_CLAIMS
    split_rep = {
        "n_rows": len(csv_dates),
        "num_train": sc["num_train"], "num_val": sc["num_val"],
        "num_test": sc["num_test"],
        "test_border1": int(border1), "test_border2": int(dset.border2),
        "scaler_mean_ot": mean_, "scaler_scale_ot": scale_,
        "train_dates": [csv_dates[0], csv_dates[sc["num_train"] - 1]],
        "val_dates": [csv_dates[sc["num_train"]],
                      csv_dates[sc["num_train"] + sc["num_val"] - 1]],
        "test_dates": [csv_dates[sc["num_train"] + sc["num_val"]],
                       csv_dates[-1]],
        "first_test_pred_row": int(row_pred_start[0]),
    }
    for k, want in (("n_rows", sc["n_rows"]), ("num_train", sc["num_train"]),
                    ("num_val", sc["num_val"]), ("num_test", sc["num_test"]),
                    ("test_border1", sc["test_border1"]),
                    ("first_test_pred_row", sc["first_test_pred_row"])):
        if split_rep[k] != want:
            sys.exit(f"[save-meta] split claim {k}={split_rep[k]} != "
                     f"protocol {want} -> HARD STOP")
    for k, want in (("train_dates", sc["train_dates"]),
                    ("val_dates", sc["val_dates"]),
                    ("test_dates", sc["test_dates"])):
        if split_rep[k] != want:
            sys.exit(f"[save-meta] split claim {k}={split_rep[k]} != "
                     f"protocol {want} -> HARD STOP")
    split_path = os.path.join(C.RESULTS_DIR, "split_report.json")
    if os.path.isfile(split_path):
        prev = C.load_json(split_path)
        if prev != split_rep:
            sys.exit(f"[save-meta] split_report drift across groups -> "
                     f"HARD STOP\nprev={prev}\nnow={split_rep}")
    else:
        C.dump_json(split_rep, split_path)
        print(f"[save-meta] split_report written: border1={border1} "
              f"mean={mean_:.6f} scale={scale_:.6f}")

    name = a.out_name or f"{a.ds}__h{a.pred_len}"
    npz_path = os.path.join(out_dir, f"{name}.npz")
    np.savez_compressed(
        npz_path,
        pred=preds.astype(np.float64), true=trues.astype(np.float64),
        row_pred_start=row_pred_start,
        date_pred_start=dates.astype("U"),
        ds=np.array(a.ds), pred_len=np.array(a.pred_len))
    done = {
        "ds": a.ds, "pred_len": a.pred_len, "windows": int(n),
        "batch": a.batch, "protocol_sha256": proto_sha,
        "protocol_token": a.protocol_token,
        "npz_sha256": C.sha256_file(npz_path),
        "official_printed": printed,
        "recompute_float32": {"mse": mse32, "mae": mae32},
        "metrics_float64": {"mse": mse64, "mae": mae64},
        "row_pred_start_range": [int(row_pred_start[0]),
                                 int(row_pred_start[-1])],
        "trues_scaler_bitwise_check": "PASS",
        "ok": True,
    }
    C.dump_json(done, os.path.join(out_dir, f"{name}.done.json"))
    print(f"[save-meta] {name}: n={n} batch={a.batch} official "
          f"mse={printed['mse']:.6f} mae={printed['mae']:.6f} "
          f"f64 mse={mse64:.6f} mae={mae64:.6f} "
          f"trues==scaler(OT) bitwise OK -> OK")


if __name__ == "__main__":
    main()
