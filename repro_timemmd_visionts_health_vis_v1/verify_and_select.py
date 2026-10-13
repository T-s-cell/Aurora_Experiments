#!/usr/bin/env python3
"""EXP-023 verify + select: read-only verification of the frozen EXP-021
Health artifacts, per-window metrics, and best/worst window selection.

Gates (any failure -> HARD STOP):
  G1  EXP-021 protocol.json sha256 == pin; its Health block == anchors
  G2  selected_p.json sha256 == pin; Health selected == 52
  G3  Health CSV sha256 == pin; 1389 rows; date/OT columns; dates
      non-decreasing 1997-09-29..2024-05-06 with exactly the 5 known
      duplicate week-boundary dates (1997/2003/2008/2014-12-29,
      2020-12-28 — frozen CSV property); split 972/140/277 with
      val_end 1112; OT finite
  G4  per H in {12,24,36,48}: npz sha256 == anchor pin == done.json
      npz_sha256; done.json protocol_token/sha, batch, periodicity,
      windows all match
  G5  per group identity: shapes (N,H,1); row_pred_start ==
      arange(1112, 1389-H+1); date_pred_start == CSV dates[s] window by
      window; scaled32 == npz true bitwise (StandardScaler fit on
      OT[:972], sklearn pinned 1.7.2 on dt); pred finite
  G6  per group recompute, official float32 replica == frozen metrics
      (atol 1e-6, expect bitwise); float64 appendix recorded
  G7  4H equal-weight mean of group recomputes == frozen 4H mean
      (atol 1e-5). Computed as mean over the four per-H values, NOT as
      a mean over the 992 per-window rows (H groups differ in size).

Selection (user fix #2): fixed H=12 only, k=5 lowest-MSE (best) and
k=5 highest-MSE (worst) of the 266 h12 test windows; ties broken by
window id h{H:02d}_s{row:04d} ascending. The other three horizons keep
their per-window metrics in the CSV (with group ranks) but are never
mixed into the selection.

Outputs (results/): gates.json, scaler.json, series_date_ot.csv (full
1389-row date+OT for offline plotting), health_window_metrics.csv (992
rows), selection.json (10 windows + pairwise overlap).
"""
import csv
import hashlib
import json
import os
import sys

import numpy as np

import anchors as A

RESULTS = A.RESULTS_DIR


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def die(msg):
    print(f"[verify] HARD STOP: {msg}")
    sys.exit(1)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def official_metric_replica(pred, true):
    """EXP-021 common.official_metric_replica: float32 arrays, np.mean
    with default dtype (float32 accumulation)."""
    pred = np.asarray(pred, dtype=np.float32)
    true = np.asarray(true, dtype=np.float32)
    mae = float(np.mean(np.abs(pred - true)))
    mse = float(np.mean((pred - true) ** 2))
    return {"mse": mse, "mae": mae}


def metrics_float64(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    return {"mse": float(np.mean((pred - true) ** 2)),
            "mae": float(np.mean(np.abs(pred - true)))}


def load_date_ot(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if "date" not in (r.fieldnames or []) or "OT" not in r.fieldnames:
            die(f"{path} lacks date/OT columns (header={r.fieldnames})")
        dates, ot = [], []
        for row in r:
            dates.append(row["date"])
            ot.append(float(row["OT"]))
    return dates, np.asarray(ot, dtype=np.float64)


def window_id(h, row):
    return f"h{h:02d}_s{row:04d}"


def main():
    os.makedirs(RESULTS, exist_ok=True)
    gates = []

    def gate(name, ok, detail):
        gates.append({"gate": name, "ok": bool(ok), "detail": detail})
        print(f"[verify] {'PASS' if ok else 'FAIL'} {name}: {detail}")
        if not ok:
            die(f"gate {name} failed")

    # ---- G1 protocol ----
    if not os.path.isfile(A.PROTO_PATH):
        die(f"EXP-021 protocol.json missing: {A.PROTO_PATH}")
    proto_sha = sha256_file(A.PROTO_PATH)
    gate("G1.protocol_sha256", proto_sha == A.BASE["protocol_sha256"],
         f"sha256={proto_sha}")
    proto = load_json(A.PROTO_PATH)
    hp = proto["datasets"]["Health"]
    want_block = {"seq_len": A.SEQ_LEN, "horizons": A.HORIZONS,
                  "windows": {str(k): v for k, v in A.WINDOWS.items()},
                  "split_rows": A.SPLIT_ROWS, "csv_sha256": A.CSV_SHA256,
                  "selected_periodicity": A.SELECTED_P,
                  "candidates": A.CANDIDATES, "dates": A.DATES}
    for k, v in want_block.items():
        gate(f"G1.protocol.Health.{k}", hp.get(k) == v,
             f"got {hp.get(k)!r} want {v!r}")

    # ---- G2 selected_p ----
    sel_sha = sha256_file(A.SEL_PATH)
    gate("G2.selected_p_sha256", sel_sha == A.BASE["selected_p_sha256"],
         f"sha256={sel_sha}")
    sel = load_json(A.SEL_PATH)
    gate("G2.Health_selected_P",
         sel["datasets"]["Health"]["selected"] == A.SELECTED_P,
         f"selected={sel['datasets']['Health']['selected']}")

    # ---- G3 CSV ----
    if not os.path.isfile(A.CSV_PATH):
        die(f"Health CSV missing: {A.CSV_PATH}")
    csv_sha = sha256_file(A.CSV_PATH)
    gate("G3.csv_sha256", csv_sha == A.CSV_SHA256, f"sha256={csv_sha}")
    dates, ot = load_date_ot(A.CSV_PATH)
    n = len(ot)
    gate("G3.n_rows", n == A.N_ROWS, f"rows={n}")
    gate("G3.dates_first_last",
         dates[0] == A.DATES["first"] and dates[-1] == A.DATES["last"],
         f"{dates[0]}..{dates[-1]}")
    dup = [(i, dates[i]) for i in range(1, n) if dates[i] == dates[i - 1]]
    dec = any(dates[i] < dates[i - 1] for i in range(1, n))
    known_dup = [(14, "1997-12-29"), (327, "2003-12-29"),
                 (588, "2008-12-29"), (901, "2014-12-29"),
                 (1214, "2020-12-28")]
    gate("G3.dates_monotonic",
         (not dec) and dup == known_dup,
         f"non-decreasing; exactly the 5 known duplicate week-boundary "
         f"dates {dup} (frozen CSV property, 2020-12-28 sits in the "
         f"test region)")
    n_train = int(n * 0.7)
    n_test = int(n * 0.2)
    val_end = n - n_test
    gate("G3.split", (n_train, val_end - n_train, n_test) ==
         tuple(A.SPLIT_ROWS) and val_end == A.VAL_END,
         f"{n_train}/{val_end - n_train}/{n_test} val_end={val_end}")
    gate("G3.ot_finite", bool(np.all(np.isfinite(ot))), "all finite")

    # ---- scaler (sklearn, pinned on dt) ----
    import sklearn
    from sklearn.preprocessing import StandardScaler
    sk_ver = getattr(sklearn, "__version__", "unknown")
    if not A.LOCAL_SMOKE and sk_ver != "1.7.2":
        die(f"sklearn {sk_ver} != pinned 1.7.2")
    if A.LOCAL_SMOKE:
        print("[verify] LOCAL_SMOKE: scaler bitwise gate relaxed to "
              f"allclose(atol={A.SCALER_SMOKE_ATOL}); strict bitwise gate "
              "runs on dt")
    scaler = StandardScaler()
    scaler.fit(ot[:A.N_TRAIN].reshape(-1, 1))
    mean_ot = float(scaler.mean_[0])
    scale_ot = float(scaler.scale_[0])
    scaled32 = scaler.transform(ot.reshape(-1, 1)).astype(np.float32).ravel()
    print(f"[verify] scaler mean_ot={mean_ot!r} scale_ot={scale_ot!r} "
          f"(fit rows [0,{A.N_TRAIN}), sklearn {sk_ver})")

    # ---- per-H npz gates + identity + recompute ----
    per_h = {}
    for h in A.HORIZONS:
        npz_p = A.npz_path(h)
        done_p = A.done_path(h)
        if not os.path.isfile(npz_p) or not os.path.isfile(done_p):
            die(f"missing {npz_p} or {done_p}")
        npz_sha = sha256_file(npz_p)
        gate(f"G4.h{h}.npz_sha256", npz_sha == A.NPZ_SHA256[h],
             f"sha256={npz_sha}")
        done = load_json(done_p)
        gate(f"G4.h{h}.done_npz_pin", done.get("npz_sha256") == npz_sha,
             "done.json npz_sha256 == file")
        gate(f"G4.h{h}.done_proto",
             done.get("protocol_token") == A.BASE["protocol_token"]
             and done.get("protocol_sha256") == A.BASE["protocol_sha256"],
             f"token={done.get('protocol_token')}")
        gate(f"G4.h{h}.done_run",
             done.get("batch") == A.BATCH
             and done.get("periodicity") == A.SELECTED_P
             and done.get("windows") == A.WINDOWS[h],
             f"batch={done.get('batch')} P={done.get('periodicity')} "
             f"windows={done.get('windows')}")
        gate(f"G4.h{h}.done_metrics_pin",
             done["metrics_float32"]["mse"] == A.FROZEN_METRICS_F32[h]["mse"]
             and done["metrics_float32"]["mae"] == A.FROZEN_METRICS_F32[h]["mae"],
             "done.json f32 == anchor pins (exact)")

        z = np.load(npz_p, allow_pickle=False)
        want_n = A.WINDOWS[h]
        gate(f"G5.h{h}.shapes",
             z["pred"].shape == (want_n, h, 1)
             and z["true"].shape == (want_n, h, 1)
             and z["pred"].dtype == np.float64
             and z["true"].dtype == np.float64
             and z["row_pred_start"].dtype == np.int64,
             f"pred/true {z['pred'].shape} f64")
        rows = z["row_pred_start"]
        want_rows = np.arange(A.VAL_END, A.N_ROWS - h + 1, dtype=np.int64)
        gate(f"G5.h{h}.row_pred_start", np.array_equal(rows, want_rows),
             f"rows {rows[0]}..{rows[-1]} (n={len(rows)})")
        want_dates = np.asarray([dates[s] for s in rows])
        gate(f"G5.h{h}.date_pred_start",
             np.array_equal(z["date_pred_start"].astype(str), want_dates),
             f"{want_dates[0]}..{want_dates[-1]}")
        true32 = z["true"].astype(np.float32)
        want_true = np.stack([scaled32[s:s + h] for s in rows])[..., None]
        if A.LOCAL_SMOKE:
            diff = float(np.max(np.abs(true32.astype(np.float64)
                                       - want_true.astype(np.float64))))
            gate(f"G5.h{h}.scaled_true", diff <= A.SCALER_SMOKE_ATOL,
                 f"LOCAL_SMOKE allclose max|diff|={diff:.3e}")
        else:
            gate(f"G5.h{h}.scaled_true_bitwise",
                 np.array_equal(true32, want_true),
                 "bitwise float32 == float32((OT-mean)/scale)")
        gate(f"G5.h{h}.pred_finite",
             bool(np.all(np.isfinite(z["pred"]))), "all finite")

        m32 = official_metric_replica(z["pred"], z["true"])
        m64 = metrics_float64(z["pred"], z["true"])
        dmse = abs(m32["mse"] - A.FROZEN_METRICS_F32[h]["mse"])
        dmae = abs(m32["mae"] - A.FROZEN_METRICS_F32[h]["mae"])
        gate(f"G6.h{h}.recompute_f32",
             dmse <= A.GROUP_ATOL and dmae <= A.GROUP_ATOL,
             f"f32 mse={m32['mse']!r} (d={dmse:.2e}) "
             f"mae={m32['mae']!r} (d={dmae:.2e})")
        per_h[h] = {"windows": want_n, "metrics_f32": m32,
                    "metrics_f64": m64,
                    "row_pred_start": rows.astype(int).tolist(),
                    "date_pred_start": want_dates.tolist()}

    mean_mse = sum(per_h[h]["metrics_f32"]["mse"] for h in A.HORIZONS) / 4.0
    mean_mae = sum(per_h[h]["metrics_f32"]["mae"] for h in A.HORIZONS) / 4.0
    gate("G7.mean_4h_equal_weight",
         abs(mean_mse - A.FROZEN_4H_MEAN_F32["mse"]) <= A.MEAN_ATOL
         and abs(mean_mae - A.FROZEN_4H_MEAN_F32["mae"]) <= A.MEAN_ATOL,
         f"mean(f32 per-H) mse={mean_mse!r} mae={mean_mae!r} "
         f"(per-H first, then 4H equal-weight average)")

    # ---- per-window metrics + selection (fixed H=12) ----
    csv_rows = []
    for h in A.HORIZONS:
        z = np.load(A.npz_path(h), allow_pickle=False)
        err = z["pred"] - z["true"]                      # (N,H,1) float64
        mse_w = (err ** 2).mean(axis=(1, 2))
        mae_w = np.abs(err).mean(axis=(1, 2))
        rows = per_h[h]["row_pred_start"]
        dstart = per_h[h]["date_pred_start"]
        items = []
        for i, s in enumerate(rows):
            items.append({
                "pred_len": h,
                "window_id": window_id(h, s),
                "row_pred_start": int(s),
                "date_pred_start": dstart[i],
                "date_pred_end": dates[s + h - 1],
                "history_start_row": int(s - A.SEQ_LEN),
                "history_start_date": dates[s - A.SEQ_LEN],
                "mse": float(mse_w[i]),
                "mae": float(mae_w[i]),
            })
        order = sorted(range(len(items)),
                       key=lambda i: (items[i]["mse"], items[i]["window_id"]))
        for rank, i in enumerate(order, 1):
            items[i]["group_rank"] = rank
        csv_rows.extend(items)

    h12 = [r for r in csv_rows if r["pred_len"] == A.SEL_H]
    if len(h12) != A.WINDOWS[A.SEL_H]:
        die(f"h12 rows {len(h12)} != {A.WINDOWS[A.SEL_H]}")
    best = sorted(h12, key=lambda r: (r["mse"], r["window_id"]))[:A.K]
    worst = sorted(h12,
                   key=lambda r: (-r["mse"], r["window_id"]))[:A.K]
    selected = {}
    for i, r in enumerate(best, 1):
        selected[r["window_id"]] = ("best", i)
    for i, r in enumerate(worst, 1):
        selected[r["window_id"]] = ("worst", i)
    for r in csv_rows:
        kind, rank = selected.get(r["window_id"], ("", ""))
        r["selected_kind"] = kind
        r["selected_rank"] = rank

    cols = ["pred_len", "window_id", "row_pred_start", "date_pred_start",
            "date_pred_end", "history_start_row", "history_start_date",
            "mse", "mae", "group_rank", "selected_kind", "selected_rank"]
    with open(os.path.join(RESULTS, "health_window_metrics.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in csv_rows:
            w.writerow({c: r[c] for c in cols})
    print(f"[verify] health_window_metrics.csv: {len(csv_rows)} rows "
          f"(all 4 horizons; selection flags on h{A.SEL_H} only)")

    # ---- pairwise overlap among the 10 selected windows ----
    ten = best + worst
    L = A.SEQ_LEN
    pairs = []
    for i in range(len(ten)):
        for j in range(i + 1, len(ten)):
            a, b = ten[i], ten[j]
            pa = (a["row_pred_start"], a["row_pred_start"] + a["pred_len"] - 1)
            pb = (b["row_pred_start"], b["row_pred_start"] + b["pred_len"] - 1)
            ov_p = max(0, min(pa[1], pb[1]) - max(pa[0], pb[0]) + 1)
            fa = (a["history_start_row"], pa[1])
            fb = (b["history_start_row"], pb[1])
            ov_f = max(0, min(fa[1], fb[1]) - max(fa[0], fb[0]) + 1)
            pairs.append({"a": a["window_id"], "b": b["window_id"],
                          "pred_overlap_rows": ov_p,
                          "full_span_overlap_rows": ov_f})
    ov_summary = {
        "n_pairs": len(pairs),
        "pairs_with_pred_overlap": sum(1 for p in pairs
                                       if p["pred_overlap_rows"] > 0),
        "max_pred_overlap_rows": max(p["pred_overlap_rows"] for p in pairs),
        "min_full_span_overlap_rows": min(p["full_span_overlap_rows"]
                                          for p in pairs),
        "note": "1 row = 1 week (Health is weekly); same-H windows start "
                "stride 1, so heavy overlap is expected",
    }

    selection = {
        "exp_id": A.EXP_ID,
        "base_experiment": {"exp_id": A.BASE["exp_id"],
                            "protocol_token": A.BASE["protocol_token"],
                            "protocol_sha256": A.BASE["protocol_sha256"]},
        "scale": "scaled space (EXP-021 evaluation scale); metrics from "
                 "frozen npz pred/true",
        "sel_h": A.SEL_H,
        "k": A.K,
        "pool": f"h{A.SEL_H} test windows only ({A.WINDOWS[A.SEL_H]} "
                f"windows); horizons never mixed (user fix #2)",
        "rule": "best = k lowest per-window MSE; worst = k highest; ties "
                "broken by window id h{H:02d}_s{row:04d} ascending",
        "windows": [{
            "kind": r["selected_kind"], "rank": r["selected_rank"],
            "window_id": r["window_id"], "pred_len": r["pred_len"],
            "row_pred_start": r["row_pred_start"],
            "date_pred_start": r["date_pred_start"],
            "date_pred_end": r["date_pred_end"],
            "history": [r["history_start_row"], r["row_pred_start"] - 1],
            "history_dates": [r["history_start_date"],
                              dates[r["row_pred_start"] - 1]],
            "mse": r["mse"], "mae": r["mae"],
            "group_rank": r["group_rank"],
        } for r in ten],
        "overlap": {"pairs": pairs, "summary": ov_summary},
    }
    with open(os.path.join(RESULTS, "selection.json"), "w",
              encoding="utf-8") as f:
        json.dump(selection, f, indent=1, ensure_ascii=False)

    with open(os.path.join(RESULTS, "series_date_ot.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "OT"])
        for d, v in zip(dates, ot.tolist()):
            w.writerow([d, repr(v)])

    gates_report = {
        "exp_id": A.EXP_ID,
        "local_smoke": A.LOCAL_SMOKE,
        "sklearn_version": sk_ver,
        "numpy_version": np.__version__,
        "scaler": {"mean_ot": mean_ot, "scale_ot": scale_ot,
                   "fit_rows": [0, A.N_TRAIN]},
        "per_h": {str(h): {"windows": per_h[h]["windows"],
                           "metrics_f32": per_h[h]["metrics_f32"],
                           "metrics_f64": per_h[h]["metrics_f64"]}
                  for h in A.HORIZONS},
        "mean_4h_f32": {"mse": mean_mse, "mae": mean_mae},
        "gates": gates,
    }
    with open(os.path.join(RESULTS, "gates.json"), "w",
              encoding="utf-8") as f:
        json.dump(gates_report, f, indent=1, ensure_ascii=False)
    with open(os.path.join(RESULTS, "scaler.json"), "w",
              encoding="utf-8") as f:
        json.dump({"ds": A.DS, "csv_sha256": A.CSV_SHA256,
                   "ot_column": "OT", "n_rows": A.N_ROWS,
                   "fit_rows": [0, A.N_TRAIN], "mean_ot": mean_ot,
                   "scale_ot": scale_ot, "sklearn_version": sk_ver,
                   "numpy_version": np.__version__,
                   "note": "StandardScaler fit on OT[0:972); "
                           "scaled32 = float32((OT-mean)/scale); inverse: "
                           "orig = scaled*scale + mean"},
                  f, indent=1, ensure_ascii=False)

    print(f"[verify] best h{A.SEL_H}: "
          + ", ".join(f"{r['window_id']}({r['mse']:.4f})" for r in best))
    print(f"[verify] worst h{A.SEL_H}: "
          + ", ".join(f"{r['window_id']}({r['mse']:.4f})" for r in worst))
    print("[verify] overlap:", ov_summary)
    print("VERIFY_SELECT OK")


if __name__ == "__main__":
    main()
