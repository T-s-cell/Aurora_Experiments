#!/usr/bin/env python3
"""Score the 4 chronological-order groups and compare against the
original-row-order baseline (EXP-019) and the paper.

Per group : PRIMARY metric = official float32 replica (identical
            computation to the official entry's printed metrics; must
            equal them within 1e-12). float64 recomputation is kept as
            an appendix column, reported, no gate.
Baseline  : EXP-019 frozen predictions/{Economy__h*}.npz are READ-ONLY
            inputs; baseline metrics are recomputed at full precision
            from the saved arrays and cross-checked against the
            baseline done.json (report only, no gate).
Per domain: equal-weight average of the 4 horizons in float32 (the
            official metric space); paper values are reference columns
            ONLY (the data protocol changed: sorted rows, test period
            2017-04..2024-03 vs baseline 1989-01..1995-12).
Outputs   : results/Economy_metrics.json,
            results/comparison_economy_chrono.csv,
            results/env_report.json, results/summary.json.
"""
import csv
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C
import protocol as P

BASE_PRED = os.path.join(P.BASE_DIR, "predictions")


def group_metrics(npz_path, done_path):
    done = C.load_json(done_path)
    z = np.load(npz_path, allow_pickle=False)
    pred64, true64 = z["pred"], z["true"]
    mae32, mse32 = C.official_metric_replica(
        pred64.astype(np.float32), true64.astype(np.float32))
    mae64, mse64 = C.metrics_float64(pred64, true64)
    if not (C.close(mse32, done["official_printed"]["mse"])
            and C.close(mae32, done["official_printed"]["mae"])):
        sys.exit(f"[evaluate] recompute gate FAIL for {os.path.basename(npz_path)}: "
                 f"replica mse={mse32!r} vs printed "
                 f"{done['official_printed']['mse']!r}")
    return done, z, mae32, mse32, mae64, mse64


def main():
    paper = C.load_json(os.path.join(HERE, "paper_ref.json"))
    base_proto = C.load_json(P.base_protocol_path())
    groups, rows = {}, []
    vals32 = {"mse": [], "mae": []}
    vals64 = {"mse": [], "mae": []}
    bvals32 = {"mse": [], "mae": []}
    missing = []

    for h in (6, 8, 10, 12):  # presentation order; run order was 12 first
        name = f"Economy__h{h}"
        npz_p = os.path.join(C.PRED_DIR, f"{name}.npz")
        done_p = os.path.join(C.PRED_DIR, f"{name}.done.json")
        if not (os.path.isfile(done_p) and os.path.isfile(npz_p)):
            missing.append(name)
            continue
        done, z, mae32, mse32, mae64, mse64 = group_metrics(npz_p, done_p)
        g = {"pred_len": h, "windows": int(z["pred"].shape[0]),
             "batch": done["batch"],
             "first_pred_date": str(z["date_pred_start"][0]),
             "last_pred_date": str(z["date_pred_start"][-1]),
             "row_pred_start_range": [int(z["row_pred_start"][0]),
                                      int(z["row_pred_start"][-1])],
             "official_printed": done["official_printed"],
             "mse_mae_PRIMARY_float32": {"mse": mse32, "mae": mae32},
             "float64_appendix": {"mse": mse64, "mae": mae64}}
        groups[str(h)] = g
        vals32["mse"].append(mse32)
        vals32["mae"].append(mae32)
        vals64["mse"].append(mse64)
        vals64["mae"].append(mae64)

        # baseline (EXP-019, original row order) — read-only recompute
        b_npz = os.path.join(BASE_PRED, f"{name}.npz")
        b_done_p = os.path.join(BASE_PRED, f"{name}.done.json")
        if not (os.path.isfile(b_done_p) and os.path.isfile(b_npz)):
            sys.exit(f"[evaluate] baseline evidence missing: {b_npz}")
        b_done = C.load_json(b_done_p)
        bz = np.load(b_npz, allow_pickle=False)
        bpred, btrue = bz["pred"], bz["true"]
        bmae32, bmse32 = C.official_metric_replica(
            bpred.astype(np.float32), btrue.astype(np.float32))
        b_xc = {"done_mse": b_done["recompute_float32"]["mse"],
                "done_mae": b_done["recompute_float32"]["mae"],
                "recomputed_mse": bmse32, "recomputed_mae": bmae32,
                "abs_drift": {"mse": abs(bmse32 - b_done["recompute_float32"]["mse"]),
                              "mae": abs(bmae32 - b_done["recompute_float32"]["mae"])}}
        g["baseline_original_order"] = {
            "mse_mAE_float32": {"mse": bmse32, "mae": bmae32},
            "windows": int(bpred.shape[0]),
            "first_pred_date": str(bz["date_pred_start"][0]),
            "last_pred_date": str(bz["date_pred_start"][-1]),
            "done_json_cross_check": b_xc}
        bvals32["mse"].append(bmse32)
        bvals32["mae"].append(bmae32)

        pm = paper["table12"]["values"]["Economy"][str(h)]
        rows.append({
            "pred_len": h,
            "windows_chrono": g["windows"], "windows_orig": int(bpred.shape[0]),
            "mse_orig_f32": round(bmse32, 6), "mae_orig_f32": round(bmae32, 6),
            "mse_chrono_f32": round(mse32, 6), "mae_chrono_f32": round(mae32, 6),
            "mse_diff_chrono_minus_orig": round(mse32 - bmse32, 6),
            "mae_diff_chrono_minus_orig": round(mae32 - bmae32, 6),
            "mse_rel_pct": round((mse32 - bmse32) / bmse32 * 100, 3),
            "mae_rel_pct": round((mae32 - bmae32) / bmae32 * 100, 3),
            "mse_paper_reference": pm[0], "mae_paper_reference": pm[1],
            "chrono_vs_paper_mse_pct": round((mse32 - pm[0]) / pm[0] * 100, 3),
        })
        C.dump_json(groups, os.path.join(C.RESULTS_DIR,
                                         "Economy_metrics.json"))

    if missing:
        sys.exit(f"[evaluate] groups missing: {missing} — run_all not "
                 f"complete, refusing to write comparisons")

    avg32 = {"mse": float(np.mean(vals32["mse"])),
             "mae": float(np.mean(vals32["mae"]))}
    avg64 = {"mse": float(np.mean(vals64["mse"])),
             "mae": float(np.mean(vals64["mae"]))}
    bavg32 = {"mse": float(np.mean(bvals32["mse"])),
              "mae": float(np.mean(bvals32["mae"]))}
    pa = paper["table12"]["values"]["Economy"]["avg"]
    rows.append({
        "pred_len": "avg4H",
        "windows_chrono": P.TOTAL_WINDOWS,
        "windows_orig": sum(r["windows_orig"] for r in rows),
        "mse_orig_f32": round(bavg32["mse"], 6),
        "mae_orig_f32": round(bavg32["mae"], 6),
        "mse_chrono_f32": round(avg32["mse"], 6),
        "mae_chrono_f32": round(avg32["mae"], 6),
        "mse_diff_chrono_minus_orig": round(avg32["mse"] - bavg32["mse"], 6),
        "mae_diff_chrono_minus_orig": round(avg32["mae"] - bavg32["mae"], 6),
        "mse_rel_pct": round((avg32["mse"] - bavg32["mse"]) / bavg32["mse"] * 100, 3),
        "mae_rel_pct": round((avg32["mae"] - bavg32["mae"]) / bavg32["mae"] * 100, 3),
        "mse_paper_reference": pa[0], "mae_paper_reference": pa[1],
        "chrono_vs_paper_mse_pct": round((avg32["mse"] - pa[0]) / pa[0] * 100, 3),
    })

    def write_csv(path, rws, cols):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rws)

    write_csv(os.path.join(C.RESULTS_DIR, "comparison_economy_chrono.csv"),
              rows, list(rows[0].keys()))
    for r in rows:
        print(f"[evaluate] H={r['pred_len']}: MSE orig "
              f"{r['mse_orig_f32']:.6f} -> chrono {r['mse_chrono_f32']:.6f} "
              f"({r['mse_rel_pct']:+.1f}%)  MAE orig "
              f"{r['mae_orig_f32']:.6f} -> chrono {r['mae_chrono_f32']:.6f} "
              f"({r['mae_rel_pct']:+.1f}%)")

    # env report (pip freezes are captured by the driver into logs/)
    env = {"runtime": C.env_info()}
    for tag in ("aurora_tmmd_v1", "torch_50_series"):
        p = os.path.join(C.LOGS_DIR, f"pip_freeze_{tag}.txt")
        if os.path.isfile(p):
            env[f"pip_freeze_{tag}"] = open(p).read().splitlines()
    C.dump_json(env, os.path.join(C.RESULTS_DIR, "env_report.json"))

    token = hashlib.sha256(
        open(C.PROTO_PATH, "rb").read()).hexdigest()[:12]
    summary = {
        "label": P.LABEL,
        "designation": "Aurora 官方脚本、TimeMMD Economy 时间升序对照、zero-shot v1",
        "exp_id": P.EXP_ID,
        "protocol_token": token,
        "protocol_sha256": C.sha256_file(C.PROTO_PATH),
        "groups_done": len(groups), "groups_expected": 4,
        "groups_missing": missing,
        "metric_primary": ("official float32 replica (identical computation "
                           "to the official entry's printed metrics); "
                           "float64 kept as appendix column, no gate"),
        "baseline": {
            "exp_id": P.BASE_EXP_ID, "dir": P.BASE_DIR_NAME,
            "protocol_token": P.BASE_PROTOCOL_TOKEN,
            "add_commit": P.BASE_ADD_COMMIT,
            "freeze_commit": P.BASE_FREEZE_COMMIT,
            "metrics": "recomputed read-only from EXP-019 frozen npz at "
                       "full precision; cross-checked vs its done.json",
        },
        "chrono_test_period": "2017-04 .. 2024-03",
        "orig_test_period": "1989-01 .. 1995-12",
        "paper_deviations": P.PAPER_DEVIATION_NOTES,
        "scope": ("Aurora official public implementation + official "
                  "benchmark pack, Economy rows sorted chronologically, "
                  "this environment only; the sort simultaneously changes "
                  "history-window composition, text selection, train/test "
                  "periods and scaler statistics, so chrono-vs-orig diffs "
                  "describe the two data protocols and do not by themselves "
                  "prove any paper error"),
    }
    C.dump_json(summary, os.path.join(C.RESULTS_DIR, "summary.json"))
    print(f"[evaluate] groups done: {len(groups)}/4")
    print(f"[evaluate] 4H avg (float32): chrono MSE {avg32['mse']:.6f} / "
          f"MAE {avg32['mae']:.6f} vs orig MSE {bavg32['mse']:.6f} / "
          f"MAE {bavg32['mae']:.6f}")


if __name__ == "__main__":
    main()
