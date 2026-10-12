#!/usr/bin/env python3
"""Score all finished groups and compare against the paper.

Per group : PRIMARY metric = official float32 replica (identical
            computation to the official entry's printed metrics; must
            equal them within 1e-12). float64 recomputation is kept as
            an appendix column, reported, no gate.
Per domain: equal-weight average of the 4 horizons in float32 (the
            official metric space), aligning paper Table 1; per-horizon
            rows align Table 12.
Outputs   : results/{ds}_metrics.json, comparison_table_12.csv,
            comparison_table_1.csv, env_report.json, summary.json.
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


def load_paper():
    return C.load_json(os.path.join(HERE, "paper_ref.json"))


def main():
    paper = load_paper()
    rows12, rows1 = [], []
    per_domain = {}
    done_groups, missing = 0, []
    for ds in P.DOMAIN_ORDER:
        dm = {"groups": {}, "avg_float32": None, "avg_float64": None}
        vals32 = {"mse": [], "mae": []}
        vals64 = {"mse": [], "mae": []}
        for h in P.DATASETS[ds]["horizons"]:
            name = f"{ds}__h{h}"
            done_p = os.path.join(C.PRED_DIR, f"{name}.done.json")
            npz_p = os.path.join(C.PRED_DIR, f"{name}.npz")
            if not (os.path.isfile(done_p) and os.path.isfile(npz_p)):
                missing.append(name)
                continue
            done = C.load_json(done_p)
            z = np.load(npz_p, allow_pickle=False)
            pred64, true64 = z["pred"], z["true"]
            mae32, mse32 = C.official_metric_replica(
                pred64.astype(np.float32), true64.astype(np.float32))
            mae64, mse64 = C.metrics_float64(pred64, true64)
            g = {"pred_len": h, "windows": int(pred64.shape[0]),
                 "batch": done["batch"],
                 "official_printed": done["official_printed"],
                 "mse_mae_PRIMARY_float32": {"mse": mse32, "mae": mae32},
                 "float64_appendix": {"mse": mse64, "mae": mae64}}
            dm["groups"][str(h)] = g
            vals32["mse"].append(mse32)
            vals32["mae"].append(mae32)
            vals64["mse"].append(mse64)
            vals64["mae"].append(mae64)
            pm = paper["table12"]["values"][ds][str(h)]
            rows12.append({
                "domain": ds, "pred_len": h, "windows": g["windows"],
                "batch": done["batch"],
                "mse_ours": round(mse32, 6), "mae_ours": round(mae32, 6),
                "mse_ours_f64_appendix": round(mse64, 6),
                "mae_ours_f64_appendix": round(mae64, 6),
                "mse_paper": pm[0], "mae_paper": pm[1],
                "mse_abs_diff": round(mse32 - pm[0], 6),
                "mae_abs_diff": round(mae32 - pm[1], 6),
                "mse_rel_diff_pct": round((mse32 - pm[0]) / pm[0] * 100, 3),
                "mae_rel_diff_pct": round((mae32 - pm[1]) / pm[1] * 100, 3),
            })
            done_groups += 1
        if vals32["mse"]:
            dm["avg_float32"] = {"mse": float(np.mean(vals32["mse"])),
                                 "mae": float(np.mean(vals32["mae"]))}
            dm["avg_float64"] = {"mse": float(np.mean(vals64["mse"])),
                                 "mae": float(np.mean(vals64["mae"]))}
            pa = paper["table12"]["values"][ds]["avg"]
            rows1.append({
                "domain": ds,
                "mse_ours": round(dm["avg_float32"]["mse"], 6),
                "mae_ours": round(dm["avg_float32"]["mae"], 6),
                "mse_ours_f64_appendix": round(dm["avg_float64"]["mse"], 6),
                "mae_ours_f64_appendix": round(dm["avg_float64"]["mae"], 6),
                "mse_paper": pa[0], "mae_paper": pa[1],
                "mse_abs_diff": round(dm["avg_float32"]["mse"] - pa[0], 6),
                "mae_abs_diff": round(dm["avg_float32"]["mae"] - pa[1], 6),
                "mse_rel_diff_pct": round(
                    (dm["avg_float32"]["mse"] - pa[0]) / pa[0] * 100, 3),
                "mae_rel_diff_pct": round(
                    (dm["avg_float32"]["mae"] - pa[1]) / pa[1] * 100, 3),
            })
        per_domain[ds] = dm
        C.dump_json(dm, os.path.join(C.RESULTS_DIR, f"{ds}_metrics.json"))

    def write_csv(path, rows, cols):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    write_csv(os.path.join(C.RESULTS_DIR, "comparison_table_12.csv"),
              rows12, list(rows12[0].keys()) if rows12 else
              ["domain", "pred_len"])
    write_csv(os.path.join(C.RESULTS_DIR, "comparison_table_1.csv"),
              rows1, list(rows1[0].keys()) if rows1 else ["domain"])

    # env report (pip freezes are captured by the watcher into logs/)
    env = {"runtime": C.env_info()}
    for tag in ("aurora_tmmd_v1", "torch_50_series"):
        p = os.path.join(C.LOGS_DIR, f"pip_freeze_{tag}.txt")
        if os.path.isfile(p):
            env[f"pip_freeze_{tag}"] = open(p).read().splitlines()
    C.dump_json(env, os.path.join(C.RESULTS_DIR, "env_report.json"))

    summary = {
        "label": P.LABEL,
        "designation": "Aurora 官方脚本、TimeMMD 原始行序、zero-shot v1",
        "protocol_token": hashlib.sha256(
            open(C.PROTO_PATH, "rb").read()).hexdigest()[:12],
        "protocol_sha256": C.sha256_file(C.PROTO_PATH),
        "groups_done": done_groups, "groups_expected": 36,
        "groups_missing": missing,
        "metric_primary": ("official float32 replica (identical computation "
                           "to the official entry's printed metrics); "
                           "float64 kept as appendix column, no gate"),
        "paper_deviations": P.PAPER_DEVIATION_NOTES,
        "scope": ("Aurora official public implementation + official "
                  "benchmark pack, original row order, this environment "
                  "only; Economy numbers valid only as evidence about the "
                  "original-row-order protocol, not about the authors' "
                  "actual pipeline"),
    }
    C.dump_json(summary, os.path.join(C.RESULTS_DIR, "summary.json"))
    print(f"[evaluate] groups done: {done_groups}/36; missing={missing}")
    if rows1:
        print("[evaluate] domain averages (ours float32 primary vs paper):")
        for r in rows1:
            print(f"  {r['domain']:12s} MSE {r['mse_ours']:.3f} vs "
                  f"{r['mse_paper']:.3f} ({r['mse_rel_diff_pct']:+.1f}%)  "
                  f"MAE {r['mae_ours']:.3f} vs {r['mae_paper']:.3f} "
                  f"({r['mae_rel_diff_pct']:+.1f}%)")


if __name__ == "__main__":
    main()
