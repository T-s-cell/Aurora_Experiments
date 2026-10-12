#!/usr/bin/env python3
"""Stage B evaluation (EXP-021): gates every group, then builds the
paper comparison.

- recompute gate: official float32 replica recomputed from each npz must
  equal the done.json record at 1e-12 (float64 appendix, no gate)
- paper reference: paper_ref_visionts.json (Table 12 per-H + Table 1
  domain means, Security OCR comma conversion on record)
- EXP-019 Aurora measured column: recomputed read-only from the frozen
  base npz (same data protocol reference point)
- outputs: results/comparison_per_h.csv (36 rows), results/
  comparison_per_domain.csv (9 rows, domain mean = equal-weight 4H mean
  for OUR numbers; the paper column uses Table 1's printed values), and
  results/summary.json
"""
import csv
import hashlib
import os
import sys

import numpy as np

import common as C
import protocol as P
from run_all import done_is_valid

PAPER_REF = os.path.join(C.HERE, "paper_ref_visionts.json")
BASE_PRED_DIR = os.path.join("/home/wlt/MMTS/Aurora_Experiments",
                             P.BASE["dir"], "predictions")


def group_metrics(ds, h, proto, proto_sha, batch, p_ds):
    name = f"{ds}__h{h}"
    ok, why = done_is_valid(name, ds, h, proto, proto_sha, batch, p_ds)
    if not ok:
        sys.exit(f"[evaluate] HARD STOP: {name} not complete: {why}")
    done = C.load_json(os.path.join(C.PRED_DIR, f"{name}.done.json"))
    z = np.load(os.path.join(C.PRED_DIR, f"{name}.npz"), allow_pickle=False)
    mae32, mse32 = C.official_metric_replica(z["pred"].astype(np.float32),
                                             z["true"].astype(np.float32))
    mae64, mse64 = C.metrics_float64(z["pred"], z["true"])
    if not (C.close(mse32, done["metrics_float32"]["mse"])
            and C.close(mae32, done["metrics_float32"]["mae"])):
        sys.exit(f"[evaluate] HARD STOP: {name} recompute vs done record "
                 f"mismatch")
    return {"mse32": mse32, "mae32": mae32, "mse64": mse64, "mae64": mae64,
            "windows": int(len(z["pred"]))}


def base_metrics(ds, h):
    z = np.load(os.path.join(BASE_PRED_DIR, f"{ds}__h{h}.npz"),
                allow_pickle=False)
    mae32, mse32 = C.official_metric_replica(z["pred"].astype(np.float32),
                                             z["true"].astype(np.float32))
    return {"mse32": mse32, "mae32": mae32}


def rel(ours, ref):
    return (ours - ref) / ref * 100.0


def main():
    import hashlib
    if not os.path.isfile(C.PROTO_PATH):
        sys.exit("[evaluate] HARD STOP: protocol.json missing")
    token = hashlib.sha256(open(C.PROTO_PATH, "rb").read()).hexdigest()[:12]
    proto_sha = C.sha256_file(C.PROTO_PATH)
    proto = C.load_json(C.PROTO_PATH)
    sp_path = P.selected_p_path()
    if C.sha256_file(sp_path) != proto["selected_p_sha256"]:
        sys.exit("[evaluate] HARD STOP: selected_p.json sha256 != pin")
    sel = C.load_json(sp_path)
    paper = C.load_json(PAPER_REF)

    bd = C.load_json(os.path.join(C.RESULTS_DIR, "batch_decision.json"))
    batch = int(bd["chosen"]["min_over_domains"])

    rows_h, rows_d = [], []
    for ds in P.DOMAIN_ORDER:
        spec = P.DATASETS[ds]
        p_ds = int(sel["datasets"][ds]["selected"])
        if p_ds != int(proto["datasets"][ds]["selected_periodicity"]):
            sys.exit(f"[evaluate] HARD STOP: {ds} P mismatch vs protocol")
        means_ours, means_paper, means_a19 = [], [], []
        for h in spec["horizons"]:
            m = group_metrics(ds, h, proto, proto_sha, batch, p_ds)
            pr = paper["domains"][ds]["per_H"][str(h)]
            b19 = base_metrics(ds, h)
            means_ours.append((m["mse32"], m["mae32"]))
            means_paper.append((pr["mse"], pr["mae"]))
            means_a19.append((b19["mse32"], b19["mae32"]))
            rows_h.append({
                "dataset": ds, "horizon": h, "selected_P": p_ds,
                "windows": m["windows"],
                "ours_mse_f32": m["mse32"], "ours_mae_f32": m["mae32"],
                "ours_mse_f64": m["mse64"], "ours_mae_f64": m["mae64"],
                "paper_visionts_mse": pr["mse"],
                "paper_visionts_mae": pr["mae"],
                "diff_mse": m["mse32"] - pr["mse"],
                "diff_mae": m["mae32"] - pr["mae"],
                "rel_pct_mse": rel(m["mse32"], pr["mse"]),
                "rel_pct_mae": rel(m["mae32"], pr["mae"]),
                "exp019_aurora_mse": b19["mse32"],
                "exp019_aurora_mae": b19["mae32"],
            })
        mo_mse = float(np.mean([x[0] for x in means_ours]))
        mo_mae = float(np.mean([x[1] for x in means_ours]))
        mp = paper["domains"][ds]["table1_mean"]
        a19_mse = float(np.mean([x[0] for x in means_a19]))
        a19_mae = float(np.mean([x[1] for x in means_a19]))
        rows_d.append({
            "dataset": ds, "selected_P": p_ds,
            "ours_4h_mean_mse_f32": mo_mse, "ours_4h_mean_mae_f32": mo_mae,
            "paper_table1_mse": mp["mse"], "paper_table1_mae": mp["mae"],
            "diff_mse": mo_mse - mp["mse"], "diff_mae": mo_mae - mp["mae"],
            "rel_pct_mse": rel(mo_mse, mp["mse"]),
            "rel_pct_mae": rel(mo_mae, mp["mae"]),
            "exp019_aurora_measured_4h_mean_mse": a19_mse,
            "exp019_aurora_measured_4h_mean_mae": a19_mae,
        })
        print(f"[evaluate] {ds}: P={p_ds} ours {mo_mse:.4f}/{mo_mae:.4f} "
              f"vs paper {mp['mse']}/{mp['mae']} "
              f"({rel(mo_mse, mp['mse']):+.1f}%/{rel(mo_mae, mp['mae']):+.1f}%)")

    def write_csv(path, rows):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv(os.path.join(C.RESULTS_DIR, "comparison_per_h.csv"), rows_h)
    write_csv(os.path.join(C.RESULTS_DIR, "comparison_per_domain.csv"),
              rows_d)

    n_groups = len(rows_h)
    if n_groups != 36:
        sys.exit(f"[evaluate] HARD STOP: {n_groups} groups != 36")
    better_mse = sum(1 for r in rows_d if r["diff_mse"] < 0)
    summary = {
        "exp_id": P.EXP_ID,
        "label": P.LABEL,
        "protocol_token": token,
        "protocol_sha256": proto_sha,
        "groups_done": n_groups,
        "groups_expected": 36,
        "batch": batch,
        "selected_periodicities": {ds: int(sel["datasets"][ds]["selected"])
                                   for ds in P.DOMAIN_ORDER},
        "selection_scores": {ds: {cs: c["per_H_mean_mse_float32"]
                                  for cs, c in
                                  sel["datasets"][ds]["candidates"].items()}
                             for ds in P.DOMAIN_ORDER},
        "domain_comparison": rows_d,
        "our_mse_better_than_paper_in":
            f"{better_mse}/9 domains (4H equal-weight mean vs Table 1)",
        "metric_primary": "official float32 replica; float64 appendix, "
                          "no gate",
        "interpretation_boundary": (
            "P here is selected by the VisionTS paper's own validation-set "
            "method; the Aurora authors' VisionTS periodicity/history "
            "settings are not public. This is a same-data-protocol "
            "re-scoring of VisionTS on TimeMMD raw row order, NOT a "
            "bit-identical replication of the paper's VisionTS numbers — "
            "differences do not by themselves prove any paper error."),
        "paper_deviations": proto["paper_deviation_notes"],
        "base_reference": {"exp_id": "EXP-019",
                           "protocol_token": P.BASE["protocol_token"],
                           "freeze_commit": P.BASE["freeze_commit"],
                           "aurora_measured": "recomputed read-only from "
                                              "frozen EXP-019 npz"},
    }
    C.dump_json(summary, os.path.join(C.RESULTS_DIR, "summary.json"))
    print(f"EVALUATE OK: {n_groups}/36 groups; better-MSE domains "
          f"{better_mse}/9")


if __name__ == "__main__":
    main()
