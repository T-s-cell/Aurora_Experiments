#!/usr/bin/env python3
"""Evaluation (EXP-022): gates every group, then builds the three-way
comparison.

- recompute gate: official float32 replica recomputed from each own npz
  must equal the done.json record at 1e-12 (float64 appendix, no gate)
- EXP-020 Aurora ascending column: recomputed read-only from the frozen
  EXP-020 npz; 4H mean gated vs paper_ref_exp020_exp021.json
  (MSE 0.276172 / MAE 0.413902, atol 1e-5)
- EXP-021 VisionTS raw-order column: recomputed read-only from the
  frozen EXP-021 Economy npz; 4H mean gated vs
  (MSE 0.017808 / MAE 0.112694, atol 1e-5)
- outputs: results/comparison_per_h.csv (4 rows), results/
  comparison_per_mean.csv (1 row, 4H equal-weight mean), and
  results/summary.json
"""
import csv
import hashlib
import json
import os
import sys

import numpy as np

import common as C
import protocol as P
from run_all import done_is_valid

PAPER_REF = os.path.join(C.HERE, "paper_ref_exp020_exp021.json")
AURORA_ROOT = "/home/wlt/MMTS/Aurora_Experiments"
DS = "Economy"


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


def anchor_metrics(base_dir, h):
    z = np.load(os.path.join(base_dir, "predictions",
                             f"{DS}__h{h}.npz"), allow_pickle=False)
    mae32, mse32 = C.official_metric_replica(z["pred"].astype(np.float32),
                                             z["true"].astype(np.float32))
    return {"mse32": mse32, "mae32": mae32}


def rel(ours, ref):
    return (ours - ref) / ref * 100.0


def gate_mean(tag, got_mse, got_mae, ref_mse, ref_mae, atol=1e-5):
    if abs(got_mse - ref_mse) > atol or abs(got_mae - ref_mae) > atol:
        sys.exit(f"[evaluate] HARD STOP: {tag} 4H mean gate failed: got "
                 f"{got_mse:.6f}/{got_mae:.6f} vs ref "
                 f"{ref_mse:.6f}/{ref_mae:.6f} (atol {atol})")


def main():
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

    spec = P.DATASETS[DS]
    p_ds = int(sel["datasets"][DS]["selected"])
    if p_ds != int(proto["datasets"][DS]["selected_periodicity"]):
        sys.exit(f"[evaluate] HARD STOP: {DS} P mismatch vs protocol")

    exp020_dir = os.path.join(AURORA_ROOT, P.BASE_DATA["dir"])
    exp021_dir = os.path.join(AURORA_ROOT, P.BASE_METHOD["dir"])

    rows_h = []
    ours, a20, a21 = [], [], []
    for h in spec["horizons"]:
        m = group_metrics(DS, h, proto, proto_sha, batch, p_ds)
        m20 = anchor_metrics(exp020_dir, h)
        m21 = anchor_metrics(exp021_dir, h)
        ours.append((m["mse32"], m["mae32"]))
        a20.append((m20["mse32"], m20["mae32"]))
        a21.append((m21["mse32"], m21["mae32"]))
        rows_h.append({
            "dataset": DS, "horizon": h, "selected_P": p_ds,
            "windows": m["windows"],
            "ours_mse_f32": m["mse32"], "ours_mae_f32": m["mae32"],
            "ours_mse_f64": m["mse64"], "ours_mae_f64": m["mae64"],
            "exp020_aurora_chrono_mse": m20["mse32"],
            "exp020_aurora_chrono_mae": m20["mae32"],
            "diff_vs_exp020_mse": m["mse32"] - m20["mse32"],
            "diff_vs_exp020_mae": m["mae32"] - m20["mae32"],
            "rel_pct_vs_exp020_mse": rel(m["mse32"], m20["mse32"]),
            "rel_pct_vs_exp020_mae": rel(m["mae32"], m20["mae32"]),
            "exp021_visionts_raw_mse": m21["mse32"],
            "exp021_visionts_raw_mae": m21["mae32"],
            "diff_vs_exp021_mse": m["mse32"] - m21["mse32"],
            "diff_vs_exp021_mae": m["mae32"] - m21["mae32"],
            "rel_pct_vs_exp021_mse": rel(m["mse32"], m21["mse32"]),
            "rel_pct_vs_exp021_mae": rel(m["mae32"], m21["mae32"]),
        })
        print(f"[evaluate] h{h}: ours {m['mse32']:.6f}/{m['mae32']:.6f} | "
              f"EXP-020 Aurora {m20['mse32']:.6f}/{m20['mae32']:.6f} | "
              f"EXP-021 VisionTS {m21['mse32']:.6f}/{m21['mae32']:.6f}")

    if len(rows_h) != 4:
        sys.exit(f"[evaluate] HARD STOP: {len(rows_h)} groups != 4")

    o_mse = float(np.mean([x[0] for x in ours]))
    o_mae = float(np.mean([x[1] for x in ours]))
    m20_mse = float(np.mean([x[0] for x in a20]))
    m20_mae = float(np.mean([x[1] for x in a20]))
    m21_mse = float(np.mean([x[0] for x in a21]))
    m21_mae = float(np.mean([x[1] for x in a21]))

    gate_mean("EXP-020 Aurora chrono", m20_mse, m20_mae,
              paper["exp020_aurora_chrono"]["reference_4h_mean_mse_float32"],
              paper["exp020_aurora_chrono"]["reference_4h_mean_mae_float32"])
    gate_mean("EXP-021 VisionTS raw order", m21_mse, m21_mae,
              paper["exp021_visionts_raw_order"]
              ["reference_4h_mean_mse_float32"],
              paper["exp021_visionts_raw_order"]
              ["reference_4h_mean_mae_float32"])
    print(f"[evaluate] anchor 4H-mean gates PASS: EXP-020 "
          f"{m20_mse:.6f}/{m20_mae:.6f}, EXP-021 {m21_mse:.6f}/{m21_mae:.6f}")

    row_mean = {
        "dataset": DS, "selected_P": p_ds,
        "ours_4h_mean_mse_f32": o_mse, "ours_4h_mean_mae_f32": o_mae,
        "exp020_aurora_chrono_4h_mean_mse": m20_mse,
        "exp020_aurora_chrono_4h_mean_mae": m20_mae,
        "diff_vs_exp020_mse": o_mse - m20_mse,
        "diff_vs_exp020_mae": o_mae - m20_mae,
        "rel_pct_vs_exp020_mse": rel(o_mse, m20_mse),
        "rel_pct_vs_exp020_mae": rel(o_mae, m20_mae),
        "exp021_visionts_raw_4h_mean_mse": m21_mse,
        "exp021_visionts_raw_4h_mean_mae": m21_mae,
        "diff_vs_exp021_mse": o_mse - m21_mse,
        "diff_vs_exp021_mae": o_mae - m21_mae,
        "rel_pct_vs_exp021_mse": rel(o_mse, m21_mse),
        "rel_pct_vs_exp021_mae": rel(o_mae, m21_mae),
    }

    def write_csv(path, rows):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv(os.path.join(C.RESULTS_DIR, "comparison_per_h.csv"), rows_h)
    write_csv(os.path.join(C.RESULTS_DIR, "comparison_per_mean.csv"),
              [row_mean])

    main_answer = (
        f"Under the SAME ascending test windows (2017-04..2024-03) and "
        f"standardization: VisionTS (EXP-022, P={p_ds}) 4H-mean MSE/MAE = "
        f"{o_mse:.6f}/{o_mae:.6f} vs measured Aurora (EXP-020) "
        f"{m20_mse:.6f}/{m20_mae:.6f} -> "
        + ("VisionTS lower" if o_mse < m20_mse and o_mae < m20_mae
           else "Aurora lower" if o_mse > m20_mse and o_mae > m20_mae
           else "mixed"))
    summary = {
        "exp_id": P.EXP_ID,
        "label": P.LABEL,
        "protocol_token": token,
        "protocol_sha256": proto_sha,
        "groups_done": len(rows_h),
        "groups_expected": 4,
        "batch": batch,
        "selected_periodicity": p_ds,
        "exp021_reference_P_raw_order":
            spec["exp021_selected_P_raw_order"],
        "selection_scores": {cs: c["per_H_mean_mse_float32"]
                             for cs, c in
                             sel["datasets"][DS]["candidates"].items()},
        "test_period_ascending": "2017-04 .. 2024-03",
        "test_period_exp021_raw_order": "1989-01 .. 1995-12",
        "per_h_comparison": rows_h,
        "four_h_mean_comparison": row_mean,
        "main_answer": main_answer,
        "metric_primary": "official float32 replica; float64 appendix, "
                          "no gate",
        "anchor_gates": {
            "exp020_aurora_chrono_4h_mean": {
                "mse": m20_mse, "mae": m20_mae,
                "ref_mse": paper["exp020_aurora_chrono"]
                ["reference_4h_mean_mse_float32"],
                "ref_mae": paper["exp020_aurora_chrono"]
                ["reference_4h_mean_mae_float32"], "atol": 1e-5,
                "ok": True},
            "exp021_visionts_raw_4h_mean": {
                "mse": m21_mse, "mae": m21_mae,
                "ref_mse": paper["exp021_visionts_raw_order"]
                ["reference_4h_mean_mse_float32"],
                "ref_mae": paper["exp021_visionts_raw_order"]
                ["reference_4h_mean_mae_float32"], "atol": 1e-5,
                "ok": True},
        },
        "interpretation_boundary": [
            "vs EXP-020 Aurora: same ascending windows/standardization, "
            "but the comparison spans model AND input modality (univariate "
            "numeric vs text+numeric) — cannot be attributed to the text "
            "modality alone",
            "vs EXP-021 VisionTS: different test period (2017-04..2024-03 "
            "vs 1989-01..1995-12), different history window, different "
            "standardization statistics and different selection basis — "
            "describes the two protocols, not row order alone",
            "P selected on the ascending validation split (2013-09.."
            "2017-03) by the VisionTS paper's own method; EXP-021's "
            "raw-order choice P=12 is a reference only",
        ],
        "anchors": {
            "exp020": {"exp_id": "EXP-020",
                       "protocol_token": P.BASE_DATA["protocol_token"],
                       "freeze_commit": P.BASE_DATA["freeze_commit"],
                       "metrics": "recomputed read-only from frozen "
                                  "EXP-020 npz"},
            "exp021": {"exp_id": "EXP-021",
                       "protocol_token": P.BASE_METHOD["protocol_token"],
                       "freeze_commit": P.BASE_METHOD["freeze_commit"],
                       "metrics": "recomputed read-only from frozen "
                                  "EXP-021 Economy npz"},
        },
    }
    C.dump_json(summary, os.path.join(C.RESULTS_DIR, "summary.json"))
    print(f"[evaluate] main: {main_answer}")
    print(f"EVALUATE OK: {len(rows_h)}/4 groups")


if __name__ == "__main__":
    main()
