#!/usr/bin/env python3
"""Scoring + VisionTS comparison (float64, original units).

Ladder aggregation (identical code path to VisionTS EXP-016): prediction
points -> windows -> sequences (equal weight inside each subset); windows
in CSV row order, sequences ascending Idx, numpy float64 means over
contiguous arrays. Per seed: {ds}_metrics__s{seed}.json + sequence CSV;
across the 3 seeds: mean +- std (ddof=1) — predictions of different seeds
are never averaged, no best-seed selection. The VisionTS reference (frozen
EXP-016 predictions under visionts_ref/) is scored with the SAME ladder;
deltas are Aurora - VisionTS at subset level (per-seed and seed-mean) and
sequence level (Aurora seed-mean - VisionTS). Three subsets reported
separately; NO cross-subset total; no comparison to ChatTime tables."""
import csv
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for p in (REPO_ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

import common as C
import protocol as P


def load_all_shards(ds, seed):
    rows = []
    for shard in range(P.n_shards(ds)):
        npz, hdr, done = P.shard_paths(ds, seed, shard)
        with np.load(npz) as z:
            idx = z["idx"]
            date = z["date"]
            pred = z["pred"]
            tgt = z["target"]
        for i in range(len(idx)):
            rows.append((int(idx[i]), str(date[i]), pred[i], tgt[i]))
    return rows


def load_vts_shards(ds):
    """VisionTS EXP-016 reference rows from visionts_ref/predictions/
    ({ds}__shardXXX.npz, EXP-016 shard naming), in shard order."""
    rows = []
    for shard in range(P.n_shards(ds)):
        npz = os.path.join(P.VTS_REF_DIR, "predictions",
                           f"{ds}__shard{shard:03d}.npz")
        with np.load(npz) as z:
            idx = z["idx"]
            date = z["date"]
            pred = z["pred"]
            tgt = z["target"]
        for i in range(len(idx)):
            rows.append((int(idx[i]), str(date[i]), pred[i], tgt[i]))
    return rows


def ladder(rows):
    """rows in fixed order -> (subset mse/mae via sequence-equal-weight,
    point-level reference, per-sequence metrics list)."""
    seq_m = defaultdict(list)   # idx -> [(mse_w, mae_w)] in row order
    all_sq, all_ab = [], []
    for ix, _date, pred, tgt in rows:
        d = pred - tgt
        mse_w = float(np.mean(d ** 2))
        mae_w = float(np.mean(np.abs(d)))
        seq_m[ix].append((mse_w, mae_w))
        all_sq.append(d ** 2)
        all_ab.append(np.abs(d))
    seqs = []
    for ix in sorted(seq_m):
        arr = np.array(seq_m[ix], dtype=np.float64)   # [n_win, 2]
        seqs.append((ix, len(seq_m[ix]),
                     float(arr[:, 0].mean()), float(arr[:, 1].mean())))
    mse = float(np.mean([s[2] for s in seqs]))
    mae = float(np.mean([s[3] for s in seqs]))
    sq = np.concatenate(all_sq)
    ab = np.concatenate(all_ab)
    return mse, mae, (float(sq.mean()), float(ab.mean())), seqs


def write_sequence_csv(path, seqs):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Idx", "n_windows", "MSE", "MAE"])
        for ix, n, m, a in seqs:
            w.writerow([ix, n, repr(m), repr(a)])


def main():
    os.makedirs(P.RESULTS_DIR, exist_ok=True)
    summary = {"protocol_name": P.PROTOCOL_NAME,
               "ladder": "point->window->sequence-equal-weight (float64, "
                         "original units; no cross-subset total)",
               "seeds": list(P.SEEDS),
               "aggregation": P.AGGREGATION,
               "subsets": {}}
    for ds in P.ORDER:
        Pl_wanted = P.EXPECTED_WINDOWS[ds]
        per_seed, seq_by_seed = {}, {}
        for seed in P.SEEDS:
            rows = load_all_shards(ds, seed)
            if len(rows) != Pl_wanted:
                sys.exit(f"{ds} s{seed}: {len(rows)} windows != "
                         f"{Pl_wanted} (run verify_cache)")
            mse, mae, point, seqs = ladder(rows)
            res = {"subset": ds, "seed": seed, "n_windows": len(rows),
                   "n_sequences": len(seqs),
                   "mse_seq_equal_weight": mse,
                   "mae_seq_equal_weight": mae,
                   "mse_point_level_ref": point[0],
                   "mae_point_level_ref": point[1]}
            C.dump_json(res, os.path.join(
                P.RESULTS_DIR, f"{ds}_metrics__s{seed}.json"))
            write_sequence_csv(os.path.join(
                P.RESULTS_DIR, f"sequence_level_{ds}__s{seed}.csv"), seqs)
            np.savez(os.path.join(P.RESULTS_DIR,
                                  f"window_level_{ds}__s{seed}.npz"),
                     idx=np.array([r[0] for r in rows], dtype=np.int64),
                     date=np.array([r[1] for r in rows]),
                     mse_w=np.array([float(np.mean((r[2] - r[3]) ** 2))
                                     for r in rows], dtype=np.float64),
                     mae_w=np.array([float(np.mean(np.abs(r[2] - r[3])))
                                     for r in rows], dtype=np.float64))
            per_seed[seed] = res
            seq_by_seed[seed] = {ix: (m, a) for ix, _n, m, a in seqs}
            print(f"[eval] {ds} s{seed}: MSE {mse:.6f}  MAE {mae:.6f}  "
                  f"({len(rows)} windows, {len(seqs)} sequences)")

        mses = [per_seed[s]["mse_seq_equal_weight"] for s in P.SEEDS]
        maes = [per_seed[s]["mae_seq_equal_weight"] for s in P.SEEDS]
        agg = {"subset": ds, "seeds": list(P.SEEDS),
               "n_windows": Pl_wanted,
               "n_sequences": per_seed[min(P.SEEDS)]["n_sequences"],
               "mse_seq_equal_weight_mean": float(np.mean(mses)),
               "mse_seq_equal_weight_std": float(np.std(mses, ddof=1)),
               "mae_seq_equal_weight_mean": float(np.mean(maes)),
               "mae_seq_equal_weight_std": float(np.std(maes, ddof=1)),
               "per_seed": {str(s): {
                   "mse_seq_equal_weight":
                       per_seed[s]["mse_seq_equal_weight"],
                   "mae_seq_equal_weight":
                       per_seed[s]["mae_seq_equal_weight"]}
                   for s in P.SEEDS}}

        # --- VisionTS reference (frozen EXP-016) with the same ladder ---
        vrows = load_vts_shards(ds)
        if len(vrows) != Pl_wanted:
            sys.exit(f"{ds}: visionts_ref has {len(vrows)} windows != "
                     f"{Pl_wanted} (check visionts_ref deployment)")
        v_mse, v_mae, _vp, v_seqs = ladder(vrows)
        v_seq = {ix: (m, a) for ix, _n, m, a in v_seqs}
        agg["visionts_ref"] = {"mse_seq_equal_weight": v_mse,
                               "mae_seq_equal_weight": v_mae}
        agg["delta_mean_minus_visionts"] = {
            "mse": agg["mse_seq_equal_weight_mean"] - v_mse,
            "mae": agg["mae_seq_equal_weight_mean"] - v_mae}
        agg["delta_per_seed"] = {
            str(s): {"mse": per_seed[s]["mse_seq_equal_weight"] - v_mse,
                     "mae": per_seed[s]["mae_seq_equal_weight"] - v_mae}
            for s in P.SEEDS}

        # sequence-level comparison (Aurora seed-mean vs VisionTS)
        v_nwin = {ix: n for ix, n, _m, _a in v_seqs}
        with open(os.path.join(P.RESULTS_DIR,
                               f"comparison_{ds}.csv"), "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["Idx", "n_windows",
                        "aurora_mse_mean", "aurora_mae_mean",
                        "visionts_mse", "visionts_mae",
                        "delta_mse", "delta_mae"])
            for ix in sorted(v_seq):
                am = float(np.mean([seq_by_seed[s][ix][0] for s in P.SEEDS]))
                aa = float(np.mean([seq_by_seed[s][ix][1] for s in P.SEEDS]))
                w.writerow([ix, v_nwin.get(ix, ""), repr(am), repr(aa),
                            repr(v_seq[ix][0]), repr(v_seq[ix][1]),
                            repr(am - v_seq[ix][0]), repr(aa - v_seq[ix][1])])

        C.dump_json(agg, os.path.join(P.RESULTS_DIR, f"{ds}_metrics.json"))
        summary["subsets"][ds] = agg
        print(f"[eval] {ds}: Aurora {agg['mse_seq_equal_weight_mean']:.6f}"
              f"+-{agg['mse_seq_equal_weight_std']:.6f} / "
              f"{agg['mae_seq_equal_weight_mean']:.6f}"
              f"+-{agg['mae_seq_equal_weight_std']:.6f}  "
              f"VisionTS MSE {v_mse:.6f} MAE {v_mae:.6f}  "
              f"delta MSE "
              f"{agg['delta_mean_minus_visionts']['mse']:+.6f} MAE "
              f"{agg['delta_mean_minus_visionts']['mae']:+.6f}")
    C.dump_json(summary, os.path.join(P.RESULTS_DIR, "summary.json"))
    print("[eval] wrote summary.json + per-seed metrics/CSVs + comparison")


if __name__ == "__main__":
    main()
