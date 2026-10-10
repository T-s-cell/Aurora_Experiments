#!/usr/bin/env python3
"""Independent metric recomputation (standalone pass).

Deliberately does NOT import or reuse evaluate.py's ladder code: reads the
Aurora shard npz files directly (pred + saved target), recomputes the
ladder with the SAME documented fixed order (shard asc, CSV row order
within a shard; sequences ascending Idx), and compares against
evaluate.py's stored per-seed results at the double tolerance
rtol=atol=1e-12 (identical reduction order should give bitwise equality;
the tolerance only absorbs library-level differences). Also cross-checks
the seed-mean/std aggregation in {ds}_metrics.json. Reports the maximum
absolute/relative difference. Writes results/recompute_check.json."""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for p in (HERE,):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

import protocol as P

RTOL = 1e-12
ATOL = 1e-12


def recompute_subset(ds, seed):
    seq_m = {}
    for shard in range(P.n_shards(ds)):
        npz_path = P.shard_paths(ds, seed, shard)[0]
        with np.load(npz_path) as z:
            idx = z["idx"]
            pred = z["pred"].astype(np.float64)
            tgt = z["target"].astype(np.float64)
        for i in range(len(idx)):
            ix = int(idx[i])
            d = pred[i] - tgt[i]
            item = (float(np.mean(d ** 2)), float(np.mean(np.abs(d))))
            seq_m.setdefault(ix, []).append(item)
    seqs = []
    for ix in sorted(seq_m):
        a = np.array(seq_m[ix], dtype=np.float64)
        seqs.append((ix, a[:, 0].mean(), a[:, 1].mean()))
    mse = float(np.mean([s[1] for s in seqs]))
    mae = float(np.mean([s[2] for s in seqs]))
    return mse, mae, seqs


def close(got, ref):
    got, ref = float(got), float(ref)
    ad = abs(got - ref)
    rd = ad / abs(ref) if ref != 0 else ad
    return (ad <= ATOL + RTOL * abs(ref)), ad, rd


def main():
    rep = {"rtol": RTOL, "atol": ATOL, "subsets": {}}
    all_ok, max_ad = True, 0.0
    for ds in P.ORDER:
        per_seed = {}
        agg_bad = 0
        for seed in P.SEEDS:
            mse, mae, seqs = recompute_subset(ds, seed)
            with open(os.path.join(P.RESULTS_DIR,
                                   f"{ds}_metrics__s{seed}.json")) as f:
                ref = json.load(f)
            with open(os.path.join(P.RESULTS_DIR,
                                   f"sequence_level_{ds}__s{seed}.csv")) as f:
                r = list(csv.reader(f))[1:]
            ref_seq = {int(row[0]): (float(row[2]), float(row[3]))
                       for row in r}
            seq_bad, seq_max_ad = 0, 0.0
            for ix, m, a in seqs:
                for got, expect in ((m, ref_seq[ix][0]), (a, ref_seq[ix][1])):
                    ok, ad, _ = close(got, expect)
                    seq_max_ad = max(seq_max_ad, ad)
                    if not ok:
                        seq_bad += 1
            m_ok, m_ad, m_rd = close(mse, ref["mse_seq_equal_weight"])
            a_ok, a_ad, a_rd = close(mae, ref["mae_seq_equal_weight"])
            ok = m_ok and a_ok and seq_bad == 0 and \
                len(seqs) == ref["n_sequences"]
            all_ok &= ok
            max_ad = max(max_ad, m_ad, a_ad, seq_max_ad)
            per_seed[str(seed)] = {
                "mse": {"got": mse, "ref": ref["mse_seq_equal_weight"],
                        "abs_diff": m_ad, "rel_diff": m_rd, "ok": bool(m_ok)},
                "mae": {"got": mae, "ref": ref["mae_seq_equal_weight"],
                        "abs_diff": a_ad, "rel_diff": a_rd, "ok": bool(a_ok)},
                "sequence_entries_mismatched": seq_bad,
                "max_seq_abs_diff": seq_max_ad,
                "n_sequences": len(seqs),
                "ok": bool(ok),
            }
            print(f"[recompute] {ds} s{seed}: dMSE_abs={m_ad:.3e} "
                  f"dMAE_abs={a_ad:.3e} seq_mismatch={seq_bad} "
                  f"-> {'OK' if ok else 'FAIL'}")
        # aggregate cross-check: recomputed seed mean/std vs {ds}_metrics
        with open(os.path.join(P.RESULTS_DIR, f"{ds}_metrics.json")) as f:
            agg = json.load(f)
        mses = [per_seed[str(s)]["mse"]["got"] for s in P.SEEDS]
        maes = [per_seed[str(s)]["mae"]["got"] for s in P.SEEDS]
        m_ok, m_ad, _ = close(float(np.mean(mses)),
                              agg["mse_seq_equal_weight_mean"])
        a_ok, a_ad, _ = close(float(np.mean(maes)),
                              agg["mae_seq_equal_weight_mean"])
        s_ok, ms_ad, _ = close(float(np.std(mses, ddof=1)),
                               agg["mse_seq_equal_weight_std"])
        sa_ok, ma_ad, _ = close(float(np.std(maes, ddof=1)),
                                agg["mae_seq_equal_weight_std"])
        agg_ok = m_ok and a_ok and s_ok and sa_ok
        agg_bad = 0 if agg_ok else 4
        all_ok &= agg_ok
        max_ad = max(max_ad, m_ad, a_ad, ms_ad, ma_ad)
        rep["subsets"][ds] = {
            "per_seed": per_seed,
            "aggregate_mean_std_ok": bool(agg_ok),
            "aggregate_max_abs_diff": max(m_ad, a_ad, ms_ad, ma_ad),
            "aggregate_entries_mismatched": agg_bad,
            "ok": bool(all(per_seed[str(s)]["ok"] for s in P.SEEDS)
                       and agg_ok),
        }
        print(f"[recompute] {ds}: aggregate mean/std dMSE_abs={m_ad:.3e} "
              f"dMAE_abs={a_ad:.3e} -> {'OK' if agg_ok else 'FAIL'}")
    rep["max_abs_diff_overall"] = max_ad
    rep["ok"] = bool(all_ok)
    C_json = os.path.join(P.RESULTS_DIR, "recompute_check.json")
    with open(C_json + ".tmp", "w") as f:
        json.dump(rep, f, indent=1)
    os.replace(C_json + ".tmp", C_json)
    print(f"[recompute] max abs diff overall = {max_ad:.3e} "
          f"(double tolerance rtol=atol={ATOL:.0e}) -> "
          f"{'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
