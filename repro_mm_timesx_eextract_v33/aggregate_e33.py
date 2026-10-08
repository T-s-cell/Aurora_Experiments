#!/usr/bin/env python3
"""Aggregation for M48T512_E33 vs EXP-012 M48T512_D2 (references: M48T512,
A48, Z0 recomputed read-only from the frozen shards).

Same aggregation ladder as EXP-012 (window -> variable -> within-domain equal
weight -> 19 domains equal weight -> 3-seed mean +/- std(ddof=1); best seed
never selected). PAIRED COMPARISON = M48T512_E33 - M48T512_D2 (same seed,
same composite key), never a plain average over windows. Negative delta =
E-Extract improves.

E33-only validations before reading:
  - every fallback/kept window's prediction is BITWISE equal to the D2
    prediction for the same (seed, key) — its input is bitwise the frozen D2
    row, so this must hold;
  - input-change ratio: rows of the E33 cache that differ from the D2 cache,
    and how many of those windows actually changed their prediction.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

SUBDIR = Path(__file__).resolve().parent
PROJECT = SUBDIR.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(SUBDIR))
sys.path.insert(0, str(PROJECT / "repro_mm_timesx_v1"))
sys.path.insert(0, str(PROJECT / "repro_mm_timesx_d2"))

from aggregate import (METRICS, dom_overall, fmt, load_set, merge_var_tables,  # noqa: E402
                       seed_stats, var_table_of)
from data_loader import DOMAIN_NAMES  # noqa: E402
from load_aurora import sha256_file  # noqa: E402
from predict import build_expectation, verify_shard_coverage  # noqa: E402

RESULTS = SUBDIR / "results"
PRED_E33 = SUBDIR / "predictions"
PRED_D2 = PROJECT / "repro_mm_timesx_d2" / "predictions"
PRED_MM = PROJECT / "repro_mm_timesx_v1" / "predictions"
PRED_ROOT = PROJECT / "predictions"
Z0_REF = PROJECT / "data" / "Z0__test.npz"
E33_METHOD = "M48T512_E33"
D2_METHOD = "M48T512_D2"
MM_METHOD = "M48T512"
A48_METHOD = "A48"
SEEDS = (2021, 2022, 2023)
ITL = 48


class AggregateError(RuntimeError):
    pass


def load_set_keys(path):
    with np.load(path, allow_pickle=False) as z:
        pred, target, d = z["pred"], z["target"], z["d"]
        vks = [str(x) for x in z["var_keys"]]
        sids = [str(x) for x in z["sample_ids"]]
    err = pred - target
    return {"keys": list(zip(vks, sids)),
            "std_mse": np.mean((err / d[:, None]) ** 2, axis=1),
            "std_mae": np.mean(np.abs(err) / d[:, None], axis=1),
            "pred": pred}


def collect_shards(pred_dir, method, seeds=SEEDS):
    per_seed = {}
    for seed in seeds:
        paths = sorted(Path(pred_dir).glob(f"{method}__*__s{seed}__test.npz"))
        if len(paths) != len(DOMAIN_NAMES):
            raise AggregateError(f"{method} s{seed}: {len(paths)} shards in "
                                 f"{pred_dir}, expected {len(DOMAIN_NAMES)}")
        per_seed[seed] = paths
    return per_seed


def verify_fingerprinted_shards(pred_dir, method, protocol, fingerprint_fn):
    import data_loader
    rows_all = data_loader.TimesXData(PROJECT / "data").test_rows()
    by_domain = {}
    for r in rows_all:
        by_domain.setdefault(r[2], []).append(r)
    expect_by_domain = {d: build_expectation(rs) for d, rs in by_domain.items()}
    checked = 0
    for seed in SEEDS:
        fp = fingerprint_fn(protocol, ITL, seed, protocol["model"]["weights_sha256"])
        for domain in DOMAIN_NAMES:
            name = f"{method}__{domain}__s{seed}__test.npz"
            npz = Path(pred_dir) / name
            header = json.loads((Path(pred_dir) / f"{name}.header.json").read_text())
            done = json.loads((Path(pred_dir) / f"{name}.done").read_text())
            if header.get("fingerprint") != fp or done.get("fingerprint") != fp:
                raise AggregateError(f"{name}: fingerprint mismatch vs "
                                     f"recomputed frozen fingerprint")
            content = sha256_file(npz)
            if header.get("content_sha256") != content or \
               done.get("content_sha256") != content:
                raise AggregateError(f"{name}: npz content sha256 mismatch")
            with np.load(npz, allow_pickle=False) as z:
                entries = [(str(z["var_keys"][i]), str(z["sample_ids"][i]),
                            z["target"][i], float(z["d"][i]))
                           for i in range(len(z["var_keys"]))]
                if not np.isfinite(z["pred"]).all():
                    raise AggregateError(f"{name}: non-finite predictions")
            verify_shard_coverage(entries, expect_by_domain[domain])
            checked += 1
    return checked


def window_maps(pred_dir, method):
    """{seed: {(vk,sid): (w_mse, w_mae)}} + {seed: {(vk,sid): pred}}."""
    out, preds = {}, {}
    for seed, paths in collect_shards(pred_dir, method).items():
        m, pm = {}, {}
        for p in paths:
            s = load_set_keys(p)
            for i, (key, mse, mae) in enumerate(zip(s["keys"], s["std_mse"],
                                                    s["std_mae"])):
                if key in m:
                    raise AggregateError(f"duplicate window key in shards: {key}")
                m[key] = (float(mse), float(mae))
                pm[key] = s["pred"][i]
        out[seed] = m
        preds[seed] = pm
    return out, preds


def paired_ladder(a_maps, b_maps, dom_of_var):
    """Window paired diff (a - b) -> variable mean -> domain equal weight ->
    overall."""
    dom_by_seed, overall_by_seed = {}, {}
    for seed in SEEDS:
        a, b = a_maps[seed], b_maps[seed]
        if set(a) != set(b):
            raise AggregateError(f"s{seed}: window key sets differ")
        per_var = {}
        for key, (a_mse, a_mae) in a.items():
            b_mse, b_mae = b[key]
            pv = per_var.setdefault(key[0], {"mse": [], "mae": []})
            pv["mse"].append(a_mse - b_mse)
            pv["mae"].append(a_mae - b_mae)
        var_mean = {vk: {k: float(np.mean(v[k])) for k in ("mse", "mae")}
                    for vk, v in per_var.items()}
        dom_t = {}
        for d in DOMAIN_NAMES:
            vs = [var_mean[vk] for vk in var_mean if dom_of_var[vk] == d]
            if not vs:
                raise AggregateError(f"no variables for domain {d}")
            dom_t[d] = {k: float(np.mean([v[k] for v in vs])) for k in ("mse", "mae")}
        overall = {k: float(np.mean([dom_t[d][k] for d in dom_t]))
                   for k in ("mse", "mae")}
        dom_by_seed[seed] = dom_t
        overall_by_seed[seed] = overall
    return dom_by_seed, overall_by_seed


def main():
    from predict_d2 import protocol_d2_sha, shard_fingerprint_d2
    from predict_e33 import protocol_e33_sha, shard_fingerprint_e33

    RESULTS.mkdir(exist_ok=True)
    protocol_e33 = json.loads((SUBDIR / "protocol_e33.json").read_text())
    protocol_d2 = json.loads((PROJECT / "repro_mm_timesx_d2" / "protocol_d2.json").read_text())
    if protocol_d2_sha() != protocol_e33["e33_provenance"]["d2_protocol_sha256"]:
        raise AggregateError("protocol_d2.json sha differs from the value "
                             "recorded in protocol_e33.e33_provenance")

    n1 = verify_fingerprinted_shards(
        PRED_E33, E33_METHOD, protocol_e33,
        lambda p, itl, s, w: shard_fingerprint_e33(p, itl, s, w))
    n2 = verify_fingerprinted_shards(
        PRED_D2, D2_METHOD, protocol_d2,
        lambda p, itl, s, w: shard_fingerprint_d2(p, itl, s, w))
    print(f"[aggregate_e33] shard verification OK: {n1} {E33_METHOD} + "
          f"{n2} {D2_METHOD} shards")

    e33_maps, e33_preds = window_maps(PRED_E33, E33_METHOD)
    d2_maps, d2_preds = window_maps(PRED_D2, D2_METHOD)
    dom_of_var = {}
    for vk, _sid in e33_maps[SEEDS[0]]:
        dom_of_var[vk] = vk.split("__", 1)[0]

    # ---- fallback/kept windows: predictions must be bitwise equal to D2 ----
    outcomes = {}
    with open(SUBDIR / "cache" / "e33_windows_test.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            outcomes[(r["var_key"], r["sample_id"])] = r["outcome"]
    assert len(outcomes) == 2474
    fallback_keys = {k for k, o in outcomes.items() if o != "compressed"}
    n_fb_bad = 0
    for seed in SEEDS:
        for k in fallback_keys:
            if not np.array_equal(e33_preds[seed][k], d2_preds[seed][k]):
                n_fb_bad += 1
    print(f"[aggregate_e33] fallback/kept windows: {len(fallback_keys)}; "
          f"bitwise pred mismatches vs D2: {n_fb_bad}")
    if n_fb_bad:
        raise AggregateError(f"{n_fb_bad} fallback windows have predictions "
                             f"differing from D2 (must be bitwise equal)")

    # ---- input-change ratio + prediction-change among changed windows ----
    ze = np.load(SUBDIR / "cache" / "text_tokens_M48T512_E33.npz", allow_pickle=False)
    zd = np.load(PROJECT / "repro_mm_timesx_d2" / "cache" /
                 "text_tokens_M48T512_D2.npz", allow_pickle=False)
    changed_mask = ~((ze["ids"] == zd["ids"]).all(axis=1)) | \
                   ~((ze["mask"] == zd["mask"]).all(axis=1))
    n_changed_inputs = int(changed_mask.sum())
    pred_changed_by_seed = {}
    keys_list = list(zip([str(x) for x in ze["var_keys"]],
                         [str(x) for x in ze["sample_ids"]]))
    for seed in SEEDS:
        c = 0
        for i, k in enumerate(keys_list):
            if changed_mask[i] and not np.array_equal(e33_preds[seed][k],
                                                      d2_preds[seed][k]):
                c += 1
        pred_changed_by_seed[seed] = c

    # ---- ladder tables ------------------------------------------------------
    sets = {"Z0": load_set(Z0_REF)}
    for m, pred_dir in ((E33_METHOD, PRED_E33), (D2_METHOD, PRED_D2),
                        (MM_METHOD, PRED_MM), (A48_METHOD, PRED_ROOT)):
        for seed, paths in collect_shards(pred_dir, m).items():
            for p in paths:
                sets[p.stem.replace("__test", "")] = load_set(p)

    z0_var = var_table_of(sets["Z0"])
    z0_dom, z0_ov = dom_overall(z0_var)

    seed_merged, seed_dom, seed_ov = {}, {}, {}
    for m in (E33_METHOD, D2_METHOD, MM_METHOD, A48_METHOD):
        for seed in SEEDS:
            rids = sorted(k for k in sets
                          if k.startswith(f"{m}__") and k.endswith(f"s{seed}"))
            merged = merge_var_tables([var_table_of(sets[rid]) for rid in rids])
            seed_merged[(m, seed)] = merged
            seed_dom[(m, seed)], seed_ov[(m, seed)] = dom_overall(merged)

    overall_rows = [["Z0", "single", fmt(z0_ov["mse"], None), fmt(z0_ov["mae"], None)]]
    for m in (A48_METHOD, MM_METHOD, D2_METHOD, E33_METHOD):
        mo = {k: seed_stats([seed_ov[(m, s)][k] for s in SEEDS]) for k in ("mse", "mae")}
        overall_rows.append([m, "mean+-std over 3 seeds", fmt(*mo["mse"]), fmt(*mo["mae"])])

    domain_rows = [["Z0", d, "single", fmt(z0_dom[d]["mse"], None),
                    fmt(z0_dom[d]["mae"], None)] for d in DOMAIN_NAMES]
    for m in (E33_METHOD, D2_METHOD):
        for d in DOMAIN_NAMES:
            domain_rows.append([m, d, "mean+-std",
                                fmt(*seed_stats([seed_dom[(m, s)][d]["mse"] for s in SEEDS])),
                                fmt(*seed_stats([seed_dom[(m, s)][d]["mae"] for s in SEEDS]))])

    all_vars = sorted(z0_var)
    assert len(all_vars) == 190
    var_rows = []
    for m in (E33_METHOD, D2_METHOD):
        for vk in all_vars:
            per_seed = [seed_merged[(m, s)][vk] for s in SEEDS]
            ms = {k: [t[k] for t in per_seed] for k in METRICS}
            var_rows.append([m, vk, per_seed[0]["n"]] +
                            [fmt(*seed_stats(ms[k])) for k in METRICS])

    # ---- paired E33 - D2 ----------------------------------------------------
    pdom, pov = paired_ladder(e33_maps, d2_maps, dom_of_var)
    p_mse = seed_stats([pov[s]["mse"] for s in SEEDS])
    p_mae = seed_stats([pov[s]["mae"] for s in SEEDS])
    paired_rows = []
    for d in DOMAIN_NAMES:
        paired_rows.append([d, "mean+-std over 3 seeds",
                            fmt(*seed_stats([pdom[s][d]["mse"] for s in SEEDS])),
                            fmt(*seed_stats([pdom[s][d]["mae"] for s in SEEDS]))])
    paired_rows.append(["OVERALL(19-domain equal weight)", "mean+-std over 3 seeds",
                        fmt(*p_mse), fmt(*p_mae)])

    comp_rows = []
    ov_mean = {m: {k: float(np.mean([seed_ov[(m, s)][k] for s in SEEDS]))
                   for k in ("mse", "mae")}
               for m in (E33_METHOD, D2_METHOD, MM_METHOD, A48_METHOD)}
    for metric in ("mse", "mae"):
        delta = ov_mean[E33_METHOD][metric] - ov_mean[D2_METHOD][metric]
        base = ov_mean[D2_METHOD][metric]
        rel = f"{-delta / base * 100:+.2f}%" if base != 0 else "ref=0"
        comp_rows.append([f"{E33_METHOD}-{D2_METHOD}", metric,
                          f"{delta:+.6f}", rel])
    for metric, st in (("std_MSE", p_mse), ("std_MAE", p_mae)):
        base = ov_mean[D2_METHOD]["mse" if metric == "std_MSE" else "mae"]
        rel = f"{-st[0] / base * 100:+.2f}%" if base != 0 else "ref=0"
        comp_rows.append([f"{E33_METHOD}-{D2_METHOD}-paired", metric,
                          f"{st[0]:+.6f}", rel])

    # per-domain improved/degraded counts vs D2
    impr_rows = []
    for metric in ("mse", "mae"):
        improved = degraded = equal = 0
        for d in DOMAIN_NAMES:
            v = float(np.mean([seed_dom[(E33_METHOD, s)][d][metric] for s in SEEDS]))
            r = float(np.mean([seed_dom[(D2_METHOD, s)][d][metric] for s in SEEDS]))
            improved += v < r
            degraded += v > r
            equal += v == r
        impr_rows.append([E33_METHOD, f"vs {D2_METHOD}", "domain(19)", metric,
                          improved, degraded, equal, 19])
        improved = degraded = equal = 0
        for vk in all_vars:
            v = float(np.mean([seed_merged[(E33_METHOD, s)][vk][metric] for s in SEEDS]))
            r = float(np.mean([seed_merged[(D2_METHOD, s)][vk][metric] for s in SEEDS]))
            improved += v < r
            degraded += v > r
            equal += v == r
        impr_rows.append([E33_METHOD, f"vs {D2_METHOD}", "var(190)", metric,
                          improved, degraded, equal, 190])

    def w_csv(name, header, rows):
        with open(RESULTS / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    w_csv("overall.csv", ["method", "seed_stat", "MSE", "MAE"], overall_rows)
    w_csv("domain_summary.csv", ["method", "domain", "seed_stat", "MSE", "MAE"], domain_rows)
    w_csv("variable_level.csv",
          ["method", "var_key", "n_windows", "std_MSE", "std_MAE", "raw_MSE", "raw_MAE"],
          var_rows)
    w_csv("paired_deltas_vs_d2.csv",
          ["domain", "seed_stat", "paired_dMSE", "paired_dMAE"], paired_rows)
    w_csv("improvement_counts.csv",
          ["method", "reference", "level", "metric", "improved", "degraded", "equal", "total"],
          impr_rows)
    w_csv("comparisons.csv",
          ["comparison", "metric", "delta", "relative_improvement"], comp_rows)

    outcome_counts = {}
    for o in outcomes.values():
        outcome_counts[o] = outcome_counts.get(o, 0) + 1
    summary = {
        "e33_overall": {"mse": list(seed_stats([seed_ov[(E33_METHOD, s)]["mse"] for s in SEEDS])),
                        "mae": list(seed_stats([seed_ov[(E33_METHOD, s)]["mae"] for s in SEEDS]))},
        "d2_overall": {"mse": list(seed_stats([seed_ov[(D2_METHOD, s)]["mse"] for s in SEEDS])),
                       "mae": list(seed_stats([seed_ov[(D2_METHOD, s)]["mae"] for s in SEEDS]))},
        "paired_vs_d2_overall": {"dMSE": list(p_mse), "dMAE": list(p_mae)},
        "fallback_kept_windows": len(fallback_keys),
        "fallback_pred_bitwise_mismatches": n_fb_bad,
        "outcomes": outcome_counts,
        "windows_input_changed_vs_d2": n_changed_inputs,
        "windows_pred_changed_vs_d2_by_seed": pred_changed_by_seed,
        "shards_verified": {E33_METHOD: n1, D2_METHOD: n2},
        "ladder": "paired window diff (M48T512_E33 - M48T512_D2) -> variable "
                  "mean -> domain equal weight -> 19-domain equal weight",
    }
    (RESULTS / "aggregate_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))
    print(f"[aggregate_e33] wrote 6 tables to {RESULTS}")


if __name__ == "__main__":
    try:
        main()
    except AggregateError as e:
        print(f"[aggregate_e33] FAILED: {e}")
        sys.exit(1)
