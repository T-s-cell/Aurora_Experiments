#!/usr/bin/env python3
"""Aggregation for M48T512 vs A48 vs Z0.

Reuses the root aggregate.py pure helpers verbatim (same aggregation ladder:
window -> variable -> within-domain equal weight -> 19-domain equal weight ->
3-seed mean +/- std(ddof=1); best seed never selected).

A48 is recomputed AT FULL PRECISION from the frozen root predictions/ shards
(never from display CSVs); Z0 from data/Z0__test.npz; M48T512 from the subdir
shards. Paired deltas (M48T512 - A48, same seed, same composite key) follow the
identical ladder — window diff -> variable mean -> domain mean -> overall —
NEVER a plain average over all windows.
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

from aggregate import (METRICS, dom_overall, fmt, load_set, merge_var_tables,  # noqa: E402
                       seed_stats, var_table_of)
from data_loader import DOMAIN_NAMES  # noqa: E402

RESULTS = SUBDIR / "results"
PRED_MM = SUBDIR / "predictions"
PRED_ROOT = PROJECT / "predictions"
Z0_REF = PROJECT / "data" / "Z0__test.npz"
MM_METHOD = "M48T512"
A48_METHOD = "A48"
SEEDS = (2021, 2022, 2023)


class AggregateError(RuntimeError):
    pass


def load_set_keys(path):
    """load_set + composite keys + per-window std metrics (for pairing)."""
    with np.load(path, allow_pickle=False) as z:
        pred, target, d = z["pred"], z["target"], z["d"]
        vks = [str(x) for x in z["var_keys"]]
        sids = [str(x) for x in z["sample_ids"]]
    err = pred - target
    return {"keys": list(zip(vks, sids)),
            "std_mse": np.mean((err / d[:, None]) ** 2, axis=1),
            "std_mae": np.mean(np.abs(err) / d[:, None], axis=1)}


def collect_shards(pred_dir, method, seeds=SEEDS):
    per_seed = {}
    for seed in seeds:
        paths = sorted(Path(pred_dir).glob(f"{method}__*__s{seed}__test.npz"))
        if len(paths) != len(DOMAIN_NAMES):
            raise AggregateError(f"{method} s{seed}: {len(paths)} shards in "
                                 f"{pred_dir}, expected {len(DOMAIN_NAMES)}")
        per_seed[seed] = paths
    return per_seed


def window_maps(pred_dir, method):
    """{seed: {(vk,sid): (w_mse, w_mae)}} over the method's shards."""
    out = {}
    for seed, paths in collect_shards(pred_dir, method).items():
        m = {}
        for p in paths:
            s = load_set_keys(p)
            for key, mse, mae in zip(s["keys"], s["std_mse"], s["std_mae"]):
                if key in m:
                    raise AggregateError(f"duplicate window key in shards: {key}")
                m[key] = (float(mse), float(mae))
        out[seed] = m
    return out


def paired_ladder(mm_maps, a48_maps, dom_of_var):
    """Window paired diff -> variable mean -> domain equal weight -> overall.

    Returns (dom_by_seed, overall_by_seed): per seed, {domain: {mse, mae}} and
    overall {mse, mae} of the PAIRED differences."""
    dom_by_seed, overall_by_seed = {}, {}
    for seed in SEEDS:
        mm, a48 = mm_maps[seed], a48_maps[seed]
        if set(mm) != set(a48):
            raise AggregateError(f"s{seed}: window key sets differ between "
                                 f"M48T512 and A48")
        per_var = {}
        for key, (m_mse, m_mae) in mm.items():
            a_mse, a_mae = a48[key]
            vk = key[0]
            pv = per_var.setdefault(vk, {"mse": [], "mae": []})
            pv["mse"].append(m_mse - a_mse)
            pv["mae"].append(m_mae - a_mae)
        var_mean = {vk: {k: float(np.mean(v[k])) for k in ("mse", "mae")}
                    for vk, v in per_var.items()}
        dom_t = {}
        for d in DOMAIN_NAMES:
            vs = [var_mean[vk] for vk in var_mean if dom_of_var[vk] == d]
            if not vs:
                raise AggregateError(f"no variables for domain {d} in paired deltas")
            dom_t[d] = {k: float(np.mean([v[k] for v in vs])) for k in ("mse", "mae")}
        overall = {k: float(np.mean([dom_t[d][k] for d in dom_t])) for k in ("mse", "mae")}
        dom_by_seed[seed] = dom_t
        overall_by_seed[seed] = overall
    return dom_by_seed, overall_by_seed


def main(methods=(MM_METHOD,)):
    RESULTS.mkdir(exist_ok=True)
    if tuple(methods) != (MM_METHOD,):
        raise AggregateError(f"aggregate_mm handles exactly [{MM_METHOD!r}]")

    mm_maps = window_maps(PRED_MM, MM_METHOD)
    a48_maps = window_maps(PRED_ROOT, A48_METHOD)
    dom_of_var = {}
    for vk, _sid in mm_maps[SEEDS[0]]:
        dom_of_var[vk] = vk.split("__", 1)[0]

    sets = {"Z0": load_set(Z0_REF)}
    for m, pred_dir in ((MM_METHOD, PRED_MM), (A48_METHOD, PRED_ROOT)):
        for seed, paths in collect_shards(pred_dir, m).items():
            for p in paths:
                sets[p.stem.replace("__test", "")] = load_set(p)

    z0_var = var_table_of(sets["Z0"])
    z0_dom, z0_ov = dom_overall(z0_var)

    seed_merged, seed_dom, seed_ov = {}, {}, {}
    for m in (MM_METHOD, A48_METHOD):
        for seed in SEEDS:
            rids = sorted(k for k in sets if k.startswith(f"{m}__") and k.endswith(f"s{seed}"))
            merged = merge_var_tables([var_table_of(sets[rid]) for rid in rids])
            seed_merged[(m, seed)] = merged
            seed_dom[(m, seed)], seed_ov[(m, seed)] = dom_overall(merged)

    method_overall = {"Z0": {"mse": (z0_ov["mse"], None), "mae": (z0_ov["mae"], None)}}
    overall_rows = [["Z0", "single", fmt(z0_ov["mse"], None), fmt(z0_ov["mae"], None)]]
    for m in (A48_METHOD, MM_METHOD):
        mo = {k: seed_stats([seed_ov[(m, s)][k] for s in SEEDS]) for k in ("mse", "mae")}
        method_overall[m] = mo
        overall_rows.append([m, "mean+-std over 3 seeds", fmt(*mo["mse"]), fmt(*mo["mae"])])

    domain_rows = [["Z0", d, "single", fmt(z0_dom[d]["mse"], None),
                    fmt(z0_dom[d]["mae"], None)] for d in DOMAIN_NAMES]
    for m in (A48_METHOD, MM_METHOD):
        for d in DOMAIN_NAMES:
            mv = [seed_dom[(m, s)][d]["mse"] for s in SEEDS]
            av = [seed_dom[(m, s)][d]["mae"] for s in SEEDS]
            domain_rows.append([m, d, "mean+-std", fmt(*seed_stats(mv)), fmt(*seed_stats(av))])

    all_vars = sorted(z0_var)
    if len(all_vars) != 190:
        raise AggregateError(f"Z0 covers {len(all_vars)} vars, expected 190")
    var_rows = [["Z0", vk, z0_var[vk]["n"], fmt(z0_var[vk]["mse"], None),
                 fmt(z0_var[vk]["mae"], None), fmt(z0_var[vk]["raw_mse"], None),
                 fmt(z0_var[vk]["raw_mae"], None)] for vk in all_vars]
    for m in (A48_METHOD, MM_METHOD):
        for vk in all_vars:
            per_seed = [seed_merged[(m, s)][vk] for s in SEEDS]
            ms = {k: [t[k] for t in per_seed] for k in METRICS}
            var_rows.append([m, vk, per_seed[0]["n"]] +
                            [fmt(*seed_stats(ms[k])) for k in METRICS])

    impr_rows = []
    for m, ref_dom, ref_var in ((A48_METHOD, z0_dom, z0_var),
                                (MM_METHOD, {d: {k: float(np.mean(
                                    [seed_dom[(A48_METHOD, s)][d][k] for s in SEEDS]))
                                    for k in METRICS} for d in DOMAIN_NAMES},
                                 {vk: {k: float(np.mean(
                                     [seed_merged[(A48_METHOD, s)][vk][k] for s in SEEDS]))
                                     for k in METRICS} for vk in all_vars})):
        ref_name = "Z0" if m == A48_METHOD else "A48"
        for metric in ("mse", "mae"):
            improved = degraded = equal = 0
            for d in DOMAIN_NAMES:
                v = float(np.mean([seed_dom[(m, s)][d][metric] for s in SEEDS]))
                improved += v < ref_dom[d][metric]
                degraded += v > ref_dom[d][metric]
                equal += v == ref_dom[d][metric]
            impr_rows.append([m, f"vs {ref_name}", "domain(19)", metric,
                              improved, degraded, equal, 19])
            improved = degraded = equal = 0
            for vk in all_vars:
                v = float(np.mean([seed_merged[(m, s)][vk][metric] for s in SEEDS]))
                improved += v < ref_var[vk][metric]
                degraded += v > ref_var[vk][metric]
                equal += v == ref_var[vk][metric]
            impr_rows.append([m, f"vs {ref_name}", "var(190)", metric,
                              improved, degraded, equal, 190])

    comp_rows = []
    mm_ov = {k: float(np.mean([seed_ov[(MM_METHOD, s)][k] for s in SEEDS]))
             for k in ("mse", "mae")}
    a48_ov = {k: float(np.mean([seed_ov[(A48_METHOD, s)][k] for s in SEEDS]))
              for k in ("mse", "mae")}
    for name, va, vb in ((f"{MM_METHOD}-{A48_METHOD}", mm_ov, a48_ov),
                         (f"{A48_METHOD}-Z0", a48_ov, {"mse": z0_ov["mse"], "mae": z0_ov["mae"]}),
                         (f"{MM_METHOD}-Z0", mm_ov, {"mse": z0_ov["mse"], "mae": z0_ov["mae"]})):
        for metric in ("mse", "mae"):
            delta = va[metric] - vb[metric]
            rel = f"{-delta / vb[metric] * 100:+.2f}%" if vb[metric] != 0 else "ref=0"
            comp_rows.append([name, metric, f"{delta:+.6f}", rel])

    pdom, pov = paired_ladder(mm_maps, a48_maps, dom_of_var)
    p_mse = seed_stats([pov[s]["mse"] for s in SEEDS])
    p_mae = seed_stats([pov[s]["mae"] for s in SEEDS])
    paired_rows = []
    for d in DOMAIN_NAMES:
        paired_rows.append([d, "mean+-std over 3 seeds",
                            fmt(*seed_stats([pdom[s][d]["mse"] for s in SEEDS])),
                            fmt(*seed_stats([pdom[s][d]["mae"] for s in SEEDS]))])
    paired_rows.append(["OVERALL(19-domain equal weight)", "mean+-std over 3 seeds",
                        fmt(*p_mse), fmt(*p_mae)])
    for metric, st in (("std_MSE", p_mse), ("std_MAE", p_mae)):
        base = a48_ov["mse" if metric == "std_MSE" else "mae"]
        rel = f"{-st[0] / base * 100:+.2f}%" if base != 0 else "ref=0"
        comp_rows.append([f"{MM_METHOD}-{A48_METHOD}-paired", metric,
                          f"{st[0]:+.6f}", rel])

    def w_csv(name, header, rows):
        with open(RESULTS / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    w_csv("overall.csv", ["method", "seed_stat", "MSE", "MAE"], overall_rows)
    w_csv("domain_summary.csv", ["method", "domain", "seed_stat", "MSE", "MAE"], domain_rows)
    w_csv("variable_level.csv",
          ["method", "var_key", "n_windows", "std_MSE", "std_MAE", "raw_MSE", "raw_MAE"], var_rows)
    w_csv("improvement_counts.csv",
          ["method", "reference", "level", "metric", "improved", "degraded", "equal", "total"],
          impr_rows)
    w_csv("comparisons.csv",
          ["comparison", "metric", "delta", "relative_improvement"], comp_rows)
    w_csv("paired_deltas_vs_a48.csv",
          ["domain", "seed_stat", "paired_dMSE", "paired_dMAE"], paired_rows)

    (RESULTS / "aggregate_summary.json").write_text(json.dumps({
        "methods_overall": {m: {"mse": list(method_overall[m]["mse"]),
                                "mae": list(method_overall[m]["mae"])}
                            for m in ("Z0", A48_METHOD, MM_METHOD)},
        "paired_vs_a48_overall": {"dMSE": list(p_mse), "dMAE": list(p_mae)},
        "ladder": "paired window diff -> variable mean -> domain equal weight -> 19-domain equal weight",
    }, indent=2))
    print(f"[aggregate_mm] wrote 7 tables to {RESULTS} "
          f"(overall / domain / variable / improvement counts / comparisons / "
          f"paired deltas / summary)")


if __name__ == "__main__":
    try:
        ms = tuple(sys.argv[1:]) or (MM_METHOD,)
        main(ms)
    except AggregateError as e:
        print(f"[aggregate_mm] FAILED: {e}")
        sys.exit(1)
