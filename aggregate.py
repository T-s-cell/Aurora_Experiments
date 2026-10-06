#!/usr/bin/env python3
"""Aggregation for the Aurora x TimesX zero-shot baseline.

Ported from VisionTS_Experiments/repro_ln_timesx_v1/aggregate.py @ commit
dcd2425. The pure helpers (load_set / var_table_of / merge_var_tables /
dom_overall / seed_stats) are numerically identical to the original so that
verify_aggregate.py can compare them at full precision.

Differences from the original (scope only, no numerics):
- No N0, no selection.json, no fine-tuning fallback logic. Sources are
  Aurora method shards {method}__{domain}__s{seed}__test.npz plus the frozen
  Z0 reference (data/Z0__test.npz).
- Aggregation ladder unchanged: window -> variable -> 19 domains merged per
  (method, seed) -> domain equal-weight -> overall -> 3-seed mean±std(ddof=1).
- Best seed is never selected.
"""
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import DOMAIN_NAMES

PROJECT = Path(__file__).resolve().parent
PREDICTIONS = PROJECT / "predictions"
RESULTS = PROJECT / "results"
Z0_REF = PROJECT / "data" / "Z0__test.npz"

METRICS = ("mse", "mae", "raw_mse", "raw_mae")


class AggregateError(RuntimeError):
    pass


def load_set(path):
    with np.load(path, allow_pickle=False) as z:
        pred, target, d = z["pred"], z["target"], z["d"]
        vks = [str(x) for x in z["var_keys"]]
    err = pred - target
    return {"var": vks,
            "std_mse": np.mean((err / d[:, None]) ** 2, axis=1),
            "std_mae": np.mean(np.abs(err) / d[:, None], axis=1),
            "raw_mse": np.mean(err ** 2, axis=1),
            "raw_mae": np.mean(np.abs(err), axis=1)}


def var_table_of(entry):
    per_var = {}
    for vk, sm, sa, rm, ra in zip(entry["var"], entry["std_mse"],
                                  entry["std_mae"], entry["raw_mse"],
                                  entry["raw_mae"]):
        pv = per_var.setdefault(vk, {k: [] for k in METRICS})
        pv["mse"].append(sm)
        pv["mae"].append(sa)
        pv["raw_mse"].append(rm)
        pv["raw_mae"].append(ra)
    return {vk: {k: float(np.mean(v[k])) for k in METRICS} | {"n": len(v["mse"])}
            for vk, v in per_var.items()}


def merge_var_tables(tables, expect_domains=len(DOMAIN_NAMES),
                     expect_vars=190, strict=True):
    merged = {}
    for t in tables:
        for vk, v in t.items():
            if vk in merged:
                raise AggregateError(f"duplicate var {vk} across domain runs")
            merged[vk] = v
    if strict:
        doms = {vk.split("__", 1)[0] for vk in merged}
        missing = sorted(set(DOMAIN_NAMES) - doms)
        extra = sorted(doms - set(DOMAIN_NAMES))
        if missing or extra:
            raise AggregateError(f"domain coverage broken: missing={missing[:3]} "
                                 f"extra={extra[:3]}")
        if len(merged) != expect_vars:
            raise AggregateError(f"{len(merged)} vars merged, expected "
                                 f"{expect_vars}")
    return merged


def dom_overall(merged, strict=True):
    dom_t = {}
    for d in DOMAIN_NAMES:
        vs = [merged[vk] for vk in merged if vk.split("__", 1)[0] == d]
        if not vs:
            if strict:
                raise AggregateError(f"no variables for domain {d}")
            continue
        dom_t[d] = {k: float(np.mean([v[k] for v in vs])) for k in METRICS}
    _assert_finite(dom_t, "domain table")
    overall = {k: float(np.mean([dom_t[d][k] for d in dom_t])) for k in METRICS}
    _assert_finite(overall, "overall")
    return dom_t, overall


def _assert_finite(obj, what):
    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}")
        elif isinstance(o, float) and not math.isfinite(o):
            raise AggregateError(f"non-finite {what} at {path}")
    walk(obj, what)


def seed_stats(values):
    if len(values) == 1:
        return values[0], None
    return float(np.mean(values)), float(np.std(values, ddof=1))


def fmt(mean, std):
    return f"{mean:.6f}" if std is None else f"{mean:.6f}+/-{std:.6f}"


def collect_method_shards(method, seeds=(2021, 2022, 2023)):
    """{(seed): [shard paths, one per domain]} with coverage hard check."""
    per_seed = {}
    for seed in seeds:
        paths = sorted(PREDICTIONS.glob(f"{method}__*__s{seed}__test.npz"))
        if len(paths) != len(DOMAIN_NAMES):
            raise AggregateError(f"{method} s{seed}: {len(paths)} shards, "
                                 f"expected {len(DOMAIN_NAMES)}")
        per_seed[seed] = paths
    return per_seed


def main(methods=("A48",)):
    RESULTS.mkdir(exist_ok=True)
    sets = {"Z0": load_set(Z0_REF)}
    for m in methods:
        for seed, paths in collect_method_shards(m).items():
            for p in paths:
                sets[p.stem.replace("__test", "")] = load_set(p)

    z0_var = var_table_of(sets["Z0"])
    z0_dom, z0_ov = dom_overall(z0_var)

    seed_merged, seed_dom, seed_ov = {}, {}, {}
    for m in methods:
        for seed in (2021, 2022, 2023):
            rids = sorted(k for k in sets if k.startswith(f"{m}__") and k.endswith(f"s{seed}"))
            merged = merge_var_tables([var_table_of(sets[rid]) for rid in rids])
            seed_merged[(m, seed)] = merged
            seed_dom[(m, seed)], seed_ov[(m, seed)] = dom_overall(merged)

    overall_rows = [["Z0", "single", fmt(z0_ov["mse"], None), fmt(z0_ov["mae"], None)]]
    method_overall = {"Z0": {"mse": (z0_ov["mse"], None), "mae": (z0_ov["mae"], None)}}
    for m in methods:
        mo = {}
        for k in ("mse", "mae"):
            vals = [seed_ov[(m, s)][k] for s in (2021, 2022, 2023)]
            mo[k] = seed_stats(vals)
        method_overall[m] = mo
        overall_rows.append([m, "mean+-std over 3 seeds", fmt(*mo["mse"]), fmt(*mo["mae"])])

    domain_rows = [["Z0", d, "single", fmt(z0_dom[d]["mse"], None),
                    fmt(z0_dom[d]["mae"], None)] for d in DOMAIN_NAMES]
    for m in methods:
        for d in DOMAIN_NAMES:
            mv = [seed_dom[(m, s)][d]["mse"] for s in (2021, 2022, 2023)]
            av = [seed_dom[(m, s)][d]["mae"] for s in (2021, 2022, 2023)]
            domain_rows.append([m, d, "mean+-std", fmt(*seed_stats(mv)), fmt(*seed_stats(av))])

    all_vars = sorted(z0_var)
    if len(all_vars) != 190:
        raise AggregateError(f"Z0 covers {len(all_vars)} vars, expected 190")
    var_rows = [["Z0", vk, z0_var[vk]["n"], fmt(z0_var[vk]["mse"], None),
                 fmt(z0_var[vk]["mae"], None), fmt(z0_var[vk]["raw_mse"], None),
                 fmt(z0_var[vk]["raw_mae"], None)] for vk in all_vars]
    for m in methods:
        for vk in all_vars:
            per_seed = [seed_merged[(m, s)][vk] for s in (2021, 2022, 2023)]
            ms = {k: [t[k] for t in per_seed] for k in METRICS}
            var_rows.append([m, vk, per_seed[0]["n"]] +
                            [fmt(*seed_stats(ms[k])) for k in METRICS])

    impr_rows = []
    for m in methods:
        for metric in ("mse", "mae"):
            improved = degraded = equal = 0
            for d in DOMAIN_NAMES:
                v = float(np.mean([seed_dom[(m, s)][d][metric] for s in (2021, 2022, 2023)]))
                ref = z0_dom[d][metric]
                improved += v < ref
                degraded += v > ref
                equal += v == ref
            impr_rows.append([m, "domain(19)", metric, improved, degraded, equal, 19])
            improved = degraded = equal = 0
            for vk in all_vars:
                v = float(np.mean([seed_merged[(m, s)][vk][metric] for s in (2021, 2022, 2023)]))
                ref = z0_var[vk][metric]
                improved += v < ref
                degraded += v > ref
                equal += v == ref
            impr_rows.append([m, "var(190)", metric, improved, degraded, equal, 190])

    comp_rows = []
    for m in methods:
        for metric in ("mse", "mae"):
            va = float(np.mean([seed_ov[(m, s)][metric] for s in (2021, 2022, 2023)]))
            vb = z0_ov[metric]
            delta = va - vb
            rel = f"{-delta / vb * 100:+.2f}%" if vb != 0 else "ref=0, ratio undefined"
            comp_rows.append([f"{m}-Z0", metric, f"{delta:+.6f}", rel])

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
          ["method", "level", "metric", "improved", "degraded", "equal", "total"], impr_rows)
    w_csv("comparisons_vs_z0.csv",
          ["comparison", "metric", "delta", "relative_improvement"], comp_rows)

    (RESULTS / "aggregate_summary.json").write_text(json.dumps(
        {"methods_overall": {m: {"mse": list(v["mse"]), "mae": list(v["mae"])}
                             for m, v in method_overall.items()}}, indent=2))
    print(f"[aggregate] wrote tables for {methods} (+Z0 reference)")


if __name__ == "__main__":
    try:
        ms = tuple(sys.argv[1:]) or ("A48",)
        main(ms)
    except AggregateError as e:
        print(f"[aggregate] FAILED: {e}")
        sys.exit(1)
