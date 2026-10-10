#!/usr/bin/env python3
"""S5b evaluate. Scoring is reimplemented here (old metrics modules are NOT
imported); correctness is pinned by the VisionTS gate: the SAME implementation
must reproduce the frozen EXP-014 full-window reference 4.2122984787894 /
0.856010666064058 within rel 1e-12 before any Aurora number is produced.

Chain: window (12-step point pred vs target, denom = frozen d incl. fallback)
-> variable mean -> domain equal-weight (assert exactly 19) -> overall.
Aurora: three seeds scored independently, reported as mean +/- std (ddof=1);
predictions are never averaged and no best seed is selected.
Also: 19-domain comparison table vs VisionTS, improvement counts, and the
2,474 LN-test subset report (pure-cache reproduction gate runs ONLY when the
subset is fully served by verified old cache; otherwise recorded as not
executable, per protocol amendment)."""
import argparse
import gzip
import json
import os
from collections import defaultdict

import numpy as np

import common as C
import protocol as P


def window_metrics(pred, target, denom):
    n = len(pred)
    raw_mse = float(np.sum((pred - target) ** 2) / n)
    raw_mae = float(np.sum(np.abs(pred - target)) / n)
    return raw_mse, raw_mae, raw_mse / (denom * denom), raw_mae / denom


def rel_err(new, ref):
    return abs(new - ref) / max(1e-12, abs(ref))


def with_metrics(rows):
    out = []
    for r in rows:
        wm = window_metrics(np.asarray(r["pred"]), np.asarray(r["target"]),
                            float(r["d"]))
        out.append({"var_key": r["var_key"], "domain": r["domain"],
                    "raw_mse": wm[0], "raw_mae": wm[1],
                    "std_mse": wm[2], "std_mae": wm[3]})
    return out


def aggregate(wm_rows):
    """window-metric rows -> (var_rows, dom_rows, overall); asserts 19 domains."""
    by_var = defaultdict(list)
    for r in wm_rows:
        by_var[r["var_key"]].append(r)
    var_rows = {}
    for vk, rs in by_var.items():
        var_rows[vk] = {
            "var_key": vk, "domain": rs[0]["domain"], "n": len(rs),
            "raw_mse": float(np.mean([x["raw_mse"] for x in rs])),
            "raw_mae": float(np.mean([x["raw_mae"] for x in rs])),
            "std_mse": float(np.mean([x["std_mse"] for x in rs])),
            "std_mae": float(np.mean([x["std_mae"] for x in rs])),
        }
    by_dom = defaultdict(list)
    for v in var_rows.values():
        by_dom[v["domain"]].append(v)
    if set(by_dom) != set(P.EXPECTED_DOMAINS):
        raise SystemExit(f"domain set mismatch: "
                         f"{sorted(set(by_dom) ^ set(P.EXPECTED_DOMAINS))}")
    dom_rows = {}
    for d, vs in by_dom.items():
        dom_rows[d] = {
            "domain": d, "n_vars": len(vs),
            "std_mse": float(np.mean([v["std_mse"] for v in vs])),
            "std_mae": float(np.mean([v["std_mae"] for v in vs])),
        }
    overall = {
        "n_domains": len(dom_rows),
        "std_mse": float(np.mean([v["std_mse"] for v in dom_rows.values()])),
        "std_mae": float(np.mean([v["std_mae"] for v in dom_rows.values()])),
    }
    return var_rows, dom_rows, overall


def read_preds_gz(path):
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


def mean_std(vals):
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs(C.RESULTS_DIR, exist_ok=True)

    # ---- VisionTS gate (scoring self-validation, must pass first) ----
    vts_rows = with_metrics(list(read_preds_gz(C.VTS_PREDS_PATH)))
    vts_vars, vts_dom, vts_overall = aggregate(vts_rows)
    e_mse = rel_err(vts_overall["std_mse"], P.REF_VTS_STD_MSE)
    e_mae = rel_err(vts_overall["std_mae"], P.REF_VTS_STD_MAE)
    gate = {
        "reference": {"std_mse": P.REF_VTS_STD_MSE, "std_mae": P.REF_VTS_STD_MAE},
        "recomputed": vts_overall,
        "rel_err": {"std_mse": e_mse, "std_mae": e_mae},
        "tol": P.REF_VTS_TOL,
        "passed": bool(e_mse <= P.REF_VTS_TOL and e_mae <= P.REF_VTS_TOL),
    }
    C.dump_json(gate, os.path.join(C.RESULTS_DIR, "vts_gate.json"))
    if not gate["passed"]:
        raise SystemExit(f"VisionTS gate FAILED: rel_err {e_mse:.3e}/{e_mae:.3e} "
                         f"> {P.REF_VTS_TOL:g} — scoring implementation is wrong")
    print(f"[evaluate] VisionTS gate PASS (rel_err {e_mse:.2e}/{e_mae:.2e})")

    # ---- Aurora per seed ----
    aur_rows = list(read_preds_gz(os.path.join(C.RESULTS_DIR, "aurora_preds.jsonl.gz")))
    if len(aur_rows) != P.PRED_TOTAL:
        raise SystemExit(f"aurora preds rows {len(aur_rows)} != {P.PRED_TOTAL}")
    by_seed = defaultdict(list)
    for r in aur_rows:
        by_seed[r["seed"]].append(r)
    if sorted(by_seed) != sorted(P.SEEDS):
        raise SystemExit(f"seed set {sorted(by_seed)} != {sorted(P.SEEDS)}")
    seed_agg = {}
    for s in P.SEEDS:
        rs = by_seed[s]
        if len(rs) != P.TOTAL_WINDOWS:
            raise SystemExit(f"seed {s}: {len(rs)} rows != {P.TOTAL_WINDOWS}")
        seed_agg[s] = aggregate(with_metrics(rs))

    per_seed_overall = {str(s): seed_agg[s][2] for s in P.SEEDS}
    dom_vals = {d: {"mse": [], "mae": []} for d in P.EXPECTED_DOMAINS}
    for s in P.SEEDS:
        for d, dr in seed_agg[s][1].items():
            dom_vals[d]["mse"].append(dr["std_mse"])
            dom_vals[d]["mae"].append(dr["std_mae"])
    mse_m, mse_sd = mean_std([per_seed_overall[str(s)]["std_mse"] for s in P.SEEDS])
    mae_m, mae_sd = mean_std([per_seed_overall[str(s)]["std_mae"] for s in P.SEEDS])

    # ---- 19-domain comparison table ----
    dom_csv = ["domain,n_vars,vts_std_mse,aurora_std_mse_mean,aurora_std_mse_std,"
               "delta_std_mse,vts_std_mae,aurora_std_mae_mean,aurora_std_mae_std,"
               "delta_std_mae,improve_mse,improve_mae"]
    for d in P.EXPECTED_DOMAINS:
        m1, s1 = mean_std(dom_vals[d]["mse"])
        m2, s2 = mean_std(dom_vals[d]["mae"])
        vt = vts_dom[d]
        dm1, dm2 = m1 - vt["std_mse"], m2 - vt["std_mae"]
        dom_csv.append(f"{d},{vt['n_vars']},{vt['std_mse']:.15e},{m1:.15e},"
                       f"{s1:.15e},{dm1:.15e},{vt['std_mae']:.15e},{m2:.15e},"
                       f"{s2:.15e},{dm2:.15e},{int(dm1 < 0)},{int(dm2 < 0)}")
    dom_csv.append(
        f"平均值（19 域等权）,{P.N_DOMAINS},{vts_overall['std_mse']:.15e},"
        f"{mse_m:.15e},{mse_sd:.15e},{mse_m - vts_overall['std_mse']:.15e},"
        f"{vts_overall['std_mae']:.15e},{mae_m:.15e},{mae_sd:.15e},"
        f"{mae_m - vts_overall['std_mae']:.15e},"
        f"{int(mse_m < vts_overall['std_mse'])},{int(mae_m < vts_overall['std_mae'])}")
    with open(os.path.join(C.RESULTS_DIR, "domain_level.csv"), "w") as f:
        f.write("\n".join(dom_csv) + "\n")

    # ---- variable table (raw only at variable level; raw averaged over seeds) ----
    if set(seed_agg[P.SEEDS[0]][0]) != set(vts_vars):
        raise SystemExit("variable key set mismatch aurora vs vts")
    var_csv = ["var_key,domain,n,vts_std_mse,vts_std_mae,aurora_std_mse_mean,"
               "aurora_std_mae_mean,delta_std_mse,delta_std_mae,"
               "aurora_raw_mse_mean,aurora_raw_mae_mean"]
    improve_vars = {"mse": 0, "mae": 0}
    for vk in sorted(vts_vars):
        ms = [seed_agg[s][0][vk]["std_mse"] for s in P.SEEDS]
        ma = [seed_agg[s][0][vk]["std_mae"] for s in P.SEEDS]
        r1, _ = mean_std(ms)
        r2, _ = mean_std(ma)
        vt = vts_vars[vk]
        a0 = seed_agg[P.SEEDS[0]][0][vk]
        dm1, dm2 = r1 - vt["std_mse"], r2 - vt["std_mae"]
        improve_vars["mse"] += int(dm1 < 0)
        improve_vars["mae"] += int(dm2 < 0)
        var_csv.append(f"{vk},{a0['domain']},{a0['n']},{vt['std_mse']:.15e},"
                       f"{vt['std_mae']:.15e},{r1:.15e},{r2:.15e},{dm1:.15e},"
                       f"{dm2:.15e},"
                       f"{float(np.mean([seed_agg[s][0][vk]['raw_mse'] for s in P.SEEDS])):.15e},"
                       f"{float(np.mean([seed_agg[s][0][vk]['raw_mae'] for s in P.SEEDS])):.15e}")
    with open(os.path.join(C.RESULTS_DIR, "variable_level.csv"), "w") as f:
        f.write("\n".join(var_csv) + "\n")

    improve_domains = {
        "mse": sum(1 for d in P.EXPECTED_DOMAINS
                   if mean_std(dom_vals[d]["mse"])[0] < vts_dom[d]["std_mse"]),
        "mae": sum(1 for d in P.EXPECTED_DOMAINS
                   if mean_std(dom_vals[d]["mae"])[0] < vts_dom[d]["std_mae"]),
    }
    overall = {
        "protocol_version": P.PROTOCOL_VERSION,
        "aurora": {
            "per_seed": per_seed_overall,
            "std_mse_mean": mse_m, "std_mse_std": mse_sd,
            "std_mae_mean": mae_m, "std_mae_std": mae_sd,
        },
        "visionts": vts_overall,
        "delta": {"std_mse": mse_m - vts_overall["std_mse"],
                  "std_mae": mae_m - vts_overall["std_mae"]},
        "improvement_counts": {"domains": improve_domains, "variables": improve_vars},
        "note": "delta = Aurora(3-seed mean) - VisionTS; negative = Aurora better",
    }
    C.dump_json(overall, os.path.join(C.RESULTS_DIR, "overall.json"))
    print(f"[evaluate] aurora overall std-MSE {mse_m:.6f}±{mse_sd:.6f} "
          f"std-MAE {mae_m:.6f}±{mae_sd:.6f} | vts "
          f"{vts_overall['std_mse']:.6f}/{vts_overall['std_mae']:.6f} | "
          f"improve domains {improve_domains} vars {improve_vars}")

    # ---- 2,474 LN-test subset ----
    ln = C.load_json(P.LN_MANIFEST)
    ln_keys = set()
    for vk, v in ln["variables"].items():
        for t in v["native"]["test"]:
            ln_keys.add(t if isinstance(t, str) else t["sample_id"])
    sub_by_seed, pure, reasons = {}, True, []
    for s in P.SEEDS:
        rows = [r for r in by_seed[s] if r["sample_id"] in ln_keys]
        if len(rows) != P.N_LN_TEST:
            raise SystemExit(f"subset seed {s}: {len(rows)} rows != {P.N_LN_TEST}")
        srcs = {r["source"] for r in rows}
        if srcs != {"cache"}:
            pure = False
            reasons.append(f"seed {s}: sources={sorted(srcs)}")
        sub_by_seed[s] = aggregate(with_metrics(rows))[2]
    sm, ss = mean_std([sub_by_seed[s]["std_mse"] for s in P.SEEDS])
    am, asd = mean_std([sub_by_seed[s]["std_mae"] for s in P.SEEDS])
    subset = {
        "n_windows_manifest": P.N_LN_TEST,
        "per_seed": {str(s): sub_by_seed[s] for s in P.SEEDS},
        "std_mse_mean": sm, "std_mse_std": ss,
        "std_mae_mean": am, "std_mae_std": asd,
        "pure_cache": pure,
    }
    if pure:
        e1 = abs(sm - P.REF_A48_2474_STD_MSE)
        e2 = abs(ss - P.REF_A48_2474_STD_MSE_STD)
        e3 = abs(am - P.REF_A48_2474_STD_MAE)
        e4 = abs(asd - P.REF_A48_2474_STD_MAE_STD)
        subset["reference"] = {
            "std_mse": P.REF_A48_2474_STD_MSE, "std_mse_std": P.REF_A48_2474_STD_MSE_STD,
            "std_mae": P.REF_A48_2474_STD_MAE, "std_mae_std": P.REF_A48_2474_STD_MAE_STD,
            "tol": P.REF_A48_TOL}
        subset["abs_err"] = {"std_mse": e1, "std_mse_std": e2,
                             "std_mae": e3, "std_mae_std": e4}
        subset["reproduced"] = bool(max(e1, e2, e3, e4) <= P.REF_A48_TOL)
        print(f"[evaluate] subset 2,474 pure-cache: {sm:.6f}±{ss:.6f} / "
              f"{am:.6f}±{asd:.6f} reproduced={subset['reproduced']}")
    else:
        subset["reproduction"] = ("NOT_EXECUTABLE — subset contains re-inferred "
                                  "windows: " + "; ".join(reasons))
        print(f"[evaluate] subset 2,474 NOT pure cache ({reasons}); actual "
              f"{sm:.6f}±{ss:.6f} / {am:.6f}±{asd:.6f}")
    C.dump_json(subset, os.path.join(C.RESULTS_DIR, "subset_2474.json"))

    C.mark_done("s5b_evaluate", {
        "aurora_std_mse_mean": mse_m, "aurora_std_mae_mean": mae_m,
        "vts_gate": "PASS", "subset_reproduced": subset.get("reproduced"),
        "sha": {"results/overall.json": C.sha256_file(os.path.join(C.RESULTS_DIR, "overall.json")),
                "results/domain_level.csv": C.sha256_file(os.path.join(C.RESULTS_DIR, "domain_level.csv")),
                "results/variable_level.csv": C.sha256_file(os.path.join(C.RESULTS_DIR, "variable_level.csv")),
                "results/subset_2474.json": C.sha256_file(os.path.join(C.RESULTS_DIR, "subset_2474.json")),
                "results/vts_gate.json": C.sha256_file(os.path.join(C.RESULTS_DIR, "vts_gate.json"))},
        "input_guards": C.stage_input_guards("evaluate")})


if __name__ == "__main__":
    main()
