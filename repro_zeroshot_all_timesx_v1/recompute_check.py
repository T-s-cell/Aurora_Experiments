#!/usr/bin/env python3
"""Standalone recompute (hard-fail). Fresh process; imports NOTHING from the
pipeline (no protocol/common/evaluate) — frozen constants are inlined and the
scoring/aggregation chain is re-implemented here in pure python (numpy is used
ONLY to read frozen_arrays.npz). It independently:

  1. recomputes d from the frozen past96 rows (population std, calib_std
     fallback when std < 1e-8) and checks it against the d embedded in BOTH
     prediction files and the frozen d array;
  2. rescoring: window (12-step point pred vs target) -> variable mean ->
     domain equal-weight (assert 19) -> overall, for VisionTS (single run)
     and Aurora (each seed independently);
  3. compares against the published results/{overall,domain_level,
     variable_level}.csv|json and against the frozen VisionTS reference
     4.2122984787894 / 0.856010666064058;
  4. recomputes improvement counts.

Tolerances (relative, denominator max(1e-12,|ref|)): d 1e-12, variable-level
1e-9, domain/overall 1e-12. Any failure -> exit 1.

The .state/recompute.done marker (incl. input_guards) is written inline and
MUST stay dict-identical to common.stage_input_guards("recompute").
"""
import csv
import gzip
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- inlined frozen constants (keep in sync with protocol.py) ----
SEEDS = (2021, 2022, 2023)
PRED_TOTAL = 24318
TOTAL_WINDOWS = 8106
EXPECTED_DOMAINS = sorted([
    'CropsAndStaples', 'Currency', 'EnergyAndFuels', 'LivestockAndFoodProducts',
    'RawMaterialsAndConstruction', 'SpecialtyAndAdvancedMaterials',
    'StrategicAndHighValueMaterials', 'arts', 'climate', 'economy',
    'electronic_technology', 'finance', 'pets', 'public_health',
    'public_policy', 'science', 'shopping', 'society', 'traffic',
])
REF_VTS_STD_MSE = 4.2122984787894
REF_VTS_STD_MAE = 0.856010666064058
TOL_D = 1e-12
TOL_VAR = 1e-9
TOL_AGG = 1e-12
EPS_STD = 1e-8

REFS = ["visionts_ref/inventory.json", "visionts_ref/frozen_arrays.npz",
        "visionts_ref/vars.json", "visionts_ref/preds.jsonl.gz",
        "visionts_ref/split_manifest.json"]
RESULT_FILES = ["results/aurora_preds.jsonl.gz", "results/variable_level.csv",
                "results/domain_level.csv", "results/overall.json"]

failures = []


def check(name, ok, detail=""):
    if not ok:
        failures.append(f"{name}: {detail}")
    return ok


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel_err(new, ref):
    return abs(new - ref) / max(1e-12, abs(ref))


def read_rows(path):
    with gzip.open(os.path.join(HERE, path), "rt") as f:
        for line in f:
            yield json.loads(line)


def pstd(vals):
    """population std, pure python (matches numpy ddof=0 within float noise)"""
    n = len(vals)
    m = math.fsum(vals) / n
    return math.sqrt(math.fsum((x - m) ** 2 for x in vals) / n)


def pmean(vals):
    return math.fsum(vals) / len(vals)


def pstd1(vals):
    """sample std ddof=1"""
    n = len(vals)
    m = pmean(vals)
    return math.sqrt(math.fsum((x - m) ** 2 for x in vals) / (n - 1))


def score(rows_by_key, d_lookup):
    """rows: iterable of dicts with var_key/domain/pred/target; d_lookup(fi)->d.
    Returns (var_rows{vk:...}, dom_rows{d:...}, overall)."""
    acc = {}
    for r in rows_by_key:
        pred, tgt = r["pred"], r["target"]
        n = len(pred)
        mse = math.fsum((a - b) ** 2 for a, b in zip(pred, tgt)) / n
        mae = math.fsum(abs(a - b) for a, b in zip(pred, tgt)) / n
        d = d_lookup(r["flat_index"])
        acc.setdefault(r["var_key"], []).append(
            (r["domain"], mse / (d * d), mae / d))
    var_rows = {vk: {"domain": rs[0][0], "std_mse": pmean([x[1] for x in rs]),
                     "std_mae": pmean([x[2] for x in rs])}
                for vk, rs in acc.items()}
    by_dom = {}
    for v in var_rows.values():
        by_dom.setdefault(v["domain"], []).append(v)
    if set(by_dom) != set(EXPECTED_DOMAINS):
        raise SystemExit(f"domain set mismatch: {sorted(set(by_dom))}")
    dom_rows = {d: {"std_mse": pmean([v["std_mse"] for v in vs]),
                    "std_mae": pmean([v["std_mae"] for v in vs]),
                    "n_vars": len(vs)}
                for d, vs in by_dom.items()}
    overall = {"std_mse": pmean([v["std_mse"] for v in dom_rows.values()]),
               "std_mae": pmean([v["std_mae"] for v in dom_rows.values()])}
    return var_rows, dom_rows, overall


def main():
    guards = {r: sha256_file(os.path.join(HERE, r)) for r in REFS + RESULT_FILES}
    guards["code_md5:recompute_check.py"] = md5_file(os.path.join(HERE, "recompute_check.py"))

    arrays = np.load(os.path.join(HERE, "visionts_ref/frozen_arrays.npz"))
    past_all, d_frozen, dsrc_frozen = arrays["past"], arrays["d"], arrays["denom_source"]
    flat_all = arrays["flat_index"]
    vars_json = json.load(open(os.path.join(HERE, "visionts_ref/vars.json")))
    calib = {v["var_key"]: float(v["calib_std"]) for v in vars_json}
    inv = json.load(open(os.path.join(HERE, "visionts_ref/inventory.json")))
    fi_of = {w["sample_id"]: w["flat_index"] for w in inv["windows"]}

    # ---- 1. independent d recompute from past96 ----
    d_recomputed = {}
    n_fallback = 0
    for fi in range(TOTAL_WINDOWS):
        row = past_all[fi]
        s = pstd([float(x) for x in row])
        if s < EPS_STD:
            n_fallback += 1
        d_recomputed[fi] = s
    check("d_fallback_count", n_fallback == 4, f"got {n_fallback}")
    max_ed_file, max_ed_frozen = 0.0, 0.0
    for fi in range(TOTAL_WINDOWS):
        if dsrc_frozen[fi] == "calib_std":
            eff = calib[inv["windows"][fi]["var_key"]]
        else:
            eff = d_recomputed[fi]
        max_ed_frozen = max(max_ed_frozen, rel_err(eff, float(d_frozen[fi])))
    check("d_vs_frozen_arrays", max_ed_frozen <= TOL_D, f"max rel {max_ed_frozen:.3e}")

    def d_eff(fi):
        return calib[inv["windows"][fi]["var_key"]] \
            if dsrc_frozen[fi] == "calib_std" else d_recomputed[fi]

    # ---- 2. rescore VisionTS ----
    vts_rows = list(read_rows("visionts_ref/preds.jsonl.gz"))
    check("vts_row_count", len(vts_rows) == TOTAL_WINDOWS, str(len(vts_rows)))
    max_ed = max(rel_err(d_eff(r["flat_index"]), float(r["d"])) for r in vts_rows)
    check("vts_d_vs_recomputed", max_ed <= TOL_D, f"max rel {max_ed:.3e}")
    vv, vd, vo = score(vts_rows, d_eff)
    check("vts_ref_mse", rel_err(vo["std_mse"], REF_VTS_STD_MSE) <= TOL_AGG,
          f"rel {rel_err(vo['std_mse'], REF_VTS_STD_MSE):.3e}")
    check("vts_ref_mae", rel_err(vo["std_mae"], REF_VTS_STD_MAE) <= TOL_AGG,
          f"rel {rel_err(vo['std_mae'], REF_VTS_STD_MAE):.3e}")

    # ---- 3. rescore Aurora per seed ----
    au_rows = list(read_rows("results/aurora_preds.jsonl.gz"))
    check("aurora_row_count", len(au_rows) == PRED_TOTAL, str(len(au_rows)))
    by_seed = {s: [] for s in SEEDS}
    for r in au_rows:
        fi = fi_of[r["sample_id"]]
        if fi != r["flat_index"]:
            check("aurora_flat_index", False, f"seed {r['seed']} {r['sample_id']}")
            continue
        max_ed_file = max(max_ed_file, rel_err(d_eff(fi), float(r["d"])))
        tgt_f = [float(x) for x in arrays["target"][fi]]
        if r["target"] != tgt_f:
            check("aurora_target_bitwise", False,
                  f"seed {r['seed']} {r['sample_id']}")
        by_seed[r["seed"]].append(r)
    check("aurora_d_vs_recomputed", max_ed_file <= TOL_D,
          f"max rel {max_ed_file:.3e}")
    check("aurora_seed_keys", sorted(by_seed) == sorted(SEEDS),
          str(sorted(by_seed)))
    seed_scores = {}
    for s in SEEDS:
        rs = by_seed[s]
        check(f"aurora_seed_{s}_rows", len(rs) == TOTAL_WINDOWS, str(len(rs)))
        seed_scores[s] = score(rs, d_eff)

    # ---- 4. compare with published results ----
    overall = json.load(open(os.path.join(HERE, "results/overall.json")))
    per_seed = {int(k): v for k, v in overall["aurora"]["per_seed"].items()}
    for s in SEEDS:
        check(f"per_seed_{s}_mse", rel_err(seed_scores[s][2]["std_mse"],
              per_seed[s]["std_mse"]) <= TOL_AGG, "overall.json")
        check(f"per_seed_{s}_mae", rel_err(seed_scores[s][2]["std_mae"],
              per_seed[s]["std_mae"]) <= TOL_AGG, "overall.json")
    means = [pmean([seed_scores[s][2]["std_mse"] for s in SEEDS]),
             pmean([seed_scores[s][2]["std_mae"] for s in SEEDS])]
    sds = [pstd1([seed_scores[s][2]["std_mse"] for s in SEEDS]),
           pstd1([seed_scores[s][2]["std_mae"] for s in SEEDS])]
    check("overall_mse_mean", rel_err(means[0], overall["aurora"]["std_mse_mean"])
          <= TOL_AGG, "overall.json")
    check("overall_mae_mean", rel_err(means[1], overall["aurora"]["std_mae_mean"])
          <= TOL_AGG, "overall.json")
    check("overall_mse_std", rel_err(sds[0], overall["aurora"]["std_mse_std"])
          <= TOL_AGG, "overall.json")
    check("overall_mae_std", rel_err(sds[1], overall["aurora"]["std_mae_std"])
          <= TOL_AGG, "overall.json")
    check("published_visionts", rel_err(overall["visionts"]["std_mse"],
          vo["std_mse"]) <= TOL_AGG and rel_err(overall["visionts"]["std_mae"],
          vo["std_mae"]) <= TOL_AGG, "overall.json visionts block")

    dom_csv = {r["domain"]: r for r in csv.DictReader(
        open(os.path.join(HERE, "results/domain_level.csv")))}
    avg_row = dom_csv.pop("平均值（19 域等权）")
    check("domain_csv_rows", set(dom_csv) == set(EXPECTED_DOMAINS),
          str(sorted(set(dom_csv) ^ set(EXPECTED_DOMAINS))))
    max_ed_dom = 0.0
    for d in EXPECTED_DOMAINS:
        dm = [seed_scores[s][1][d]["std_mse"] for s in SEEDS]
        am = [seed_scores[s][1][d]["std_mae"] for s in SEEDS]
        row = dom_csv[d]
        for val, col in ((pmean(dm), "aurora_std_mse_mean"),
                         (pstd1(dm), "aurora_std_mse_std"),
                         (pmean(am), "aurora_std_mae_mean"),
                         (pstd1(am), "aurora_std_mae_std"),
                         (vd[d]["std_mse"], "vts_std_mse"),
                         (vd[d]["std_mae"], "vts_std_mae")):
            max_ed_dom = max(max_ed_dom, rel_err(val, float(row[col])))
    check("domain_level_csv", max_ed_dom <= TOL_AGG, f"max rel {max_ed_dom:.3e}")
    check("domain_avg_row",
          rel_err(means[0], float(avg_row["aurora_std_mse_mean"])) <= TOL_AGG
          and rel_err(float(avg_row["vts_std_mse"]), vo["std_mse"]) <= TOL_AGG,
          "domain_level.csv average row")

    var_csv = {r["var_key"]: r for r in csv.DictReader(
        open(os.path.join(HERE, "results/variable_level.csv")))}
    check("variable_csv_keys", set(var_csv) == set(vv), "variable key set")
    max_ed_var = 0.0
    for vk, row in var_csv.items():
        for val, col in ((vv[vk]["std_mse"], "vts_std_mse"),
                         (vv[vk]["std_mae"], "vts_std_mae"),
                         (pmean([seed_scores[s][0][vk]["std_mse"] for s in SEEDS]),
                          "aurora_std_mse_mean"),
                         (pmean([seed_scores[s][0][vk]["std_mae"] for s in SEEDS]),
                          "aurora_std_mae_mean")):
            max_ed_var = max(max_ed_var, rel_err(val, float(row[col])))
    check("variable_level_csv", max_ed_var <= TOL_VAR, f"max rel {max_ed_var:.3e}")

    imp = overall["improvement_counts"]
    my_imp = {
        "domains": {"mse": sum(1 for d in EXPECTED_DOMAINS
                               if pmean([seed_scores[s][1][d]["std_mse"] for s in SEEDS])
                               < vd[d]["std_mse"]),
                    "mae": sum(1 for d in EXPECTED_DOMAINS
                               if pmean([seed_scores[s][1][d]["std_mae"] for s in SEEDS])
                               < vd[d]["std_mae"])},
        "variables": {"mse": sum(1 for vk in vv
                                 if pmean([seed_scores[s][0][vk]["std_mse"] for s in SEEDS])
                                 < vv[vk]["std_mse"]),
                      "mae": sum(1 for vk in vv
                                 if pmean([seed_scores[s][0][vk]["std_mae"] for s in SEEDS])
                                 < vv[vk]["std_mae"])},
    }
    check("improvement_counts", json.dumps(imp, sort_keys=True) ==
          json.dumps(my_imp, sort_keys=True), f"{imp} != {my_imp}")

    report = {
        "stage": "recompute_check",
        "passed": not failures,
        "tolerances": {"d": TOL_D, "variable": TOL_VAR, "aggregate": TOL_AGG},
        "checks": {
            "d_fallback_windows": n_fallback,
            "max_rel_d_vs_frozen": max_ed_frozen,
            "max_rel_d_vs_pred_files": max_ed_file,
            "vts_overall_recomputed": vo,
            "aurora_overall_recomputed": {
                str(s): seed_scores[s][2] for s in SEEDS},
            "max_rel_domain_csv": max_ed_dom,
            "max_rel_variable_csv": max_ed_var,
            "improvement_counts_recomputed": my_imp,
        },
        "failures": failures,
        "input_guards": guards,
    }
    out = os.path.join(HERE, "results/recompute_check.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=1)
    os.replace(tmp, out)

    if failures:
        for x in failures:
            print(f"[recompute] FAIL {x}", file=sys.stderr)
        print(f"[recompute] FAILED ({len(failures)} checks) -> {out}")
        sys.exit(1)
    print(f"[recompute] PASS (d rel {max_ed_file:.2e}, domain csv rel "
          f"{max_ed_dom:.2e}, variable csv rel {max_ed_var:.2e}) -> {out}")

    # ---- marker (inline; identical to common.mark_done + stage_input_guards("recompute")) ----
    marker = {
        "sha": {"results/recompute_check.json": sha256_file(out)},
        "input_guards": guards,
        "status": "done",
    }
    state_dir = os.path.join(HERE, ".state")
    os.makedirs(state_dir, exist_ok=True)
    mp = os.path.join(state_dir, "recompute.done")
    mtmp = mp + ".tmp"
    with open(mtmp, "w") as f:
        json.dump(marker, f, indent=1)
    os.replace(mtmp, mp)


if __name__ == "__main__":
    main()
