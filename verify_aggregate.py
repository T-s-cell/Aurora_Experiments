#!/usr/bin/env python3
"""Preflight I: verify the ported aggregation against the ORIGINAL EXP-004
aggregate.py at full precision on Z0__test.npz.

Safety: the original module is imported READ-ONLY under a unique module name
with PYTHONDONTWRITEBYTECODE=1; only its pure helpers are called, never its
main(), so nothing is written anywhere in the original tree. (Importing the
original common.py only re-runs os.makedirs(exist_ok=True) on dirs that
already exist.)

Comparison levels (protocol: full float64, tolerance atol<=1e-12; identical
numpy on identical input is expected to be bitwise):
  1. variable level (190 vars, all 4 metrics)
  2. domain level (19 domains, all 4 metrics)
  3. overall
Corroboration (weaker): ported overall vs EXP-004 archived overall.csv
(6-dp display), tolerance <=5e-7.
"""
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np

PROJECT = Path(__file__).resolve().parent
_orig_tree = Path("/home/wlt/MMTS/VisionTS_Experiments/repro_ln_timesx_v1")
if (_orig_tree / "aggregate.py").exists():
    ORIG_DIR = _orig_tree
    ORIG_PROVENANCE = "original VisionTS tree"
    ORIG_TIMESX_DATA = Path("/home/wlt/MMTS/VisionTS_Experiments/repro_zeroshot_ext/timesx_data.py")
else:
    ORIG_DIR = PROJECT / "reference" / "exp004"
    ORIG_PROVENANCE = "vendored read-only copy (reference/README.md)"
    ORIG_TIMESX_DATA = ORIG_DIR / "timesx_data.py"
sys.path.insert(0, str(PROJECT))

ATOL = 1e-12
CSV_TOL = 5e-7


def verify_reference_files():
    """sha256-pin the vendored/original files this check depends on."""
    expect = {
        "aggregate.py": "1cd627f80f38d1e94504e155e6516cd0319bdc4e7f941939c9fe348da9e4124c",
        "common.py": "371c6f462e37fda0b26e1cb19c921b95ed234f56052a890f2d29d1b62b37dad7",
        "timesx_data.py": "a27ec66e9b7e5c11bf95dba8bafc5f7d6b3cfdd6f2683a9774e5da80577a2cfa",
        "configs/v1.yaml": "1a3710c9f837a6d65ffe74a716fa261360150d39764a0a6f0335345e8f26d140",
    }
    out = {}
    for rel, sha in expect.items():
        p = ORIG_TIMESX_DATA if rel == "timesx_data.py" else ORIG_DIR / rel
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        out[rel] = {"ok": got == sha, "sha256": got}
    return out


def load_orig_aggregate():
    spec = importlib.util.spec_from_file_location(
        "orig_exp004_aggregate_readonly", ORIG_DIR / "aggregate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def compare(name, a, b, atol=ATOL):
    diff = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    return {"item": name, "max_abs_diff": diff, "ok": diff <= atol}


def main(z0_path=None):
    from aggregate import (DOMAIN_NAMES, load_set, var_table_of,
                           merge_var_tables, dom_overall)
    z0_path = Path(z0_path) if z0_path else PROJECT / "data" / "Z0__test.npz"

    ref_check = verify_reference_files()
    if not all(v["ok"] for v in ref_check.values()):
        print("[verify_aggregate] reference file sha256 mismatch:", ref_check)
        sys.exit(2)

    orig = load_orig_aggregate()

    entry_p = load_set(z0_path)
    entry_o = orig.load_set(str(z0_path))

    var_p = var_table_of(entry_p)
    var_o = orig.var_table_of(entry_o)
    dom_p, ov_p = dom_overall(var_p)
    dom_o, ov_o = orig.dom_overall(var_o)

    assert set(var_p) == set(var_o) and len(var_p) == 190
    results = []
    for vk in sorted(var_p):
        for k in ("mse", "mae", "raw_mse", "raw_mae"):
            results.append(compare(f"var:{vk}:{k}", var_p[vk][k], var_o[vk][k]))
    for d in DOMAIN_NAMES:
        for k in ("mse", "mae", "raw_mse", "raw_mae"):
            results.append(compare(f"dom:{d}:{k}", dom_p[d][k], dom_o[d][k]))
    for k in ("mse", "mae", "raw_mse", "raw_mae"):
        results.append(compare(f"overall:{k}", ov_p[k], ov_o[k]))

    bitwise = all(r["max_abs_diff"] == 0.0 for r in results)
    all_ok = all(r["ok"] for r in results)

    # corroboration vs archived 6-dp CSV
    arch = {}
    with open(ORIG_DIR / "results" / "overall.csv") as f:
        for row in csv.DictReader(f):
            if row["method"] == "Z0":
                arch["Z0"] = (float(row["MSE"]), float(row["MAE"]))
    corr = {
        "overall_mse_vs_archive": {"ported": ov_p["mse"], "archived": arch["Z0"][0],
                                   "abs_diff": abs(ov_p["mse"] - arch["Z0"][0]),
                                   "ok": abs(ov_p["mse"] - arch["Z0"][0]) <= CSV_TOL},
        "overall_mae_vs_archive": {"ported": ov_p["mae"], "archived": arch["Z0"][1],
                                   "abs_diff": abs(ov_p["mae"] - arch["Z0"][1]),
                                   "ok": abs(ov_p["mae"] - arch["Z0"][1]) <= CSV_TOL},
    }

    report = {
        "z0_source": str(z0_path),
        "original_module_dir": str(ORIG_DIR),
        "original_provenance": ORIG_PROVENANCE,
        "reference_files": ref_check,
        "n_comparisons": len(results),
        "all_within_atol": all_ok,
        "bitwise_equal": bitwise,
        "max_abs_diff": max(r["max_abs_diff"] for r in results),
        "atol": ATOL,
        "corroboration_vs_archive_csv": corr,
        "ported_overall_full_precision": {k: ov_p[k] for k in ("mse", "mae", "raw_mse", "raw_mae")},
        "failed": [r for r in results if not r["ok"]][:20],
    }
    out = PROJECT / "preflight" / "aggregate_verify.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("n_comparisons", "all_within_atol", "bitwise_equal",
                       "max_abs_diff", "ported_overall_full_precision")}, indent=2))
    if not all_ok or not corr["overall_mse_vs_archive"]["ok"] or not corr["overall_mae_vs_archive"]["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
