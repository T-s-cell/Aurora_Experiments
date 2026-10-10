#!/usr/bin/env python3
"""Acceptance R1-R7 (hard gate; any failure -> exit 1, no threshold is ever
loosened, no window is ever dropped):

  R1 per-seed key set == frozen inventory (8,106), no duplicates, flat_index
     aligned with inventory, target bitwise == frozen target row, d == frozen d
     (exact float round-trip), denom_source == frozen denom_source;
  R2 all pred/target finite, pred shape (12,), d positive and finite;
  R3 constant-history windows: denom_source == calib_std on EXACTLY the 4
     frozen fallback windows x 3 seeds and nowhere else (no silent drops);
  R4 provenance: every row's (pred_file, pred_file_sha256) matches the reuse
     plan/manifest record for its source class, the file re-hashes to the
     recorded sha256, and the npz row is bitwise the emitted pred;
  R5 independent recompute passed (results/recompute_check.json, pure
     re-implementation, three aggregation levels);
  R6 VisionTS recompute gate passed with rel_err <= 1e-12
     (results/vts_gate.json);
  R7 fill set == exact set difference (frozen keys - verified cache keys) per
     seed, cache/new disjoint, 24,318 rows total, counts == assemble_report.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

import common as C
import protocol as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs(C.RESULTS_DIR, exist_ok=True)

    failures = []
    rule_results = {}

    def rule(rid, ok, detail=""):
        rule_results[rid] = bool(ok)
        if not ok:
            failures.append(f"{rid}: {detail}")
        return bool(ok)

    inv = C.load_json(C.INV_PATH)
    win_by_sid = {w["sample_id"]: w for w in inv["windows"]}
    arrays = np.load(C.ARRAYS_PATH)
    target_all, d_all, dsrc_all = arrays["target"], arrays["d"], arrays["denom_source"]

    rows = []
    with __import__("gzip").open(os.path.join(C.RESULTS_DIR,
                                              "aurora_preds.jsonl.gz"), "rt") as f:
        for line in f:
            rows.append(json.loads(line))

    final = C.load_json(C.REUSE_FINAL)
    plan = C.load_json(C.REUSE_PLAN)
    manifest = C.load_json(C.MANIFEST_PATH)

    # ---- expected provenance records ----
    cache_prov = {}   # (seed, var_key) -> (file, sha)
    for key, v in final["variables"].items():
        if v["verdict"] != "reuse":
            continue
        s_str, vk = key.split("|", 1)
        rec = plan["per_seed"][s_str]["vars"][vk]
        cache_prov[(int(s_str), vk)] = (rec["file"], rec["sha256"])
    new_prov = {}     # (seed, var_key) -> (file, sha)
    for key, ent in manifest["entries"].items():
        new_prov[(ent["seed"], ent["var_key"])] = (ent["file"], ent["sha256"])

    # ---- R4 file re-hash + npz row extraction ----
    npz_rows = defaultdict(dict)   # (abs_path) -> {sid: pred_row}
    file_cache = {}

    def load_npz(abs_path):
        if abs_path not in file_cache:
            with np.load(abs_path, allow_pickle=False) as z:
                vks = ([str(x) for x in z["var_keys"]]
                       if "var_keys" in z.files else None)
                file_cache[abs_path] = (
                    [str(x) for x in z["sample_ids"]], vks,
                    np.asarray(z["pred"], dtype=np.float64))
        return file_cache[abs_path]

    covered_by_seed = {s: set() for s in P.SEEDS}
    r4_bad = []
    for s in P.SEEDS:
        for vk, rec in plan["per_seed"][str(s)]["vars"].items():
            if rec.get("verdict") != "reuse_candidate":
                continue
            if final["variables"].get(f"{s}|{vk}", {}).get("verdict") != "reuse":
                continue
            path = os.path.join(P.OLD_PRED_DIR, rec["file"])
            if C.sha256_file(path) != rec["sha256"]:
                r4_bad.append(f"old shard {rec['file']} re-hash mismatch")
                continue
            sids, vks, preds = load_npz(path)
            for i, sid in enumerate(sids):
                if vks[i] == vk:
                    covered_by_seed[s].add((s, sid))
                    npz_rows[path][(s, sid)] = preds[i]

    for key, ent in manifest["entries"].items():
        path = os.path.join(C.CACHE_DIR, ent["file"])
        if C.sha256_file(path) != ent["sha256"]:
            r4_bad.append(f"cache_all {ent['file']} re-hash mismatch")
            continue
        sids, _, preds = load_npz(path)
        for i, sid in enumerate(sids):
            npz_rows[path][(ent["seed"], sid)] = preds[i]
    rule("R4_files", not r4_bad, "; ".join(r4_bad[:5]))

    # ---- per-row checks (R1 R2 R3 R4) ----
    all_keys = {(s, w["sample_id"]) for s in P.SEEDS for w in inv["windows"]}
    seen_keys, cache_keys, new_keys = set(), set(), set()
    const_frozen_sids = {win_by_sid_inv["sample_id"]
                         for win_by_sid_inv in inv["windows"]
                         if dsrc_all[win_by_sid_inv["flat_index"]] == "calib_std"}
    const_rows = defaultdict(list)
    r1234_bad = []

    for r in rows:
        s, sid, vk = r["seed"], r["sample_id"], r["var_key"]
        key = (s, sid)
        if key in seen_keys:
            r1234_bad.append(f"duplicate row seed={s} {sid}")
            continue
        seen_keys.add(key)
        w = win_by_sid.get(sid)
        if w is None or w["var_key"] != vk:
            r1234_bad.append(f"identity mismatch {sid}")
            continue
        fi = r["flat_index"]
        if fi != w["flat_index"]:
            r1234_bad.append(f"flat_index misaligned {sid}")
        pred = np.asarray(r["pred"], dtype=np.float64)
        tgt = np.asarray(r["target"], dtype=np.float64)
        if pred.shape != (P.PRED_LEN,) or tgt.shape != (P.PRED_LEN,):
            r1234_bad.append(f"shape seed={s} {sid}")
        if not (np.isfinite(pred).all() and np.isfinite(tgt).all()):
            r1234_bad.append(f"non-finite seed={s} {sid}")
        if not np.isfinite(r["d"]) or r["d"] <= 0:
            r1234_bad.append(f"bad d seed={s} {sid}")
        if not np.array_equal(tgt, target_all[fi]):
            r1234_bad.append(f"target != frozen seed={s} {sid}")
        if float(r["d"]) != float(d_all[fi]):
            r1234_bad.append(f"d != frozen seed={s} {sid}")
        if r["denom_source"] != str(dsrc_all[fi]):
            r1234_bad.append(f"denom_source != frozen seed={s} {sid}")
        if r["denom_source"] == "calib_std":
            const_rows[s].append(sid)
        prov = new_prov if r["source"] == "new" else cache_prov
        want = prov.get((s, vk))
        if want is None or (r["pred_file"], r["pred_file_sha256"]) != want:
            r1234_bad.append(f"provenance seed={s} {sid} {r['source']}")
        else:
            path = (os.path.join(C.CACHE_DIR, r["pred_file"])
                    if r["source"] == "new"
                    else os.path.join(P.OLD_PRED_DIR, r["pred_file"]))
            row_pred = npz_rows.get(path, {}).get(key)
            if row_pred is None or not np.array_equal(pred, row_pred):
                r1234_bad.append(f"pred != npz row seed={s} {sid}")
        (new_keys if r["source"] == "new" else cache_keys).add(key)

    rule("R1_keyset", seen_keys == all_keys,
         f"missing={len(all_keys - seen_keys)} extra={len(seen_keys - all_keys)}")
    rule("R1234_rows", not r1234_bad, "; ".join(r1234_bad[:8])
         + (f" (+{len(r1234_bad)-8} more)" if len(r1234_bad) > 8 else ""))
    ok_const = (set(const_rows) == set(P.SEEDS)
                and all(set(const_rows[s]) == const_frozen_sids for s in P.SEEDS))
    const_counts = {str(s): len(const_rows.get(s, [])) for s in P.SEEDS}
    rule("R3_constant_windows",
         ok_const and len(const_frozen_sids) == P.ZERO_VAR_EXPECTED,
         f"frozen={sorted(const_frozen_sids)} rows={const_counts}")

    # ---- R7 fill == set difference ----
    r7_bad = []
    for s in P.SEEDS:
        expected_new = {k for k in all_keys if k[0] == s} - covered_by_seed[s]
        new_s = {k for k in new_keys if k[0] == s}
        if new_s != expected_new:
            r7_bad.append(f"seed {s}: new {len(new_s)} != diff "
                          f"{len(expected_new)}")
        if {k for k in cache_keys if k[0] == s} != covered_by_seed[s]:
            r7_bad.append(f"seed {s}: cache keys != covered set")
    rule("R7_fill_set_difference", not r7_bad and len(rows) == P.PRED_TOTAL,
         "; ".join(r7_bad[:4]) + f" rows={len(rows)}")
    rep_path = os.path.join(C.RESULTS_DIR, "assemble_report.json")
    rep = C.load_json(rep_path)
    rule("R7_assemble_report",
         rep["rows"] == len(rows) and rep["cache_rows"] == len(cache_keys)
         and rep["new_rows"] == len(new_keys)
         and rep["fill_equals_set_difference"] is True, str(rep)[:200])

    # ---- R5 recompute ----
    rec = C.load_json(os.path.join(C.RESULTS_DIR, "recompute_check.json"))
    rule("R5_recompute", rec.get("passed") is True,
         str(rec.get("failures"))[:200])

    # ---- R6 VisionTS gate ----
    gate = C.load_json(os.path.join(C.RESULTS_DIR, "vts_gate.json"))
    rule("R6_vts_gate",
         gate.get("passed") is True
         and gate["rel_err"]["std_mse"] <= P.REF_VTS_TOL
         and gate["rel_err"]["std_mae"] <= P.REF_VTS_TOL,
         f"rel_err={gate.get('rel_err')}")

    # ---- subset report sanity (row-count enforced in evaluate; flag here) ----
    subset = C.load_json(os.path.join(C.RESULTS_DIR, "subset_2474.json"))
    ln = C.load_json(P.LN_MANIFEST)
    ln_keys = set()
    for vk, v in ln["variables"].items():
        for t in v["native"]["test"]:
            ln_keys.add(t if isinstance(t, str) else t["sample_id"])
    n_sub_rows = sum(1 for r in rows if r["sample_id"] in ln_keys)
    rule("subset_2474_rows", n_sub_rows == P.N_LN_TEST * len(P.SEEDS),
         f"{n_sub_rows} != {P.N_LN_TEST * len(P.SEEDS)}")

    report = {
        "stage": "acceptance",
        "passed": not failures,
        "rules": {
            "R1_keyset_equal_frozen": rule_results.get("R1_keyset", False)
                and rule_results.get("R1234_rows", False),
            "R2_finite_shape": rule_results.get("R1234_rows", False),
            "R3_constant_windows_4x3_calib":
                rule_results.get("R3_constant_windows", False),
            "R4_provenance_file_sha_npz_row": rule_results.get("R4_files", False)
                and rule_results.get("R1234_rows", False),
            "R5_recompute_pass": rule_results.get("R5_recompute", False),
            "R6_vts_gate_1e-12": rule_results.get("R6_vts_gate", False),
            "R7_fill_equals_set_difference":
                rule_results.get("R7_fill_set_difference", False)
                and rule_results.get("R7_assemble_report", False),
        },
        "counts": {"rows": len(rows), "cache": len(cache_keys),
                   "new": len(new_keys),
                   "constant_rows_per_seed": {str(s): len(const_rows[s])
                                              for s in P.SEEDS}},
        "subset_2474": {"n_rows": n_sub_rows, "pure_cache": subset["pure_cache"],
                        "reproduced": subset.get("reproduced")},
        "failures": failures,
    }
    C.dump_json(report, os.path.join(C.RESULTS_DIR, "acceptance.json"))
    if failures:
        for x in failures:
            print(f"[accept] FAIL {x}")
        raise SystemExit(f"acceptance FAILED ({len(failures)}) — stop, locate, "
                         "never loosen thresholds")
    C.mark_done("accept", {
        "passed": True,
        "subset_reproduced": subset.get("reproduced"),
        "sha": {"results/acceptance.json": C.sha256_file(
            os.path.join(C.RESULTS_DIR, "acceptance.json"))},
        "input_guards": C.stage_input_guards("accept")})
    print(f"[accept] R1-R7 ALL PASS ({len(rows)} rows, cache={len(cache_keys)}, "
          f"new={len(new_keys)}, subset pure_cache={subset['pure_cache']})")


if __name__ == "__main__":
    main()
