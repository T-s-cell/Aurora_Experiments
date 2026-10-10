#!/usr/bin/env python3
"""S1b verify-cache: verify every EXP-010 A48 shard (57 = 19 domains x 3 seeds)
against the frozen references BEFORE any new inference. Per shard, ALL of:

  1. header + .done parse; fingerprint pins match: weights_sha256, hf_revision,
     itl=48, num_samples=100, base_seed=shard seed, split_manifest_sha256,
     data_cache_sha256 (old hashes kept as provenance — never rewritten)
  2. content_sha256 (header AND .done) == sha256(npz)
  3. header.code_md5 == CURRENT md5 of the old repo files -> the seed rule and
     forward path that produced the cache are byte-identical to the read-only
     code we path-import
  4. header.aurora_package_md5 identical across all 57 shards (equality with
     the GPU-host package is asserted again at S3 preflight)
  5. n rows == header.n_rows; sample_ids subset of the fixed LN-test key set
     (2,474 identity); pred finite (N,12) float64
  6. target bitwise == frozen target row; d within rel 1e-12 of frozen d
  7. past96 rebuilt from the OLD frozen data_cache.npz (sha-pinned) matches the
     frozen past row for every window

A shard failing any check marks its (seed, var_key) verdict non-reusable; the
missing/invalid windows simply move to the fill set. The LN-test KEY SET must
be exactly 2,474 (identity); the number of *reusable* windows may be smaller.
Output: reuse_plan.json.
"""
import argparse
import json
import os

import numpy as np

import common as C
import protocol as P

REL_TOL_D = P.RECOMPUTE_TOL_D  # 1e-12 relative


def rel_diff(a, b):
    return abs(float(a) - float(b)) / max(1e-12, abs(float(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    inv = C.load_json(C.INV_PATH)
    win_by_key = {(w["var_key"], w["sample_id"]): w for w in inv["windows"]}
    arrays = np.load(C.ARRAYS_PATH)
    past_all, target_all, d_all = arrays["past"], arrays["target"], arrays["d"]

    ln = C.load_json(P.LN_MANIFEST)
    ln_keys = set()
    for vk, v in ln["variables"].items():
        for t in v["native"]["test"]:
            ln_keys.add(t if isinstance(t, str) else t["sample_id"])

    # old frozen data cache -> past96 rebuilding (path import of read-only code)
    dc_path = os.path.join(P.OLD_DATA_DIR, "data_cache.npz")
    dc_sha = C.sha256_file(dc_path)
    if dc_sha != P.OLD_DATA_CACHE_SHA256:
        raise SystemExit(f"old data_cache sha {dc_sha} != pin {P.OLD_DATA_CACHE_SHA256}")
    import data_loader  # read-only old module (numpy only)
    td = data_loader.TimesXData(P.OLD_DATA_DIR)

    cur_code_md5 = C.code_md5_map(P.OLD_HEADER_CODE_FILES, P.OLD_DIR)
    pkg_md5_ref = None
    per_seed = {}

    for seed in P.SEEDS:
        vars_out = {}
        n_shard_ok, n_shard_bad = 0, 0
        for domain in P.EXPECTED_DOMAINS:
            name = P.old_shard_name(domain, seed)
            npz_path = os.path.join(P.OLD_PRED_DIR, name)
            rec = {"file": name, "sha256": None, "n": 0, "verdict": "pending",
                   "reason": None}
            if not os.path.isfile(npz_path):
                rec["verdict"] = "missing"
                rec["reason"] = "shard file absent"
                for vk in sorted({w["var_key"] for w in inv["windows"] if w["domain"] == domain}):
                    vars_out.setdefault(vk, rec)
                continue
            try:
                hdr_p = npz_path + ".header.json"
                done_p = npz_path + ".done"
                hdr = C.load_json(hdr_p)
                dn = C.load_json(done_p)
                fp = hdr["fingerprint"]
                if fp["weights_sha256"] != P.WEIGHTS_SHA256:
                    raise ValueError("weights_sha256 pin mismatch")
                if fp["hf_revision"] != P.HF_REVISION:
                    raise ValueError("hf_revision pin mismatch")
                if int(fp["itl"]) != P.ITL or int(fp["num_samples"]) != P.NUM_SAMPLES:
                    raise ValueError("itl/num_samples pin mismatch")
                if int(fp["base_seed"]) != seed:
                    raise ValueError("base_seed != shard seed")
                if fp["split_manifest_sha256"] != P.OLD_SPLIT_MANIFEST_SHA256:
                    raise ValueError("split_manifest_sha256 pin mismatch")
                if fp["data_cache_sha256"] != P.OLD_DATA_CACHE_SHA256:
                    raise ValueError("data_cache_sha256 pin mismatch")
                if fp["code_md5"] != cur_code_md5:
                    diff = {k for k in set(fp["code_md5"]) | set(cur_code_md5)
                            if fp["code_md5"].get(k) != cur_code_md5.get(k)}
                    raise ValueError(f"old code changed since the run: {sorted(diff)}")
                if pkg_md5_ref is None:
                    pkg_md5_ref = fp["aurora_package_md5"]
                elif fp["aurora_package_md5"] != pkg_md5_ref:
                    raise ValueError("aurora_package_md5 differs across shards")
                content = C.sha256_file(npz_path)
                if hdr.get("content_sha256") != content or dn.get("content_sha256") != content:
                    raise ValueError("content_sha256 != header/.done record")

                with np.load(npz_path, allow_pickle=False) as z:
                    sids = [str(s) for s in z["sample_ids"]]
                    vks = [str(v) for v in z["var_keys"]]
                    pred = z["pred"]
                    tgt = z["target"]
                    dv = z["d"]
                    n = len(sids)
                    if n != hdr.get("n_rows"):
                        raise ValueError("row count != header n_rows")
                    if pred.shape != (n, P.PRED_LEN) or not np.isfinite(pred).all():
                        raise ValueError("pred shape/non-finite")
                    bad = []
                    for i in range(n):
                        key = (vks[i], sids[i])
                        w = win_by_key.get(key)
                        if w is None or sids[i] not in ln_keys:
                            bad.append((key, "not an LN-test window"))
                            continue
                        fi = w["flat_index"]
                        if not np.array_equal(np.asarray(tgt[i], dtype=np.float64),
                                              target_all[fi]):
                            bad.append((key, "target mismatch vs frozen"))
                            continue
                        if rel_diff(dv[i], d_all[fi]) > REL_TOL_D:
                            bad.append((key, "d exceeds rel 1e-12"))
                            continue
                        x_old, _ = td.split_window_by_id(vks[i], sids[i])
                        if not np.array_equal(np.asarray(x_old, dtype=np.float64),
                                              past_all[fi]):
                            bad.append((key, "past96 mismatch vs frozen"))
                if bad:
                    raise ValueError(f"{len(bad)} window checks failed, e.g. {bad[:3]}")
                rec["verdict"] = "reuse_candidate"
                rec["sha256"] = content
                rec["n"] = n
                n_shard_ok += 1
            except Exception as e:  # noqa: BLE001 — every failure is recorded, never fatal
                rec["verdict"] = "invalid"
                rec["reason"] = str(e)[:300]
                n_shard_bad += 1
            # shard-level verdict applies to every var of the domain it covers;
            # a valid shard covers only its own sids, so store per-var entries
            var_sids = {}
            if rec["verdict"] == "reuse_candidate":
                with np.load(npz_path, allow_pickle=False) as z:
                    for vk_s in (str(v) for v in z["var_keys"]):
                        var_sids.setdefault(vk_s, 0)
                        var_sids[vk_s] += 1
                for vk, cnt in var_sids.items():
                    vars_out[vk] = {"file": name, "sha256": content, "n": cnt,
                                    "verdict": "reuse_candidate", "reason": None,
                                    "domain": domain}
            else:
                for vk in sorted({w["var_key"] for w in inv["windows"]
                                  if w["domain"] == domain}):
                    vars_out.setdefault(vk, dict(rec))
        per_seed[str(seed)] = {
            "vars": dict(sorted(vars_out.items())),
            "n_test_fixed": P.N_LN_TEST,
            "n_reuse_vars": sum(1 for v in vars_out.values()
                                if v["verdict"] == "reuse_candidate"),
            "n_reuse_windows": sum(int(v["n"]) for v in vars_out.values()
                                   if v["verdict"] == "reuse_candidate"),
        }
        print(f"[verify] seed {seed}: shards ok={n_shard_ok} bad/missing={n_shard_bad} "
              f"reuse_vars={per_seed[str(seed)]['n_reuse_vars']} "
              f"reuse_windows={per_seed[str(seed)]['n_reuse_windows']}")

    plan = {
        "protocol_version": P.PROTOCOL_VERSION,
        "created": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        "pins": {"weights_sha256": P.WEIGHTS_SHA256, "hf_revision": P.HF_REVISION,
                 "itl": P.ITL, "num_samples": P.NUM_SAMPLES,
                 "old_split_manifest_sha256": P.OLD_SPLIT_MANIFEST_SHA256,
                 "old_data_cache_sha256": P.OLD_DATA_CACHE_SHA256,
                 "old_code_md5": cur_code_md5,
                 "aurora_package_md5_expected_on_gpu": pkg_md5_ref},
        "per_seed": per_seed,
        "note": ("LN-test KEY SET is fixed at 2,474 per seed; n_reuse_windows is "
                 "the ACTUAL verified-reusable count and may be smaller. "
                 "aurora_package_md5_expected_on_gpu is re-asserted at S3."),
    }
    C.dump_json(plan, C.REUSE_PLAN)
    C.mark_done("s1b_verify_cache", {
        "shards_ok": n_shard_ok, "shards_bad_or_missing": n_shard_bad,
        "reuse_windows_per_seed": {s: per_seed[s]["n_reuse_windows"] for s in per_seed},
        "sha": {"reuse_plan.json": C.sha256_file(C.REUSE_PLAN)},
        "input_guards": C.stage_input_guards("verify")})
    print(f"[verify] done: {n_shard_ok} shards verified, "
          f"{n_shard_bad} bad/missing -> {C.REUSE_PLAN}")


if __name__ == "__main__":
    main()
