#!/usr/bin/env python3
"""S1a freeze-ref: copy the frozen VisionTS EXP-014 references into
visionts_ref/ with sha256 assertions, then cross-check the reference data
itself (counts, d/denom_source recomputation, LN-test subset, calib_std).

Read-only sources:
  {VTS}/frozen/inventory.json      {VTS}/frozen/frozen_arrays.npz
  {VTS}/frozen/vars.json           {VTS}/results/preds.jsonl.gz
  {LN}  split_manifest.json (sha-pinned by the OLD protocol too)
"""
import argparse
import gzip
import json
import os
import shutil

import numpy as np

import common as C
import protocol as P


def sha_ok(path, expect):
    got = C.sha256_file(path)
    if got != expect:
        raise SystemExit(f"sha256 mismatch for {path}: {got} != {expect}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(C.REF_DIR, exist_ok=True)
    srcs = {
        "inventory.json": (os.path.join(P.VTS_DIR, "frozen", "inventory.json"),
                           P.VTS_INVENTORY_SHA256),
        "frozen_arrays.npz": (os.path.join(P.VTS_DIR, "frozen", "frozen_arrays.npz"),
                              P.VTS_ARRAYS_SHA256),
        "vars.json": (os.path.join(P.VTS_DIR, "frozen", "vars.json"),
                      P.VTS_VARS_SHA256),
        "preds.jsonl.gz": (os.path.join(P.VTS_DIR, "results", "preds.jsonl.gz"),
                           P.VTS_PREDS_SHA256),
    }
    for name, (src, sha) in srcs.items():
        sha_ok(src, sha)
        dst = os.path.join(C.REF_DIR, name)
        if args.force or not os.path.isfile(dst) or C.sha256_file(dst) != sha:
            shutil.copyfile(src, dst)
            sha_ok(dst, sha)
        print(f"[freeze_ref] {name}: sha ok")

    ln_sha = C.sha256_file(P.LN_MANIFEST_SOURCE)
    if ln_sha != P.OLD_SPLIT_MANIFEST_SHA256:
        raise SystemExit(f"LN manifest sha {ln_sha} != old-protocol pin "
                         f"{P.OLD_SPLIT_MANIFEST_SHA256}")
    ln_dst = os.path.join(C.REF_DIR, "split_manifest.json")
    if args.force or not os.path.isfile(ln_dst) or C.sha256_file(ln_dst) != ln_sha:
        shutil.copyfile(P.LN_MANIFEST_SOURCE, ln_dst)
        if C.sha256_file(ln_dst) != ln_sha:
            raise SystemExit("LN manifest copy sha mismatch")
    print("[freeze_ref] LN manifest frozen (sha matches the old EXP-010 pin)")

    inv = C.load_json(C.INV_PATH)
    wins = inv["windows"]
    if len(wins) != P.TOTAL_WINDOWS:
        raise SystemExit(f"inventory windows {len(wins)} != {P.TOTAL_WINDOWS}")
    sids = [w["sample_id"] for w in wins]
    if len(set(sids)) != len(sids):
        raise SystemExit("inventory sample_id not unique")
    domains = {w["domain"] for w in wins}
    if domains != set(P.EXPECTED_DOMAINS):
        raise SystemExit("inventory domain set mismatch")
    vars_meta = {v["var_key"]: v for v in C.load_json(C.VARS_PATH)}
    if len(vars_meta) != P.N_VARS:
        raise SystemExit(f"vars {len(vars_meta)} != {P.N_VARS}")
    split_census = {}
    for w in wins:
        split_census[w["split_in_old_protocol"]] = split_census.get(w["split_in_old_protocol"], 0) + 1
    print(f"[freeze_ref] inventory ok: {len(wins)} windows / {len(vars_meta)} vars / "
          f"{len(domains)} domains; splits={split_census}")

    arrays = np.load(C.ARRAYS_PATH)
    past, target = arrays["past"], arrays["target"]
    d_stored, dsrc_stored = arrays["d"], arrays["denom_source"]
    flat = arrays["flat_index"]
    assert past.shape == (P.TOTAL_WINDOWS, P.CONTEXT_LEN)
    assert target.shape == (P.TOTAL_WINDOWS, P.PRED_LEN)
    assert list(flat) == list(range(P.TOTAL_WINDOWS)), "flat_index not 0..8105"
    if not (np.isfinite(past).all() and np.isfinite(target).all()):
        raise SystemExit("frozen arrays contain non-finite values")

    bad_d, bad_src = 0, 0
    calib_of = {}
    for i, w in enumerate(wins):
        ws = float(np.std(past[i]))
        vk = w["var_key"]
        if ws >= P.EPS_STD:
            d, src = ws, "window"
        else:
            if vk not in calib_of:
                calib_of[vk] = float(vars_meta[vk]["calib_std"])
            d, src = calib_of[vk], "calib_std"
        if abs(d - float(d_stored[i])) > P.RECOMPUTE_TOL_D * max(1.0, abs(float(d_stored[i]))):
            bad_d += 1
        if src != str(dsrc_stored[i]):
            bad_src += 1
    if bad_d or bad_src:
        raise SystemExit(f"frozen d/denom_source mismatch: {bad_d} d, {bad_src} source")
    n_calib = int((dsrc_stored == "calib_std").sum())
    if n_calib != P.ZERO_VAR_EXPECTED:
        raise SystemExit(f"calib-std windows {n_calib} != {P.ZERO_VAR_EXPECTED}")
    print(f"[freeze_ref] d/denom_source recompute ok ({n_calib} calib fallback)")

    ln = C.load_json(P.LN_MANIFEST)
    ln_keys = set()
    for vk, v in ln["variables"].items():
        for t in v["native"]["test"]:
            ln_keys.add(t if isinstance(t, str) else t["sample_id"])
    if len(ln_keys) != P.N_LN_TEST:
        raise SystemExit(f"LN native.test {len(ln_keys)} != {P.N_LN_TEST}")
    inv_keys = set(sids)
    if not ln_keys <= inv_keys:
        raise SystemExit(f"{len(ln_keys - inv_keys)} LN test keys missing from inventory")
    print(f"[freeze_ref] LN test subset ok: {len(ln_keys)} keys, all inside inventory")

    ln_vars = ln["variables"]
    bad_fb = [vk for vk, v in vars_meta.items()
              if abs(float(v["calib_std"]) - float(ln_vars[vk]["fallback_std"]))
              > P.RECOMPUTE_TOL_D * max(1.0, abs(float(v["calib_std"])))]
    if bad_fb:
        raise SystemExit(f"vars.json calib_std vs LN fallback_std mismatch: {bad_fb[:5]}")
    zv = [w for i, w in enumerate(wins) if str(dsrc_stored[i]) == "calib_std"]
    zv_not_ln = [w for w in zv if w["sample_id"] not in ln_keys]
    print(f"[freeze_ref] calib fallback cross-check ok; zero-var windows "
          f"{len(zv)}, outside LN test: {len(zv_not_ln)}")

    # spot: VisionTS preds row count
    n_rows = 0
    with gzip.open(C.VTS_PREDS_PATH, "rt") as f:
        for _ in f:
            n_rows += 1
    if n_rows != P.TOTAL_WINDOWS:
        raise SystemExit(f"VisionTS preds rows {n_rows} != {P.TOTAL_WINDOWS}")

    report = {
        "stage": "S1a_freeze_ref",
        "protocol_version": P.PROTOCOL_VERSION,
        "sources": {k: {"sha256": v[1]} for k, v in srcs.items()},
        "ln_manifest_sha256": ln_sha,
        "windows": len(wins), "vars": len(vars_meta), "domains": len(domains),
        "split_census": split_census,
        "ln_test_keys": len(ln_keys),
        "calib_fallback_windows": n_calib,
        "d_mismatch": bad_d, "denom_source_mismatch": bad_src,
        "vts_pred_rows": n_rows,
    }
    C.dump_json(report, C.REF_REPORT)
    C.mark_done("s1a_freeze_ref", {
        "windows": len(wins), "ln_test": len(ln_keys),
        "sha": {"visionts_ref/ref_report.json": C.sha256_file(C.REF_REPORT),
                "visionts_ref/inventory.json": C.sha256_file(os.path.join(C.REF_DIR, "inventory.json")),
                "visionts_ref/frozen_arrays.npz": C.sha256_file(os.path.join(C.REF_DIR, "frozen_arrays.npz")),
                "visionts_ref/vars.json": C.sha256_file(os.path.join(C.REF_DIR, "vars.json")),
                "visionts_ref/preds.jsonl.gz": C.sha256_file(os.path.join(C.REF_DIR, "preds.jsonl.gz")),
                "visionts_ref/split_manifest.json": C.sha256_file(os.path.join(C.REF_DIR, "split_manifest.json"))},
        "input_guards": C.stage_input_guards("freeze_ref")})
    print("[freeze_ref] done -> visionts_ref/ + ref_report.json")


if __name__ == "__main__":
    main()
