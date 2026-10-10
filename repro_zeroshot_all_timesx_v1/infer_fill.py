#!/usr/bin/env python3
"""S4 infer-fill (GPU host): forward every (seed, window) in the fill set —
the SET DIFFERENCE between the 8,106 frozen keys and the keys actually covered
by final-reusable old cache — and store per-(var, seed) npz under cache_all/.

Resume: a manifest entry is honoured only when the file exists with the full
key set, its recorded sha256 matches, n and sample_ids match the wanted list,
and all predictions are finite. Any mismatch => that (var, seed) is re-inferred.

Binding: reuse_final.json.forward_binding must equal the freshly computed
forward fingerprint (same host/env/code/data as S3), otherwise refuse to run.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import common as C
import protocol as P


def reusable_by_seed(final):
    out = {}
    for key, v in final["variables"].items():
        s_str, vk = key.split("|", 1)
        if v["verdict"] == "reuse":
            out.setdefault(int(s_str), set()).add(vk)
    return out


def covered_sids(reuse_by_seed, plan):
    """{seed: set(sample_id)} actually present in the reusable old shards."""
    covered = {}
    for s, vars_s in reuse_by_seed.items():
        cov = set()
        for vk in sorted(vars_s):
            v = plan["per_seed"][str(s)]["vars"][vk]
            if v["verdict"] != "reuse_candidate":
                raise SystemExit(f"{vk} final-reuse at seed {s} but S1b verdict "
                                 f"{v['verdict']} — inconsistent state")
            path = os.path.join(P.OLD_PRED_DIR, v["file"])
            if C.sha256_file(path) != v["sha256"]:
                raise SystemExit(f"old shard {v['file']} changed since S1b")
            with np.load(path, allow_pickle=False) as z:
                for i, sid in enumerate(str(x) for x in z["sample_ids"]):
                    if str(z["var_keys"][i]) == vk:
                        cov.add(sid)
        covered[s] = cov
    return covered


def validate_entry(npz_path, recorded_sha, want_sids):
    if not os.path.isfile(npz_path):
        return False, "file missing"
    with np.load(npz_path, allow_pickle=False) as z:
        if set(z.files) != {"sample_ids", "pred"}:
            return False, f"key set {sorted(z.files)}"
        if C.sha256_file(npz_path) != recorded_sha:
            return False, "sha256 mismatch"
        sids = [str(s) for s in z["sample_ids"]]
        if sids != want_sids:
            return False, "sample_ids mismatch"
        if z["pred"].shape != (len(want_sids), P.PRED_LEN) or not np.isfinite(z["pred"]).all():
            return False, "pred invalid"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    ap.add_argument("--only-var", default=None, help="substring filter (smoke; does not write .state)")
    ap.add_argument("--out-dir", default=C.CACHE_DIR)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    inv = C.load_json(C.INV_PATH)
    final = C.load_json(C.REUSE_FINAL)

    fp = C.forward_fingerprint()
    binding = final.get("forward_binding")
    if binding is None:
        raise SystemExit("reuse_final.json has no forward_binding — preflight predates "
                         "the binding; re-run S3 (--run)")
    if binding != fp:
        diff = [k for k in sorted(set(binding) | set(fp))
                if binding.get(k) != fp.get(k)]
        raise SystemExit(f"forward_binding mismatch on keys {diff} — "
                         "model/config/code/data/env changed since S3; re-run preflight")

    plan = C.load_json(C.REUSE_PLAN)
    reuse_by_seed = reusable_by_seed(final)
    covered = {s: set() for s in P.SEEDS}   # a seed may have zero reusable cache
    covered.update(covered_sids(reuse_by_seed, plan))
    win_by_sid = {w["sample_id"]: w for w in inv["windows"]}
    all_sids = [w["sample_id"] for w in sorted(inv["windows"],
                                               key=lambda w: w["flat_index"])]
    fill_by_seed = {}
    for s in P.SEEDS:
        fill = [sid for sid in all_sids if sid not in covered[s]]
        expect = final["totals"]["per_seed"][str(s)]["fill_windows"]
        if len(fill) != expect:
            raise SystemExit(f"seed {s}: fill set {len(fill)} != reuse_final "
                             f"fill_windows {expect}")
        if len(covered[s]) + len(fill) != P.TOTAL_WINDOWS:
            raise SystemExit(f"seed {s}: covered+fill != {P.TOTAL_WINDOWS}")
        fill_by_seed[s] = fill
        print(f"[fill] seed {s}: covered(cached)={len(covered[s])} "
              f"fill={len(fill)} (expected fill={expect})")

    import torch
    from load_aurora import load_aurora
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    model, load_rep = load_aurora(None, args.device, verify_package=False)
    import predict as OP

    arrays = np.load(C.ARRAYS_PATH)
    past_all = arrays["past"]

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    manifest = {"fingerprint": fp, "entries": {}}
    if os.path.isfile(manifest_path) and not args.force:
        old = C.load_json(manifest_path)
        if old.get("fingerprint") == fp:
            manifest["entries"] = old.get("entries", {})
        else:
            raise SystemExit("cache_all/manifest.json fingerprint mismatch — refusing to "
                             "mix; pass --force to restart cache_all")

    t0 = time.time()
    n_fwd = 0
    entries = manifest["entries"]
    for s in P.SEEDS:
        by_var = {}
        for sid in fill_by_seed[s]:
            by_var.setdefault(win_by_sid[sid]["var_key"], []).append(sid)
        if args.only_var:
            by_var = {k: v for k, v in by_var.items() if args.only_var in k}
        for vk in sorted(by_var):
            sids = sorted(by_var[vk], key=lambda sid: win_by_sid[sid]["flat_index"])
            key = f"{vk}__s{s}"
            npz_path = os.path.join(args.out_dir, f"{key}.npz")
            ent = entries.get(key)
            if ent:
                ok, why = validate_entry(npz_path, ent.get("sha256"), sids)
                if ok:
                    continue
                print(f"[fill] {key}: invalid entry ({why}) — re-inferring")
            preds = np.zeros((len(sids), P.PRED_LEN), dtype=np.float64)
            with torch.inference_mode():
                for i, sid in enumerate(sids):
                    w = win_by_sid[sid]
                    preds[i] = OP.predict_window(
                        model, past_all[w["flat_index"]], P.ITL, P.NUM_SAMPLES,
                        OP.derive_seed(s, vk, sid))
            tmp = npz_path + ".tmp.npz"
            with open(tmp, "wb") as f:
                np.savez(f, sample_ids=np.array(sids, dtype=np.str_), pred=preds)
            os.replace(tmp, npz_path)
            sha = C.sha256_file(npz_path)
            entries[key] = {"seed": s, "var_key": vk, "n": len(sids),
                            "file": os.path.basename(npz_path), "sha256": sha}
            manifest["entries"] = entries
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)
            n_fwd += len(sids)
            print(f"[fill] {key}: {len(sids)} windows ({time.time()-t0:.0f}s)")

    print(f"[fill] done: forwards_executed={n_fwd}, entries={len(entries)}, "
          f"{time.time()-t0:.1f}s -> {args.out_dir}")
    if not args.only_var:
        C.mark_done("s4_infer_fill", {
            "fill_windows_total": n_fwd,
            "entries": len(entries),
            "sha": {"cache_all/manifest.json": C.sha256_file(manifest_path)},
            "input_guards": C.stage_input_guards("fill")})


if __name__ == "__main__":
    main()
