#!/usr/bin/env python3
"""S5a assemble: merge old-cache rows (source=cache) + new rows (source=new)
into results/aurora_preds.jsonl.gz — exactly PRED_TOTAL rows, each carrying
identity, target, frozen d/denom_source, and provenance (file + sha256,
re-hashed at merge time). Asserts the fill set equals the exact set difference
per seed and that every target matches the frozen array bitwise."""
import argparse
import gzip
import json
import os

import numpy as np

import common as C
import protocol as P


def reuse_keys_from_final(final, plan):
    """{(seed, sid): (pred_file, pred_sha256, row_index)} for final-reuse vars."""
    out = {}
    for key, v in final["variables"].items():
        if v["verdict"] != "reuse":
            continue
        s_str, vk = key.split("|", 1)
        s = int(s_str)
        rec = plan["per_seed"][s_str]["vars"][vk]
        out.setdefault((s, vk), (rec["file"], rec["sha256"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    inv = C.load_json(C.INV_PATH)
    win_by_sid = {w["sample_id"]: w for w in inv["windows"]}
    arrays = np.load(C.ARRAYS_PATH)
    target_all, d_all, dsrc_all = arrays["target"], arrays["d"], arrays["denom_source"]
    final = C.load_json(C.REUSE_FINAL)
    plan = C.load_json(C.REUSE_PLAN)
    manifest = C.load_json(C.MANIFEST_PATH)

    # ---- cache rows (old shards, final-reuse vars only), re-hashed ----
    cache_rows = {}
    for (s, vk), (fname, sha) in reuse_keys_from_final(final, plan).items():
        path = os.path.join(P.OLD_PRED_DIR, fname)
        got = C.sha256_file(path)
        if got != sha:
            raise SystemExit(f"old shard {fname} sha changed since S1b")
        with np.load(path, allow_pickle=False) as z:
            sids = [str(x) for x in z["sample_ids"]]
            vks = [str(x) for x in z["var_keys"]]
            for i in range(len(sids)):
                if vks[i] != vk:
                    continue
                if (s, sids[i]) in cache_rows:
                    raise SystemExit(f"duplicate cache row seed={s} {sids[i]}")
                cache_rows[(s, sids[i])] = (fname, sha, z["pred"][i])

    # ---- new rows (cache_all manifest entries), re-hashed ----
    new_rows = {}
    for key, ent in manifest["entries"].items():
        path = os.path.join(C.CACHE_DIR, ent["file"])
        got = C.sha256_file(path)
        if got != ent["sha256"]:
            raise SystemExit(f"cache_all {ent['file']} sha mismatch vs manifest")
        with np.load(path, allow_pickle=False) as z:
            sids = [str(x) for x in z["sample_ids"]]
            if len(sids) != ent["n"] or z["pred"].shape != (ent["n"], P.PRED_LEN) \
                    or not np.isfinite(z["pred"]).all():
                raise SystemExit(f"cache_all {ent['file']} content invalid")
            for i, sid in enumerate(sids):
                s = ent["seed"]
                if (s, sid) in new_rows:
                    raise SystemExit(f"duplicate new row seed={s} {sid}")
                new_rows[(s, sid)] = (ent["file"], ent["sha256"], z["pred"][i])

    # ---- fill-set arithmetic: new == frozen keys − cache keys, per seed ----
    all_keys = {(s, w["sample_id"]) for s in P.SEEDS for w in inv["windows"]}
    expected_new = all_keys - set(cache_rows)
    if set(new_rows) != expected_new:
        missing = sorted(expected_new - set(new_rows))[:5]
        extra = sorted(set(new_rows) - expected_new)[:5]
        raise SystemExit(f"fill set mismatch: missing={missing} extra={extra}")
    if set(new_rows) & set(cache_rows):
        raise SystemExit("cache/new overlap non-empty")

    # ---- emit ----
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(C.RESULTS_DIR, "aurora_preds.jsonl.gz")
    n_written = 0
    with gzip.open(out_path, "wt") as f:
        for s in P.SEEDS:
            for w in sorted(inv["windows"], key=lambda w: w["flat_index"]):
                sid = w["sample_id"]
                fi = w["flat_index"]
                if (s, sid) in cache_rows:
                    fname, sha, pred = cache_rows[(s, sid)]
                    source = "cache"
                else:
                    fname, sha, pred = new_rows[(s, sid)]
                    source = "new"
                pred = np.asarray(pred, dtype=np.float64)
                if pred.shape != (P.PRED_LEN,) or not np.isfinite(pred).all():
                    raise SystemExit(f"bad pred seed={s} {sid}")
                row = {
                    "seed": s, "flat_index": fi, "var_key": w["var_key"],
                    "sample_id": sid, "domain": w["domain"], "source": source,
                    "pred_file": fname, "pred_file_sha256": sha,
                    "pred": [float(x) for x in pred],
                    "target": [float(x) for x in target_all[fi]],
                    "d": float(d_all[fi]), "denom_source": str(dsrc_all[fi]),
                }
                f.write(json.dumps(row) + "\n")
                n_written += 1
    if n_written != P.PRED_TOTAL:
        raise SystemExit(f"wrote {n_written} rows != {P.PRED_TOTAL}")

    report = {
        "stage": "S5a_assemble",
        "rows": n_written,
        "cache_rows": len(cache_rows),
        "new_rows": len(new_rows),
        "cache_rows_per_seed": {str(s): sum(1 for k in cache_rows if k[0] == s)
                                for s in P.SEEDS},
        "new_rows_per_seed": {str(s): sum(1 for k in new_rows if k[0] == s)
                              for s in P.SEEDS},
        "fill_equals_set_difference": True,
    }
    C.dump_json(report, os.path.join(C.RESULTS_DIR, "assemble_report.json"))
    C.mark_done("s5a_assemble", {
        "rows": n_written, "cache": len(cache_rows), "new": len(new_rows),
        "sha": {"results/aurora_preds.jsonl.gz": C.sha256_file(out_path),
                "results/assemble_report.json": C.sha256_file(
                    os.path.join(C.RESULTS_DIR, "assemble_report.json"))},
        "input_guards": C.stage_input_guards("assemble")})
    print(f"[assemble] OK: {n_written} rows (cache={len(cache_rows)}, "
          f"new={len(new_rows)}) -> {out_path}")


if __name__ == "__main__":
    main()
