#!/usr/bin/env python3
"""Extract the frozen D2 test cache + preflight train/val text from the
text-budget diagnostic traces (analysis-text-budget-v1 @ 3e09d4b).

Outputs (repro_mm_timesx_d2/cache/, gitignored):
  text_tokens_M48T512_D2.npz    ids/mask int32 [2474,512] + var_keys/sample_ids/domains
                                (schema identical to the EXP-011 frozen cache)
  text_meta_M48T512_D2.jsonl    per-window trace records (test keys, npz row order)
  preflight_text_trainval.npz   D0+D2 ids/mask for deterministic train/val windows
                                (PF preflight ONLY; never test/excluded keys)

Hard gates (assert): extraction rows bitwise-equal the source trace rows; test
key set == manifest native.test == EXP-011 frozen cache keys == test_rows();
D0 trace rows bitwise-equal the EXP-011 frozen cache on the test keys; preflight
keys are train/val only. Prints every hash needed to freeze protocol_d2.json.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SUBDIR = Path(__file__).resolve().parent
PROJECT = SUBDIR.parent
sys.path.insert(0, str(PROJECT))

TRACE_DIR = PROJECT / "analysis" / "text_budget_v1" / "outputs"
EXP011_CACHE = (PROJECT / "repro_mm_timesx_v1" / "cache" /
                "text_tokens_M48T512.npz")
FROZEN_COMMIT = "3e09d4b715c2ecdd803688b11ad6856f4231be8e"
D2_SRC_FILES = ["clean_rules.py", "allocate.py", "build_diag.py",
                "d2_rules.json", "budget_config.json"]


def sha256_file(p):
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()


def md5_file(p):
    h = hashlib.md5()
    h.update(Path(p).read_bytes())
    return h.hexdigest()


def manifest_scopes():
    manifest = json.loads((PROJECT / "data" / "split_manifest.json").read_text())
    scope_of, test_keys = {}, set()
    for vk, v in manifest["variables"].items():
        nat = v["native"]
        for sid in nat.get("train", []):
            scope_of[(vk, sid)] = "train"
        for sid in nat.get("val", []):
            scope_of[(vk, sid)] = "val"
        for sid in nat.get("test", []):
            scope_of[(vk, sid)] = "test"
            test_keys.add((vk, sid))
        for e in nat.get("excluded", []):
            scope_of[(vk, e["sample_id"])] = f"excluded:{e['reason']}"
    return manifest, scope_of, test_keys


def trace_index(z):
    keys = list(zip([str(x) for x in z["var_keys"]],
                    [str(x) for x in z["sample_ids"]]))
    assert len(set(keys)) == len(keys), "duplicate keys in trace npz"
    return keys, {k: i for i, k in enumerate(keys)}


def main():
    cache_dir = SUBDIR / "cache"
    cache_dir.mkdir(exist_ok=True)

    manifest, scope_of, test_keys = manifest_scopes()
    assert len(test_keys) == 2474, f"manifest test keys {len(test_keys)} != 2474"

    # ---- D2 test cache extraction -----------------------------------------
    z2 = np.load(TRACE_DIR / "trace_D2.npz", allow_pickle=False)
    keys2, idx2 = trace_index(z2)
    sel = sorted(test_keys)
    missing = [k for k in sel if k not in idx2]
    assert not missing, f"test keys missing from trace_D2: {missing[:5]}"
    rows = np.array([idx2[k] for k in sel])
    ids2 = z2["ids"][rows]
    mask2 = z2["mask"][rows]
    assert ids2.shape == (2474, 512) and ids2.dtype == np.int32
    vks = np.array([k[0] for k in sel], dtype=np.str_)
    sids = np.array([k[1] for k in sel], dtype=np.str_)
    doms = np.array([k[0].split("__", 1)[0] for k in sel], dtype=np.str_)

    out_npz = cache_dir / "text_tokens_M48T512_D2.npz"
    tmp = Path(str(out_npz) + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, ids=ids2, mask=mask2, var_keys=vks, sample_ids=sids,
                 domains=doms)
    tmp.replace(out_npz)
    zchk = np.load(out_npz, allow_pickle=False)
    assert np.array_equal(zchk["ids"], ids2) and np.array_equal(zchk["mask"], mask2)
    assert zchk["ids"].dtype == np.int32 and zchk["mask"].dtype == np.int32

    # key set == EXP-011 frozen cache keys == data_loader.test_rows() keys
    z11 = np.load(EXP011_CACHE, allow_pickle=False)
    keys11_list = list(zip([str(x) for x in z11["var_keys"]],
                           [str(x) for x in z11["sample_ids"]]))
    assert len(set(keys11_list)) == 2474
    assert set(keys11_list) == test_keys, "EXP-011 cache keys != manifest test keys"
    import data_loader
    rows_test = data_loader.TimesXData(PROJECT / "data").test_rows()
    assert {(vk, sid) for vk, sid, *_ in rows_test} == test_keys

    # D0 trace rows bitwise-equal EXP-011 frozen cache on the test keys
    z0 = np.load(TRACE_DIR / "trace_D0.npz", allow_pickle=False)
    keys0, idx0 = trace_index(z0)
    r0 = np.array([idx0[k] for k in keys11_list])
    assert np.array_equal(z0["ids"][r0], z11["ids"]), "D0 trace != EXP-011 cache"
    assert np.array_equal(z0["mask"][r0], z11["mask"]), "D0 trace != EXP-011 cache"

    # meta jsonl in npz row order
    meta_out = cache_dir / "text_meta_M48T512_D2.jsonl"
    rec_by_key = {}
    with open(TRACE_DIR / "trace_D2.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rec_by_key[(r["var_key"], r["sample_id"])] = r
    with open(meta_out, "w", encoding="utf-8") as f:
        for k in sel:
            rec = rec_by_key[k]
            assert rec["scope"] == "test" and rec["domain"] == k[0].split("__", 1)[0]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- preflight train/val text (D0+D2) ---------------------------------
    td = data_loader.TimesXData(PROJECT / "data")
    daily = [d for d in data_loader.DOMAIN_NAMES if td.manifest["variables"][
        td.domain_vars[d][0]]["frequency"] == "daily"]
    weekly = [d for d in data_loader.DOMAIN_NAMES if d not in daily]
    pf = []
    for d in (daily[:2] + weekly[:2]):
        pf.extend(td.trainval_rows(d, max_per_var=1, splits=("train",))[:1])
        pf.extend(td.trainval_rows(d, max_per_var=1, splits=("val",))[:1])
    assert len(pf) == 8, f"preflight windows {len(pf)} != 8"
    pf_keys = [(vk, sid) for vk, sid, *_ in pf]
    for k in pf_keys:
        assert scope_of[k] in ("train", "val"), f"preflight key not train/val: {k}"
        assert k not in test_keys, f"preflight key is a test key: {k}"
    ids0pf = z0["ids"][[idx0[k] for k in pf_keys]]
    mask0pf = z0["mask"][[idx0[k] for k in pf_keys]]
    ids2pf = z2["ids"][[idx2[k] for k in pf_keys]]
    mask2pf = z2["mask"][[idx2[k] for k in pf_keys]]
    pf_npz = cache_dir / "preflight_text_trainval.npz"
    tmp = Path(str(pf_npz) + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f,
                 var_keys=np.array([k[0] for k in pf_keys], dtype=np.str_),
                 sample_ids=np.array([k[1] for k in pf_keys], dtype=np.str_),
                 scope=np.array([scope_of[k] for k in pf_keys], dtype=np.str_),
                 domain=np.array([p[2] for p in pf], dtype=np.str_),
                 freq=np.array([manifest["variables"][k[0]]["frequency"]
                                for k in pf_keys], dtype=np.str_),
                 ids_d0=ids0pf, mask_d0=mask0pf,
                 ids_d2=ids2pf, mask_d2=mask2pf)
    tmp.replace(pf_npz)

    # ---- provenance --------------------------------------------------------
    prov = {
        "frozen_source_branch": "analysis-text-budget-v1",
        "frozen_source_commit": FROZEN_COMMIT,
        "trace_d0_npz_sha256": sha256_file(TRACE_DIR / "trace_D0.npz"),
        "trace_d2_npz_sha256": sha256_file(TRACE_DIR / "trace_D2.npz"),
        "trace_d2_jsonl_sha256": sha256_file(TRACE_DIR / "trace_D2.jsonl"),
        "cache_npz_sha256": sha256_file(out_npz),
        "cache_meta_jsonl_sha256": sha256_file(meta_out),
        "preflight_npz_sha256": sha256_file(pf_npz),
        "d2_source_code_md5": {n: md5_file(PROJECT / "analysis" / "text_budget_v1" / n)
                               for n in D2_SRC_FILES},
        "checks": {
            "test_rows": len(sel),
            "keys_equal_exp011_cache": True,
            "keys_equal_test_rows": True,
            "bitwise_vs_trace_d2": True,
            "d0_trace_bitwise_exp011_cache": True,
            "preflight_windows": len(pf),
            "preflight_scopes": sorted({scope_of[k] for k in pf_keys}),
        },
    }
    (SUBDIR / "results").mkdir(exist_ok=True)
    (SUBDIR / "results" / "cache_provenance.json").write_text(
        json.dumps(prov, indent=2))
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
