#!/usr/bin/env python3
"""Structural verification for analysis-text-budget-v1 outputs.

Checks (all windows, all schemes): ids[0]==CLS, SEP exactly at content+1,
mask == ones(content+2)+zeros, pad after SEP, per-scheme npz/jsonl key
identity, full 8106 key set, test key set == frozen 2474, and re-asserts the
D0 bitwise gate against the frozen EXP-011 cache.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys_p = str(PROJECT / "repro_mm_timesx_v1")
import sys  # noqa: E402
sys.path.insert(0, sys_p)
sys.path.insert(0, str(HERE))

FROZEN_NPZ = PROJECT / "repro_mm_timesx_v1" / "cache" / "text_tokens_M48T512.npz"
OUT = HERE / "outputs"
SCHEMES = ("D0", "D1", "D2")
CLS_ID, SEP_ID, PAD_ID = 101, 102, 0
TOTAL_LEN = 512


def main():
    ok = True
    keys_ref = None
    test_keys = set()
    for s in SCHEMES:
        z = np.load(OUT / f"trace_{s}.npz", allow_pickle=False)
        ids, mask = z["ids"], z["mask"]
        n = ids.shape[0]
        assert ids.shape == (n, TOTAL_LEN) and mask.shape == (n, TOTAL_LEN)
        recs = [json.loads(l) for l in
                open(OUT / f"trace_{s}.jsonl", encoding="utf-8")]
        assert len(recs) == n, f"{s}: npz/jsonl row mismatch"
        keys = [(str(a), str(b)) for a, b in
                zip(z["var_keys"], z["sample_ids"])]
        jkeys = [(r["var_key"], r["sample_id"]) for r in recs]
        assert keys == jkeys, f"{s}: npz vs jsonl key order mismatch"
        if keys_ref is None:
            keys_ref = keys
        else:
            assert keys == keys_ref, f"{s}: key order differs across schemes"

        c = np.array([r["content_tokens"] for r in recs])
        sep_pos = c + 1
        bad_cls = int((ids[:, 0] != CLS_ID).sum())
        bad_sep = int((ids[np.arange(n), sep_pos] != SEP_ID).sum())
        exp_mask = (np.arange(TOTAL_LEN)[None, :] < (c + 2)[:, None]).astype(
            mask.dtype)
        bad_mask = int((mask != exp_mask).sum())
        bad_pad = 0
        for i in range(n):
            if (ids[i, c[i] + 2:] != PAD_ID).any():
                bad_pad += 1
        # content region must not overrun and SEP not inside content
        over = int((c + 2 > TOTAL_LEN).sum())
        # per-record start/end spans must tile content contiguously
        span_bad = 0
        for r, ci in zip(recs, c):
            pos = 1
            bad = False
            for name, b in r["blocks"].items():
                if b["skipped"]:
                    if b["alloc"] != 0:
                        bad = True
                    continue
                if b["start"] != pos or b["end"] != pos + b["alloc"]:
                    bad = True
                    break
                pos = b["end"]
            if bad or pos != ci + 1:
                span_bad += 1
        print(f"[{s}] n={n} bad_cls={bad_cls} bad_sep={bad_sep} "
              f"bad_mask={bad_mask} bad_pad={bad_pad} over={over} "
              f"span_bad={span_bad}")
        ok &= not (bad_cls or bad_sep or bad_mask or bad_pad or over
                   or span_bad)

        if s == "D0":
            fz = np.load(FROZEN_NPZ, allow_pickle=False)
            fz_keys = [(str(v), str(i)) for v, i in
                       zip(fz["var_keys"], fz["sample_ids"])]
            pos_of = {k: i for i, k in enumerate(keys)}
            mism = 0
            for row, k in enumerate(fz_keys):
                i = pos_of[k]
                if not (np.array_equal(ids[i], fz["ids"][row])
                        and np.array_equal(mask[i], fz["mask"][row])):
                    mism += 1
            print(f"[D0] bitwise vs frozen: mismatches {mism}/{len(fz_keys)}")
            ok &= mism == 0
            test_keys = set(fz_keys)

    n_test = sum(1 for k in keys_ref if k in test_keys)
    print(f"[keys] total={len(keys_ref)} unique={len(set(keys_ref))} "
          f"test={n_test}")
    ok &= len(keys_ref) == 8106 and len(set(keys_ref)) == 8106
    ok &= n_test == 2474

    print("[verify] ALL OK" if ok else "[verify] FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
