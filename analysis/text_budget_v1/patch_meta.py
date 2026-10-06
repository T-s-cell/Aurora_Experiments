#!/usr/bin/env python3
"""One-shot migration of the 2026-10-07 run-1 trace JSONLs to corrected
metadata (audit findings #1/#3); npz ids/mask are unaffected and untouched.

- D2 records: raw_tokens was written as the CLEANED count (shortening was 0
  everywhere). The true raw count is taken per key from the D0 trace (same
  tokenizer, same texts, alloc paths identical).
- all records: scope was 'trainval' for every non-test window; it is now the
  frozen split_manifest.json native split (train/val/test/excluded).

Serialization must be byte-identical to build_diag.py's writer
(json.dumps(..., ensure_ascii=False)) so a fresh build with the fixed code
reproduces these files exactly. Superseded for future runs by the fixed
build_diag.py; kept for the audit trail.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
OUT = HERE / "outputs"
SCHEMES = ("D0", "D1", "D2")


def load_split_scope():
    man = json.loads((PROJECT / "data" / "split_manifest.json").read_text())
    scope = {}
    for vk, v in man["variables"].items():
        for sp, entries in v["native"].items():
            for e in entries:
                if isinstance(e, dict):
                    sid, reason = str(e["sample_id"]), e.get("reason")
                else:
                    sid, reason = str(e), None
                key = (vk, sid)
                assert key not in scope, f"duplicate manifest key: {key}"
                scope[key] = (sp, reason)
    return scope


def main():
    scope_map = load_split_scope()
    print(f"[patch] manifest keys: {len(scope_map)}")

    def read(s):
        return [json.loads(l) for l in
                open(OUT / f"trace_{s}.jsonl", encoding="utf-8")]

    d0 = read("D0")
    d1 = read("D1")
    d2 = read("D2")
    for name, recs in (("D0", d0), ("D1", d1), ("D2", d2)):
        assert len(recs) == len(d0)
    d0_raw = {(r["var_key"], r["sample_id"]):
              {f: b["raw_tokens"] for f, b in r["blocks"].items()} for r in d0}

    n_scope, n_raw, n_reason = 0, 0, 0
    for recs in (d0, d1, d2):
        for r in recs:
            key = (r["var_key"], r["sample_id"])
            new_scope, reason = scope_map[key]
            if r["scope"] != new_scope:
                r["scope"] = new_scope
                n_scope += 1
            if reason and r.get("excluded_reason") != reason:
                # appended as the LAST key to match the fixed builder's order
                r["excluded_reason"] = reason
                n_reason += 1
    for r in d2:
        key = (r["var_key"], r["sample_id"])
        for f, b in r["blocks"].items():
            true_raw = d0_raw[key][f]
            if b["raw_tokens"] != true_raw:
                b["raw_tokens"] = true_raw
                n_raw += 1

    for s, recs in (("D0", d0), ("D1", d1), ("D2", d2)):
        tmp = OUT / f"trace_{s}.jsonl.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(OUT / f"trace_{s}.jsonl")
    print(f"[patch] scope fields rewritten: {n_scope} "
          f"(expect 5632x3 = {5632 * 3})")
    print(f"[patch] excluded_reason appended: {n_reason} "
          f"(expect 696x3 = {696 * 3})")
    print(f"[patch] D2 raw_tokens corrected: {n_raw} block entries "
          f"(expect ~{24 * 8106} minus equal-count blocks)")
    print("[patch] DONE")


if __name__ == "__main__":
    main()
