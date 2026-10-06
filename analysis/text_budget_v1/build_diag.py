#!/usr/bin/env python3
"""Build D0/D1/D2 inputs for all 8,106 samples + the frozen 2,474 test windows
(analysis-text-budget-v1). Statistics-only: no Aurora model, no prediction.

Hard gate: D0 on the 2,474 test windows must be bitwise identical to the
frozen EXP-011 cache (ids + mask).

Per-window trace JSONL keeps (user correction 2026-10-07): the cleaned
UNTRUNCATED field texts and deletion records (D2), and the final input as
token IDs (npz) + decoded text — so rule-deletion and budget-truncation are
distinguishable.
"""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "repro_mm_timesx_v1"))

import numpy as np  # noqa: E402

from allocate import (BLOCKS, CONTENT_BUDGET, TOTAL_LEN, alloc_d0, alloc_d1,  # noqa: E402
                      assemble, block_token_counts)
from clean_rules import _R1, _R4, clean_fields, normalize  # noqa: E402
from text_builder import find_bert_config  # noqa: E402
from text_cache import ZIP_PATH, build_index  # noqa: E402

FROZEN_NPZ = PROJECT / "repro_mm_timesx_v1" / "cache" / "text_tokens_M48T512.npz"
OUT = HERE / "outputs"
RESULTS = HERE / "results"
SCHEMES = ("D0", "D1", "D2")

CLEANED_SCEN_HEAD = re.compile(
    r"^Prediction period: \d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}\. ")
CLEANED_COV_HEAD = re.compile(r"^\d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}: ")
TAG_RE = re.compile(r"<(\d+)>")


def md5_file(p):
    h = hashlib.md5()
    h.update(Path(p).read_bytes())
    return h.hexdigest()


def load_split_scope():
    """Frozen split assignment from data/split_manifest.json. native splits
    are plain sid lists for train/val/test but {sample_id, reason} dicts for
    excluded. Returns {(var_key, sample_id): (split, excluded_reason)}."""
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


def tokenizer_provenance():
    from text_builder import BLOCKS as TB_BLOCKS  # sanity: frozen builder importable
    tok_dir = find_bert_config()
    files = {"config": "config.json", "tokenizer_json": "tokenizer.json",
             "tokenizer_config": "tokenizer_config.json", "vocab": "vocab.txt"}
    local, missing = {}, []
    for k, fn in files.items():
        p = Path(tok_dir) / fn
        if p.exists():
            local[k] = md5_file(p)
        else:
            missing.append(fn)
    proto = json.loads((PROJECT / "repro_mm_timesx_v1" / "protocol_mm.json").read_text())
    pinned = proto["text"]["tokenizer"]
    pin_map = {}
    for k, v in pinned.items():
        if k.endswith("_md5") and isinstance(v, str):
            pin_map[k[:-4]] = v
    # every pinned file must exist locally AND match; missing file => fail
    match = {k: (local.get(k) == v) for k, v in pin_map.items()}
    all_matched = all(match.values()) and not missing
    import transformers
    return {
        "transformers_version": transformers.__version__,
        "tokenizer_class": "BertTokenizer (slow, identical to EXP-011)",
        "bert_config_dir": str(tok_dir),
        "local_md5": local,
        "local_missing_files": missing,
        "protocol_mm_pinned_md5": pin_map,
        "md5_match_protocol_mm": match,
        "all_matched": all_matched,
        "zip_sha256": hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest(),
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "head_commit": __import__("subprocess").run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True,
            text=True).stdout.strip(),
    }


def split_events(field_text, head_re):
    """Split 'Name: ' + field into [head][p0][tag1][prose1]... segments; returns
    dict(head, segments, events=[(tag_piece, prose_piece)], ok). ok=False on any
    structural surprise (p0 non-empty, tags not 1..m strictly increasing)."""
    m = head_re.match(field_text)
    if not m:
        return None
    head = m.group(0)
    body = field_text[m.end():]
    pieces = re.split(r"(<\d+>)", body)
    ok = pieces[0].strip() == ""
    events, seq_no, i = [], [], 1
    while i < len(pieces):
        tm = TAG_RE.fullmatch(pieces[i])
        if not tm:
            ok = False
            break
        seq_no.append(int(tm.group(1)))
        prose = pieces[i + 1] if i + 1 < len(pieces) else ""
        events.append((pieces[i], prose))
        i += 2
    ok = ok and seq_no == list(range(1, len(seq_no) + 1))
    return {"head": head, "segments": [head] + pieces, "events": events, "ok": ok}


def split_cov_entries(field_text, head_re):
    m = head_re.match(field_text)
    if not m:
        return None
    head = m.group(0)
    rest = field_text[m.end():]
    entries = re.split(r"(?=; )", rest)
    return {"head": head, "segments": [head] + entries, "n_entries": len(entries)}


def coverage_from_parts(parts_lens, block_meta, unit_lens):
    """Generic coverage of consecutive units inside a kept block span.
    parts_lens: token lengths of [head, seg...] (concat == block kept-full ids);
    unit_lens: list of (first_part_idx, n_parts) per unit; block start/end in
    512-coords. Returns per-unit (n_tokens_in_kept, n_tokens_total)."""
    start, end = block_meta["start"], block_meta["end"]
    offs, o = [], 0
    for L in parts_lens:
        offs.append(o)
        o += L
    out = []
    for first, n in unit_lens:
        u_start = start + offs[first]
        u_end = start + offs[first + n - 1] + parts_lens[first + n - 1]
        total = u_end - u_start
        in_kept = max(0, min(u_end, end) - max(u_start, start))
        out.append((in_kept, total))
    return out


def encode_parts(tokenizer, parts):
    return [tokenizer.encode(s, add_special_tokens=False) for s in parts]


def build_window(texts, tokenizer):
    """Build all three schemes for one window. `texts` = raw four fields."""
    norm = {f: normalize(texts[f]) for n, f, _b in BLOCKS}
    counts_raw = block_token_counts(norm, tokenizer)
    cleaned, drec = clean_fields(norm["background"], norm["scenario"],
                                 norm["holiday_info"], norm["covariates_info"])
    counts_cln = block_token_counts(cleaned, tokenizer)

    allocs = {"D0": alloc_d0(counts_raw), "D1": alloc_d1(counts_raw),
              "D2": alloc_d1(counts_cln)}
    counts = {"D0": counts_raw, "D1": counts_raw, "D2": counts_cln}
    texts_by = {"D0": norm, "D1": norm, "D2": cleaned}

    seqs, masks, metas = {}, {}, {}
    for s in SCHEMES:
        seqs[s], masks[s], metas[s] = assemble(counts[s], allocs[s], tokenizer)

    # --- Events coverage (D0/D1 on raw scenario; D2 on cleaned scenario) ---
    ev = {}
    for s in SCHEMES:
        field = texts_by[s]["scenario"]
        head_re = _R1 if s in ("D0", "D1") else CLEANED_SCEN_HEAD
        sp = split_events(field, head_re)
        info = {"ok": False, "n_total": 0, "n_complete": 0, "n_partial": 0,
                "n_absent": 0}
        if sp is not None and sp["ok"] and not counts[s]["Events"]["skipped"]:
            # whole = encode("Events: " + field); fold the name into the head
            # segment so concat-of-parts is comparable to the whole-block ids
            parts = encode_parts(tokenizer,
                                 ["Events: " + sp["segments"][0]]
                                 + sp["segments"][1:])
            whole = counts[s]["Events"]["raw_ids"]
            concat = [i for p in parts for i in p]
            if concat == whole:
                lens = [len(p) for p in parts]
                # event k occupies parts[2k] (tag) + parts[2k+1] (prose), k>=1
                units = [(2 * k, 2) for k in range(1, len(sp["events"]) + 1)]
                cov = coverage_from_parts(lens, metas[s]["Events"], units)
                info.update(ok=True, n_total=len(cov),
                            n_complete=sum(1 for a, b in cov if a == b),
                            n_partial=sum(1 for a, b in cov if 0 < a < b),
                            n_absent=sum(1 for a, b in cov if a == 0))
        ev[s] = info

    # --- Covariates entry coverage ---
    cv = {}
    for s in SCHEMES:
        field = texts_by[s]["covariates_info"]
        head_re = _R4 if s in ("D0", "D1") else CLEANED_COV_HEAD
        sp = split_cov_entries(field, head_re)
        info = {"ok": False, "n_entries": 0, "n_complete": 0,
                "has_partial_entry": False}
        if sp is not None and not counts[s]["Covariates"]["skipped"]:
            parts = encode_parts(tokenizer,
                                 ["Covariates: " + sp["segments"][0]]
                                 + sp["segments"][1:])
            whole = counts[s]["Covariates"]["raw_ids"]
            concat = [i for p in parts for i in p]
            if concat == whole:
                lens = [len(p) for p in parts]
                units = [(j + 1, 1) for j in range(sp["n_entries"])]
                cov = coverage_from_parts(lens, metas[s]["Covariates"], units)
                info.update(ok=True, n_entries=sp["n_entries"],
                            n_complete=sum(1 for a, b in cov if a == b),
                            has_partial_entry=any(0 < a < b for a, b in cov))
        cv[s] = info

    rows = {}
    for s in SCHEMES:
        content = sum(allocs[s].values())
        blk = {}
        for name, _f, _b in BLOCKS:
            m_ = metas[s][name]
            # raw_tokens is ALWAYS the raw (pre-cleanup) count, for every
            # scheme, so D2 shortening = 1 - cleaned/raw is well-defined;
            # cleaned_tokens (D2 only) is the post-cleanup count.
            e = {"skipped": m_["skipped"],
                 "raw_tokens": counts_raw[name]["raw_tokens"],
                 "alloc": allocs[s][name],
                 "start": m_["start"], "end": m_["end"]}
            if s == "D2":
                e["cleaned_tokens"] = counts_cln[name]["raw_tokens"]
            blk[name] = e
        ids = np.array(seqs[s], dtype=np.int32)
        rows[s] = {
            "ids": ids, "mask": np.array(masks[s], dtype=np.int32),
            "blocks": blk, "content_tokens": content,
            "idle_tokens": TOTAL_LEN - 2 - content,
            "still_truncated": any(blk[n]["alloc"] < blk[n][
                "cleaned_tokens" if s == "D2" else "raw_tokens"]
                for n, _f, _b in BLOCKS),
            "events": ev[s], "covariates": cv[s],
            "decoded_input": tokenizer.decode(
                ids[1:1 + content].tolist(), skip_special_tokens=False),
        }
    return rows, cleaned, drec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke-test: first N keys")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    prov = tokenizer_provenance()
    print("[prov] tokenizer md5 match protocol_mm:",
          prov["md5_match_protocol_mm"], "| missing:",
          prov["local_missing_files"], "| all_matched:", prov["all_matched"])
    assert prov["all_matched"], (
        f"tokenizer files differ from frozen EXP-011 "
        f"(mismatches: {[k for k, v in prov['md5_match_protocol_mm'].items() if not v]}, "
        f"missing: {prov['local_missing_files']})")
    (RESULTS / "provenance.json").write_text(json.dumps(prov, indent=2))

    from transformers import BertTokenizer
    tok = BertTokenizer.from_pretrained(find_bert_config(), local_files_only=True)

    index = build_index()
    keys_all = sorted(index.keys())
    if args.limit:
        keys_all = keys_all[: args.limit]
    print(f"[build] {len(keys_all)} windows (full index {len(index)})")

    fz = np.load(FROZEN_NPZ, allow_pickle=False)
    fz_keys = [(str(v), str(s)) for v, s in zip(fz["var_keys"], fz["sample_ids"])]
    fz_set = set(fz_keys)
    built_test = [k for k in keys_all if k in fz_set]
    print(f"[build] test windows in scope: {len(built_test)} / {len(fz_set)}")

    scope_map = load_split_scope()
    assert set(scope_map) == set(keys_all), (
        f"manifest/index key mismatch: manifest {len(scope_map)} vs index "
        f"{len(keys_all)}; only_index={len(set(keys_all) - set(scope_map))} "
        f"only_manifest={len(set(scope_map) - set(keys_all))}")
    assert {k for k, (sp, _r) in scope_map.items() if sp == "test"} == fz_set, \
        "manifest test split != frozen cache keys"
    from collections import Counter
    print("[build] splits:", dict(Counter(sp for sp, _r in scope_map.values())))

    data = {s: {"ids": [], "mask": [], "rec": []} for s in SCHEMES}
    for i, key in enumerate(keys_all):
        vk, sid = key
        rec = index[key]
        rows, cleaned, drec = build_window(rec["fields"], tok)
        scope, ex_reason = scope_map[key]
        for s in SCHEMES:
            r = rows[s]
            data[s]["ids"].append(r["ids"])
            data[s]["mask"].append(r["mask"])
            jr = {"var_key": vk, "sample_id": sid, "scope": scope,
                  "domain": vk.split("__", 1)[0],
                  "freq": rec["freq"], "blocks": r["blocks"],
                  "content_tokens": r["content_tokens"],
                  "idle_tokens": r["idle_tokens"],
                  "still_truncated": r["still_truncated"],
                  "events": r["events"], "covariates": r["covariates"],
                  "decoded_input": r["decoded_input"]}
            if s == "D2":
                jr["cleaned_text"] = cleaned
                jr["deletions"] = drec
            if ex_reason:
                jr["excluded_reason"] = ex_reason
            data[s]["rec"].append(jr)
        if (i + 1) % 1000 == 0:
            print(f"[build] {i + 1}/{len(keys_all)}")

    # hard gate on test windows (only meaningful without --limit)
    gate = {"n_test_built": len(built_test), "d0_bitwise_mismatch": None,
            "domain_mismatch": None}
    if not args.limit:
        pos = {k: i for i, k in enumerate(keys_all)}
        mism = 0
        dom_mism = 0
        for row, key in enumerate(fz_keys):
            i = pos[key]
            if not (np.array_equal(data["D0"]["ids"][i], fz["ids"][row])
                    and np.array_equal(data["D0"]["mask"][i], fz["mask"][row])):
                mism += 1
            if data["D0"]["rec"][i]["domain"] != str(fz["domains"][row]):
                dom_mism += 1
        gate["d0_bitwise_mismatch"] = mism
        gate["domain_mismatch"] = dom_mism
        print(f"[gate] D0 vs frozen cache bitwise mismatches: {mism} "
              f"/ {len(fz_keys)}; domain mismatches: {dom_mism}")
        assert mism == 0, "D0 reimplementation diverges from frozen EXP-011 cache"
        assert dom_mism == 0, "domain derivation diverges from frozen cache"

    for s in SCHEMES:
        np.savez(OUT / f"trace_{s}.npz",
                 ids=np.stack(data[s]["ids"]).astype(np.int32),
                 mask=np.stack(data[s]["mask"]).astype(np.int32),
                 var_keys=np.array([r["var_key"] for r in data[s]["rec"]]),
                 sample_ids=np.array([r["sample_id"] for r in data[s]["rec"]]))
        with open(OUT / f"trace_{s}.jsonl", "w", encoding="utf-8") as f:
            for jr in data[s]["rec"]:
                f.write(json.dumps(jr, ensure_ascii=False) + "\n")
        print(f"[out] trace_{s}: {len(data[s]['rec'])} windows")

    if args.limit:
        for s in SCHEMES:
            (OUT / f"trace_{s}.npz").unlink()
            (OUT / f"trace_{s}.jsonl").unlink()
        print("[smoke] outputs removed (limit run)")
    print("[build] DONE")


if __name__ == "__main__":
    main()
