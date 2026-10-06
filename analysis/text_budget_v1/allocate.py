#!/usr/bin/env python3
"""D1 budget allocator + shared assembly (analysis-text-budget-v1).

D1 = recover idle budget only. Field texts, block order, and missing-handling
are identical to D0 (EXP-011 frozen builder); only the per-block token
allocation changes. The assembly adds the block name uniformly — clean
functions never include it (user correction 2026-10-07).
"""
import json
from pathlib import Path

SUBDIR = Path(__file__).resolve().parent
CFG = json.loads((SUBDIR / "budget_config.json").read_text())
BLOCKS = [(b["name"], b["field"], b["budget"]) for b in CFG["blocks"]]
TOTAL_LEN = CFG["total_len"]
CONTENT_BUDGET = CFG["content_budget"]
SKIP_VALUES = set(CFG["missing_values_skip"])
_REDIST = list(CFG["d1"]["redistribute_order"])
_ORDER = [b[0] for b in BLOCKS]
assert sum(b[2] for b in BLOCKS) == CONTENT_BUDGET
assert set(_REDIST) == set(_ORDER)

_name_of_field = {f: n for n, f, _b in BLOCKS}


def block_token_counts(texts, tokenizer):
    """Tokenize 'Name: raw' per block exactly like D0 (add_special_tokens=False,
    no truncation). Returns per-block dict:
    {name: {skipped, raw_ids (list or None), raw_tokens}}."""
    out = {}
    for name, field, _b in BLOCKS:
        raw = texts[field]
        if raw is None or str(raw).strip().lower() in SKIP_VALUES:
            out[name] = {"skipped": True, "raw_ids": None, "raw_tokens": 0}
            continue
        ids = tokenizer.encode(f"{name}: {raw}", add_special_tokens=False)
        out[name] = {"skipped": False, "raw_ids": ids, "raw_tokens": len(ids)}
    return out


def alloc_d0(counts):
    return {n: (0 if c["skipped"] else min(c["raw_tokens"], b))
            for n, _f, b in BLOCKS
            for c in [counts[n]]}


def alloc_d1(counts):
    """Two-step allocation; asserts the D1>=D0 invariant and the total cap."""
    alloc = alloc_d0(counts)
    base = dict(alloc)
    leftover = CONTENT_BUDGET - sum(alloc.values())
    assert leftover >= 0
    for name in _REDIST:
        if leftover <= 0:
            break
        c = counts[name]
        if c["skipped"]:
            continue
        need = c["raw_tokens"] - alloc[name]
        if need <= 0:
            continue
        give = min(need, leftover)
        alloc[name] += give
        leftover -= give
    assert all(alloc[n] >= base[n] for n in alloc)
    assert sum(alloc.values()) <= CONTENT_BUDGET
    return alloc


def assemble(counts, alloc, tokenizer):
    """[CLS] + blocks in fixed order (each truncated to alloc) + [SEP], pad to
    512. Returns (ids, mask, blocks_meta) with per-block [start, end) spans and
    kept ids (needed downstream for event/entry coverage)."""
    cls_id, sep_id, pad_id = (tokenizer.cls_token_id, tokenizer.sep_token_id,
                              tokenizer.pad_token_id)
    seq, kept_ids, meta = [cls_id], {}, {}
    for name, _field, _b in BLOCKS:
        a = alloc[name]
        if a == 0:
            meta[name] = {"skipped": True, "kept_tokens": 0,
                          "start": None, "end": None, "kept_ids": []}
            continue
        ids = counts[name]["raw_ids"][:a]
        start = len(seq)
        seq.extend(ids)
        meta[name] = {"skipped": False, "kept_tokens": a,
                      "start": start, "end": len(seq), "kept_ids": ids}
    seq.append(sep_id)
    assert len(seq) <= TOTAL_LEN, f"assembly overflow: {len(seq)}"
    mask = [1] * len(seq) + [0] * (TOTAL_LEN - len(seq))
    seq = seq + [pad_id] * (TOTAL_LEN - len(seq))
    return seq, mask, meta
