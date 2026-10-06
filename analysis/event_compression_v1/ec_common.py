#!/usr/bin/env python3
"""Shared helpers for analysis/event_compression_v1.

Reuses the frozen text_budget_v1 pipeline (clean_rules/allocate) and the
EXP-011 text_cache/text_builder via sys.path — nothing is copied or modified.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "analysis" / "text_budget_v1"))
sys.path.insert(0, str(PROJECT / "repro_mm_timesx_v1"))

import numpy as np  # noqa: E402

CONFIG = json.loads((HERE / "config.json").read_text())

TRACE_NPZ = PROJECT / "analysis" / "text_budget_v1" / "outputs" / "trace_D2.npz"
TRACE_JSONL = PROJECT / "analysis" / "text_budget_v1" / "outputs" / "trace_D2.jsonl"
OUT = HERE / "outputs"
RESULTS = HERE / "results"


def load_tokenizer():
    from text_builder import find_bert_config
    from transformers import BertTokenizer
    return BertTokenizer.from_pretrained(find_bert_config(), local_files_only=True)


def sha256_of(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def trace_npz_index():
    """(var_key, sample_id) -> row index into trace_D2.npz arrays."""
    z = np.load(TRACE_NPZ)
    return z, {(str(v), str(s)): i for i, (v, s) in
               enumerate(zip(z["var_keys"], z["sample_ids"]))}


def iter_trace_jsonl(scope_filter=None):
    """Stream trace_D2.jsonl records (optionally filtered by scope)."""
    with open(TRACE_JSONL) as f:
        for line in f:
            r = json.loads(line)
            if scope_filter is None or r.get("scope") in scope_filter:
                yield r


def manifest_scope_sets():
    """Frozen split assignment -> (train_val_keys, test_keys, excluded_keys)."""
    man = json.loads((PROJECT / "data" / "split_manifest.json").read_text())
    trv, test, exc = set(), set(), set()
    for vk, v in man["variables"].items():
        for sp, entries in v["native"].items():
            for e in entries:
                sid = str(e["sample_id"]) if isinstance(e, dict) else str(e)
                (trv if sp in ("train", "val") else test if sp == "test"
                 else exc).add((vk, sid))
    return trv, test, exc


def rebuild_d2(fields, tokenizer):
    """Frozen D2 pipeline on raw four fields -> dict with ids/mask/counts/
    alloc/cleaned and the event split of the cleaned scenario."""
    from allocate import (BLOCKS, alloc_d1, assemble, block_token_counts)
    from clean_rules import clean_fields, normalize

    norm = {f: normalize(fields[f]) for _n, f, _b in BLOCKS}
    cleaned, drec = clean_fields(norm["background"], norm["scenario"],
                                 norm["holiday_info"], norm["covariates_info"])
    counts = block_token_counts(cleaned, tokenizer)
    alloc = alloc_d1(counts)
    ids, mask, meta = assemble(counts, alloc, tokenizer)
    return {"norm": norm, "cleaned": cleaned, "deletions": drec,
            "counts": counts, "alloc": alloc, "ids": ids, "mask": mask,
            "meta": meta}


def verify_bitwise_vs_trace(ids, mask, vk, sid, z, idx):
    row = idx[(vk, sid)]
    return (np.array_equal(np.asarray(ids, dtype=np.int32), z["ids"][row])
            and np.array_equal(np.asarray(mask, dtype=np.int32), z["mask"][row]))


def split_events_clean(cleaned_scenario, tokenizer):
    """Split the D2-cleaned scenario into head + events using the frozen
    CLEANED_SCEN_HEAD regex (build_diag.py logic, re-imported)."""
    from build_diag import CLEANED_SCEN_HEAD
    from clean_rules import normalize
    from build_diag import split_events
    return split_events(cleaned_scenario, CLEANED_SCEN_HEAD)


def event_pieces(split, tokenizer):
    """Tokenize ['Events: '+head] + segments; returns (head_ids, [tag_ids_k],
    [prose_ids_k]). Segment layout after split_events: segments = [head, p0,
    tag1, prose1, ..., tagN, proseN] — p0 is the blank leading piece, so tags
    start at index 2 and proses at index 3 (build_diag units = (2k, 2) on the
    same layout)."""
    from build_diag import encode_parts
    parts = encode_parts(tokenizer,
                         ["Events: " + split["segments"][0]]
                         + split["segments"][1:])
    return parts[0], parts[2::2], parts[3::2]


def compute_event_budgets(split, head_ids, tag_ids, prose_ids, E):
    """Per-event prose budgets from original lengths (frozen rule).

    Returns dict with: pool, n, need[k], base[k], budget[k], fit[k],
    zero_budget[k], status ('ok' | 'events_absent' | 'parse_fail' |
    'overhead_overflow'). Deterministic; shared by both new schemes.
    """
    n = len(split["events"])
    if not split["ok"]:
        return {"status": "parse_fail", "n": n}
    if n == 0:
        return {"status": "events_absent", "n": 0}
    overhead = len(head_ids) + sum(len(t) for t in tag_ids)
    pool = E - overhead
    if pool < 0:
        return {"status": "overhead_overflow", "n": n, "overhead": overhead,
                "E": E}
    base = [pool // n + (1 if k < pool % n else 0) for k in range(n)]
    need = [len(p) for p in prose_ids]
    budget = list(base)
    slack = sum(max(0, budget[k] - need[k]) for k in range(n))
    for k in range(n):
        if slack <= 0:
            break
        d = need[k] - budget[k]
        if d > 0:
            give = min(d, slack)
            budget[k] += give
            slack -= give
    fit = [need[k] <= budget[k] for k in range(n)]
    zero = [(budget[k] <= 0 and need[k] > 0) for k in range(n)]
    return {"status": "ok", "n": n, "overhead": overhead, "E": E, "pool": pool,
            "need": need, "base": base, "budget": budget, "fit": fit,
            "zero_budget": zero}
