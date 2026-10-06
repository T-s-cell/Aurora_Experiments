#!/usr/bin/env python3
"""M48T512 fixed text-budget builder (frozen rules, method tag M48T512).

Security/independence rule: the constructor receives ONLY the four text-field
strings (background, scenario, holiday_info, covariates_info) — never the raw
sample dict — so future-window values are structurally unreachable here.

Frozen assembly (budgets INCLUDE the block name tokens):
  1. whitespace-normalize each field; skip a block if empty or literal Unknown
  2. per block, tokenize "Name: <field>" with the in-package bert_config
     BertTokenizer (add_special_tokens=False) and keep the FIRST <budget> tokens
  3. concatenate blocks in fixed order (no budget transfer between blocks;
     budgets sum to 48+270+64+128 = 510)
  4. [CLS] + content + [SEP], right-pad to 512 (pad id 0, attention mask 0)
"""
import os
import importlib.util
import hashlib

TOTAL_LEN = 512
BLOCKS = (
    ("Background", "background", 48),
    ("Events", "scenario", 270),
    ("Calendar", "holiday_info", 64),
    ("Covariates", "covariates_info", 128),
)
assert sum(b[2] for b in BLOCKS) == TOTAL_LEN - 2  # CLS + SEP

_SKIP_VALUES = {"", "unknown"}


def find_bert_config():
    spec = importlib.util.find_spec("aurora")
    if spec is None:
        raise FileNotFoundError("aurora package not importable; cannot locate bert_config")
    return os.path.join(os.path.dirname(spec.origin), "bert_config")


def normalize_field(text):
    return " ".join(str(text).split())


def build_tokens(texts, tokenizer):
    """texts: dict with the four field names. Returns (ids, mask, meta).

    meta carries per-block raw/kept token counts and [start, end) boundaries in
    the 512 sequence, the decoded text, and its sha256 — for audit trails.
    """
    content = []
    blocks_meta = []
    for name, field, budget in BLOCKS:
        raw = normalize_field(texts[field]) if field in texts else ""
        skip = raw.lower() in _SKIP_VALUES
        if skip:
            blocks_meta.append({"block": name, "field": field, "budget": budget,
                                "raw_chars": len(raw), "skipped": True,
                                "raw_tokens": 0, "kept_tokens": 0,
                                "start": None, "end": None})
            continue
        block_text = f"{name}: {raw}"
        raw_ids = tokenizer.encode(block_text, add_special_tokens=False)
        ids = raw_ids[:budget]
        blocks_meta.append({"block": name, "field": field, "budget": budget,
                            "raw_chars": len(raw), "skipped": False,
                            "raw_tokens": len(raw_ids),
                            "kept_tokens": len(ids),
                            "start": None, "end": None})
        content.append((name, ids))

    seq = [tokenizer.cls_token_id]
    for name, ids in content:
        start = len(seq)
        seq.extend(ids)
        for bm in blocks_meta:
            if bm["block"] == name:
                bm["start"], bm["end"] = start, len(seq)
    seq.append(tokenizer.sep_token_id)
    assert len(seq) <= TOTAL_LEN, f"content length {len(seq)} exceeds {TOTAL_LEN}"

    mask = [1] * len(seq) + [0] * (TOTAL_LEN - len(seq))
    ids_full = seq + [tokenizer.pad_token_id] * (TOTAL_LEN - len(seq))

    decoded = tokenizer.decode(seq, skip_special_tokens=False)
    meta = {
        "blocks": blocks_meta,
        "content_tokens": len(seq),
        "pad_tokens": TOTAL_LEN - len(seq),
        "decoded_text": decoded,
        "decoded_sha256": hashlib.sha256(decoded.encode("utf-8")).hexdigest(),
        "blocks_skipped": [bm["block"] for bm in blocks_meta if bm["skipped"]],
    }
    return ids_full, mask, meta
