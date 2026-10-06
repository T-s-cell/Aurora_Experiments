#!/usr/bin/env python3
"""M48T512 per-window inference + fingerprint-bound shards (subdir-local).

Reuses the root frozen machinery (derive_seed, ShardStore, verify_shard_coverage)
verbatim; only the fingerprint and the generate() call differ (explicit text
tensors). The text=None path calls generate() with all-text kwargs None — the
preflight (MM-A) checks it is bitwise identical to the root predict_window.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

SUBDIR = Path(__file__).resolve().parent
PROJECT = SUBDIR.parent
sys.path.insert(0, str(PROJECT))

from predict import (ShardStore, ShardVerificationError, aurora_pkg_md5,  # noqa: E402,F401
                     build_expectation, code_md5 as root_code_md5, derive_seed,
                     verify_shard_coverage)

MM_CODE_FILES = ["text_builder.py", "text_cache.py", "predict_mm.py",
                 "run_eval_mm.py", "aggregate_mm.py"]
BERT_CONFIG_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt"]
CACHE_NPZ = SUBDIR / "cache" / "text_tokens_M48T512.npz"


def protocol_mm_sha():
    return hashlib.sha256((SUBDIR / "protocol_mm.json").read_bytes()).hexdigest()


def mm_code_md5():
    out = {}
    for name in MM_CODE_FILES:
        h = hashlib.md5()
        h.update((SUBDIR / name).read_bytes())
        out[name] = h.hexdigest()
    return out


def bert_config_md5():
    import importlib.util
    from text_builder import find_bert_config
    base = Path(find_bert_config())
    out = {}
    for name in BERT_CONFIG_FILES:
        h = hashlib.md5()
        h.update((base / name).read_bytes())
        out[name] = h.hexdigest()
    return out


def shard_fingerprint_mm(protocol, itl, base_seed, weights_sha):
    return {
        "protocol_mm_sha256": protocol_mm_sha(),
        "split_manifest_sha256": protocol["data"]["split_manifest_sha256"],
        "data_cache_sha256": protocol["data"]["data_cache_sha256"],
        "weights_sha256": weights_sha,
        "hf_revision": protocol["model"]["hf_revision"],
        "itl": itl,
        "num_samples": protocol["inference"]["num_samples"],
        "base_seed": base_seed,
        "text_cache_npz_sha256": protocol["text"]["cache"]["npz_sha256"],
        "code_md5_root": root_code_md5(),
        "code_md5_mm": mm_code_md5(),
        "bert_config_md5": bert_config_md5(),
        "aurora_package_md5": aurora_pkg_md5(),
    }


class TextTokenStore:
    """Loads the frozen text cache; lookup STRICTLY by (var_key, sample_id)."""

    def __init__(self, protocol=None):
        protocol = protocol or __import__("json").loads(
            (SUBDIR / "protocol_mm.json").read_text())
        got = hashlib.sha256(CACHE_NPZ.read_bytes()).hexdigest()
        expect = protocol["text"]["cache"]["npz_sha256"]
        if got != expect:
            raise ShardVerificationError(
                f"text cache sha256 mismatch: got {got} != frozen {expect}")
        z = np.load(CACHE_NPZ, allow_pickle=False)
        self._keys = list(zip([str(x) for x in z["var_keys"]],
                              [str(x) for x in z["sample_ids"]]))
        if len(set(self._keys)) != len(self._keys):
            raise ShardVerificationError("duplicate composite keys in text cache")
        self._ids = z["ids"]
        self._mask = z["mask"]
        self._index = {k: i for i, k in enumerate(self._keys)}

    def keys(self):
        return set(self._keys)

    def get(self, var_key, sample_id, device):
        i = self._index.get((var_key, sample_id))
        if i is None:
            raise KeyError(f"window missing from text cache: {(var_key, sample_id)}")
        ids = torch.from_numpy(self._ids[i].astype(np.int64)).unsqueeze(0).to(device)
        mask = torch.from_numpy(self._mask[i].astype(np.int64)).unsqueeze(0).to(device)
        ttids = torch.zeros_like(ids)
        return ids, mask, ttids


def predict_window_mm(model, past, itl, num_samples, seed,
                      text_ids=None, text_mask=None, text_typeids=None):
    """past: float64 (96,) -> float64 (12,) point forecast (sample-mean).

    Mirrors root predict_window; text tensors (already on model.device) are
    passed explicitly. With all-None text this is the EXP-010 path (MM-A checks
    bitwise equality against the root function)."""
    x = torch.from_numpy(np.asarray(past, dtype=np.float32)).unsqueeze(0)
    x = x.to(next(model.parameters()).device)
    torch.manual_seed(int(seed))
    with torch.inference_mode():
        out = model.generate(
            inputs=x,
            text_inputs=None,
            text_input_ids=text_ids,
            text_attention_mask=text_mask,
            text_token_type_ids=text_typeids,
            vision_inputs=None,
            revin=True,
            num_samples=num_samples,
            max_output_length=12,
            inference_token_len=itl,
        )
    pred = out.to(torch.float64).mean(dim=1).squeeze(0).cpu().numpy()
    return pred
