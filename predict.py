"""Per-window Aurora inference + fingerprint-bound sharded output.

Randomness: the only stochastic source inside generate() is the flow-matching
initial gaussian noise (single torch.randn). torch.manual_seed(w) right before
each generate() makes the run reproducible; w is derived from the composite
window key (var_key, sample_id) so resume reproduces identical numbers.
"""

import hashlib
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from load_aurora import sha256_file

PROJECT = Path(__file__).resolve().parent
PREDICTIONS = PROJECT / "predictions"

KEY_CODE_FILES = ["load_aurora.py", "data_loader.py", "predict.py", "aggregate.py"]


class ShardVerificationError(RuntimeError):
    pass


def build_expectation(rows):
    """rows: [(var_key, sample_id, domain, past, target, d)] ->
    {(var_key, sample_id): (target float64 (12,), d float)} over any window scope
    (full 2474 test set, or a preflight drill subset)."""
    out = {}
    for vk, sid, _dom, _past, tgt, dv in rows:
        out[(vk, sid)] = (np.asarray(tgt, dtype=np.float64), float(dv))
    return out


def verify_shard_coverage(entries, expect):
    """OFFICIAL coverage validation (run_eval final check and preflight L).

    entries: iterable of (var_key, sample_id, target(12,), d).
    expect:  {(var_key, sample_id): (target, d)} from the frozen data.

    Hard-fails unless ALL hold:
      1. row count == len(expect)
      2. keys unique (catches duplicate-one-window / missing-one-window combos
         that preserve row count)
      3. key set == expect key set exactly
      4. every target and d bitwise equal to the frozen expectation
    Returns report dict; raises ShardVerificationError on any failure.
    """
    entries = list(entries)
    if len(entries) != len(expect):
        raise ShardVerificationError(
            f"coverage: {len(entries)} rows != expected {len(expect)}")
    keys = [(vk, sid) for vk, sid, *_ in entries]
    dup_counts = Counter(keys)
    dups = sorted(k for k, c in dup_counts.items() if c > 1)
    if dups:
        raise ShardVerificationError(f"coverage: duplicate keys: {dups[:5]}")
    if set(keys) != set(expect):
        missing = sorted(set(expect) - set(keys))[:5]
        extra = sorted(set(keys) - set(expect))[:5]
        raise ShardVerificationError(
            f"coverage: key set mismatch missing={missing} extra={extra}")
    mism = []
    for vk, sid, tgt, dv in entries:
        et, ed = expect[(vk, sid)]
        t = np.asarray(tgt, dtype=np.float64)
        if t.shape != (12,) or not np.array_equal(t, et) or float(dv) != ed:
            mism.append((vk, sid))
    if mism:
        raise ShardVerificationError(f"coverage: target/d mismatch at {mism[:5]}")
    return {"ok": True, "n_rows": len(entries), "n_unique_keys": len(set(keys)),
            "target_d_bitwise": True}


def code_md5():
    out = {}
    for name in KEY_CODE_FILES:
        p = PROJECT / name
        h = hashlib.md5()
        h.update(p.read_bytes())
        out[name] = h.hexdigest()
    return out


def aurora_pkg_md5():
    from load_aurora import verify_aurora_package
    rep = verify_aurora_package()
    return {k: v["got"] for k, v in rep["files"].items()}


def shard_fingerprint(protocol, itl, base_seed, weights_sha):
    return {
        "protocol_sha256": hashlib.sha256((PROJECT / "protocol.json").read_bytes()).hexdigest(),
        "split_manifest_sha256": protocol["data"]["split_manifest_sha256"],
        "data_cache_sha256": protocol["data"]["data_cache_sha256"],
        "weights_sha256": weights_sha,
        "hf_revision": protocol["model"]["hf_revision"],
        "itl": itl,
        "num_samples": protocol["inference"]["num_samples"],
        "base_seed": base_seed,
        "code_md5": code_md5(),
        "aurora_package_md5": aurora_pkg_md5(),
    }


def derive_seed(base_seed, var_key, sample_id):
    token = f"aurora-timesx-v1|{base_seed}|{var_key}|{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % (2 ** 31 - 1)


def predict_window(model, past, itl, num_samples, seed):
    """past: float64 (96,) -> float64 (12,) point forecast (sample-mean)."""
    x = torch.from_numpy(np.asarray(past, dtype=np.float32)).unsqueeze(0)
    x = x.to(next(model.parameters()).device)
    torch.manual_seed(int(seed))
    with torch.inference_mode():
        out = model.generate(
            inputs=x,
            text_inputs=None,
            vision_inputs=None,
            revin=True,
            num_samples=num_samples,
            max_output_length=12,
            inference_token_len=itl,
        )
    pred = out.to(torch.float64).mean(dim=1).squeeze(0).cpu().numpy()
    return pred


def shard_name(method, domain, base_seed):
    return f"{method}__{domain}__s{base_seed}__test.npz"


class ShardStore:
    """Fingerprint-bound sharded npz writer with verified resume + quarantine."""

    def __init__(self, out_dir=None):
        self.out_dir = Path(out_dir) if out_dir else PREDICTIONS
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir = self.out_dir / "quarantine"
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

    def _paths(self, method, domain, base_seed):
        base = self.out_dir / shard_name(method, domain, base_seed)
        return base, Path(str(base) + ".header.json"), Path(str(base) + ".done")

    def reuse_or_init(self, method, domain, base_seed, fingerprint, expect=None):
        """Decide whether an existing shard may be reused (skipping inference).

        Reuse requires ALL of:
          - header and .done exist, parse, and their fingerprints match the
            current fingerprint exactly
          - sha256(npz) matches the content hash recorded in BOTH header and .done
          - npz loads; row count matches header; all predictions finite
          - if `expect` (domain-scope {(vk,sid): (target,d)}) is given, the shard's
            entries pass verify_shard_coverage against it (keys + bitwise target/d)

        Any failure -> shard moved to quarantine/ and False returned.
        """
        npz, header, done = self._paths(method, domain, base_seed)
        if not npz.exists():
            return False
        try:
            hdr = json.loads(header.read_text()) if header.exists() else None
            dn = json.loads(done.read_text()) if done.exists() else None
            reasons = []
            if not isinstance(hdr, dict):
                reasons.append("header missing/unreadable")
            if not isinstance(dn, dict):
                reasons.append(".done missing/unreadable")
            if isinstance(hdr, dict) and hdr.get("fingerprint") != fingerprint:
                reasons.append("header fingerprint mismatch")
            if isinstance(dn, dict) and dn.get("fingerprint") != fingerprint:
                reasons.append(".done fingerprint mismatch")
            content = sha256_file(npz)
            if isinstance(hdr, dict) and hdr.get("content_sha256") != content:
                reasons.append("npz content hash != header record")
            if isinstance(dn, dict) and dn.get("content_sha256") != content:
                reasons.append("npz content hash != .done record")
            if reasons:
                raise ShardVerificationError("; ".join(reasons))

            with np.load(npz, allow_pickle=False) as z:
                n = len(z["var_keys"])
                if n != hdr.get("n_rows"):
                    raise ShardVerificationError("row count != header n_rows")
                if not np.isfinite(z["pred"]).all():
                    raise ShardVerificationError("non-finite values in stored pred")
                if expect is not None:
                    entries = [(str(z["var_keys"][i]), str(z["sample_ids"][i]),
                                z["target"][i], float(z["d"][i])) for i in range(n)]
                    verify_shard_coverage(entries, expect)
            return True
        except Exception as e:
            print(f"[shard] reject {npz.name}: {e}")
            self.quarantine(method, domain, base_seed)
            return False

    def quarantine(self, method, domain, base_seed):
        npz, header, done = self._paths(method, domain, base_seed)
        tag = time.strftime("%Y%m%d_%H%M%S")
        for p in (npz, header, done):
            if p.exists():
                dst = self.quarantine_dir / f"{tag}__{p.name}"
                shutil.move(str(p), dst)

    def save(self, method, domain, base_seed, fingerprint, rows):
        """rows: list of dicts with var_key, sample_id, domain, pred(12), target(12), d.
        Atomic npz + header sidecar + .done marker."""
        npz, header, done = self._paths(method, domain, base_seed)
        tmp = Path(str(npz) + ".tmp")
        with open(tmp, "wb") as f:  # file handle: np.savez must not append .npz to tmp
            np.savez(
                f,
                sample_ids=np.array([r["sample_id"] for r in rows], dtype=np.str_),
                var_keys=np.array([r["var_key"] for r in rows], dtype=np.str_),
                domains=np.array([r["domain"] for r in rows], dtype=np.str_),
                pred=np.array([r["pred"] for r in rows], dtype=np.float64),
                target=np.array([r["target"] for r in rows], dtype=np.float64),
                d=np.array([r["d"] for r in rows], dtype=np.float64),
                method=np.str_(method),
                seed=np.int64(base_seed),
            )
        os.replace(tmp, npz)  # same filesystem -> atomic
        content_sha = sha256_file(npz)
        header.write_text(json.dumps({"fingerprint": fingerprint,
                                      "n_rows": len(rows),
                                      "content_sha256": content_sha,
                                      "shard": npz.name}, indent=2))
        done.write_text(json.dumps({"fingerprint": fingerprint,
                                    "content_sha256": content_sha}, indent=2))

    def entries(self, method, domain, base_seed):
        """(var_key, sample_id, target, d) rows of a stored shard."""
        npz, _, _ = self._paths(method, domain, base_seed)
        with np.load(npz, allow_pickle=False) as z:
            return [(str(z["var_keys"][i]), str(z["sample_ids"][i]),
                     z["target"][i], float(z["d"][i]))
                    for i in range(len(z["var_keys"]))]

    def load(self, method, domain, base_seed):
        npz, _, _ = self._paths(method, domain, base_seed)
        return np.load(npz, allow_pickle=False)
