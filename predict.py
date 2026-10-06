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
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent
PREDICTIONS = PROJECT / "predictions"

KEY_CODE_FILES = ["load_aurora.py", "data_loader.py", "predict.py", "aggregate.py"]


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

    def reuse_or_init(self, method, domain, base_seed, fingerprint):
        """Return True if a compatible completed shard exists; otherwise quarantine
        any incompatible leftovers and start a fresh buffer."""
        npz, header, done = self._paths(method, domain, base_seed)
        if npz.exists():
            hp = header if header.exists() else None
            old = None
            if hp is not None:
                try:
                    old = json.loads(hp.read_text())
                except Exception:
                    old = None
            if old is not None and old.get("fingerprint") == fingerprint and done.exists():
                return True
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
        header.write_text(json.dumps({"fingerprint": fingerprint,
                                      "n_rows": len(rows),
                                      "shard": npz.name}, indent=2))
        done.write_text(json.dumps({"fingerprint": fingerprint}, indent=2))

    def load(self, method, domain, base_seed):
        npz, _, _ = self._paths(method, domain, base_seed)
        return np.load(npz, allow_pickle=False)
