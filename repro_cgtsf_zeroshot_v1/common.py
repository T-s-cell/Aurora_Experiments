#!/usr/bin/env python3
"""Shared plumbing: hashing, atomic JSON, in-memory model-state digest,
per-window seed derivation, shard ledger helpers. Protocol constants live
in protocol.py; anything the protocol pins (code/data/weights/package)
failing verification is a hard stop."""
import hashlib
import json
import os

import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def derive_seed(base_seed, dataset, idx, date):
    """Per-window seed (user-ruled form, frozen in protocol.SEED_RULE):
    int.from_bytes(sha256(key)[:8], 'big') % (2**31 - 1) with
    key = f'{SEED_PREFIX}|{base}|{dataset}|{Idx}|{Date}'."""
    import protocol as P
    key = f"{P.SEED_PREFIX}|{base_seed}|{dataset}|{idx}|{date}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],
                          "big") % P.SEED_MODULUS


def model_state_digest(model):
    """sha256 over sorted (name, raw bytes) of the full state_dict (params +
    buffers). Complements the on-disk weights hash: proves the actually-
    loaded in-memory model is unchanged across inference."""
    h = hashlib.sha256()
    sd = model.state_dict()
    for name in sorted(sd):
        t = sd[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(b"\0")
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def all_finite(arr):
    return bool(np.isfinite(arr).all())


def protocol_sha256():
    import protocol as P
    return sha256_file(P.PROTO_PATH) if os.path.isfile(P.PROTO_PATH) else None


def aurora_pkg_md5():
    import protocol as P
    d = P.aurora_pkg_dir()
    return {n: md5_file(os.path.join(d, n)) for n in sorted(P.AURORA_PKG_FILES)}


def core_guards():
    """Machine-independent fingerprints every prediction shard is bound to:
    protocol json, weights, CSVs, experiment code, old-root code, aurora
    package sources, visionts_ref manifest."""
    import protocol as P
    return {
        "protocol_sha256": protocol_sha256(),
        "weights_sha256": P.WEIGHTS_SHA256,
        "csv_sha256": dict(P.CSV_SHA256),
        "code_md5": {n: md5_file(os.path.join(P.HERE, n))
                     for n in P.CODE_FILES},
        "old_code_md5": {n: md5_file(os.path.join(P.REPO_ROOT, n))
                         for n in P.OLD_CODE_FILES},
        "aurora_pkg_md5": aurora_pkg_md5(),
        "vts_ref_manifest_sha256": sha256_file(P.VTS_REF_MANIFEST_PATH)
        if os.path.isfile(P.VTS_REF_MANIFEST_PATH) else None,
    }


def current_guards(batch=None, preflight_report_sha256=None):
    """Full guard set: core fingerprints + runtime binding (batch, env,
    preflight report). run_infer resume compares the FULL set on the same
    host (any drift = tier-1 hard stop); verify_cache compares the core set
    and cross-checks runtime.batch against the preflight report."""
    return {
        "core": core_guards(),
        "runtime": {
            "batch": batch,
            "env": env_info(),
            "preflight_report_sha256": preflight_report_sha256,
        },
    }


def guards_equal(a, b):
    return a == b


def env_info():
    import platform
    import sys
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "host": os.uname().nodename,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED", "")}
    import torch
    info["torch"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    try:
        from importlib.metadata import version
        info["aurora_model"] = version("aurora-model")
    except Exception:
        info["aurora_model"] = None
    import numpy
    info["numpy"] = numpy.__version__
    return info
