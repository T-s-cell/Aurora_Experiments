#!/usr/bin/env python3
"""Shared helpers for repro_timemmd_zeroshot_v1 (Aurora TimeMMD zero-shot).

official_metric_replica() intentionally mirrors TimeMMD/utils/metrics.py
(MAE/MSE on float32 numpy via np.mean default dtype) so the recompute gate
compares the SAME float32 computation the official entry printed. float64
recomputation is reported separately and carries no 1e-12 gate.
"""
import hashlib
import json
import os
import platform
import re
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OFFICIAL_TMM = os.path.join(HERE, "official", "TimeMMD")
DATASET_DIR = os.path.join(HERE, "dataset")
PRED_DIR = os.path.join(HERE, "predictions")
RESULTS_DIR = os.path.join(HERE, "results")
LOGS_DIR = os.path.join(HERE, "logs")
PROTO_PATH = os.path.join(HERE, "protocol.json")
BATCH_DECISION_PATH = os.path.join(RESULTS_DIR, "batch_decision.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)),
                               suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=False, ensure_ascii=False)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def env_info():
    info = {"python": platform.python_version(),
            "executable": sys.executable,
            "numpy": np.__version__}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu_name"] = p.name
            info["gpu_total_mem_gib"] = round(p.total_memory / (1 << 30), 1)
    except Exception as e:  # pragma: no cover - env reporting only
        info["torch_error"] = repr(e)
    for name in ("transformers", "pandas", "sklearn", "einops",
                 "safetensors", "matplotlib"):
        try:
            mod = __import__(name)
            info[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            info[name] = "MISSING"
    return info


def official_metric_replica(pred, true):
    """Byte-faithful replica of TimeMMD utils/metrics.py MAE/MSE path.

    Keeps the official float32 accumulation semantics (np.mean default
    dtype); returns python floats of those float32 scalars.
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    mae = np.mean(np.abs(pred - true))
    mse = np.mean((pred - true) ** 2)
    return float(mae), float(mse)


def metrics_float64(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    mae = np.mean(np.abs(pred - true))
    mse = np.mean((pred - true) ** 2)
    return float(mae), float(mse)


def close(actual, ref, rtol=1e-12, atol=1e-12):
    return abs(actual - ref) <= atol + rtol * abs(ref)


def parse_printed_metrics(log_text):
    """Parse the official entry line:  mse:{}, mae:{}, rse:{}"""
    hits = re.findall(r"mse:([-\d.eE+]+), mae:([-\d.eE+]+), "
                      r"rse:([-\d.eE+NaN]+)", log_text)
    if not hits:
        return None
    mse, mae, rse = hits[-1]
    return {"mse": float(mse), "mae": float(mae), "rse": float(rse)}


def state_dict_digest(state_dict):
    """Order-stable digest over an in-memory state_dict INCLUDING buffers
    (BatchNorm running_mean/running_var/num_batches_tracked included via
    state_dict). Proves the loaded model was not updated during inference.
    """
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        t = state_dict[k]
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def expected_windows(n_rows, pred_len):
    return int(n_rows * 0.2) - pred_len + 1
