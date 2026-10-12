#!/usr/bin/env python3
"""Shared helpers for repro_timemmd_visionts_zeroshot_v1 (EXP-021).

Inference-configuration single path (applies to BOTH validation period
selection and the official test run): build_visionts() constructs the
vendored Keytoyze VisionTS (arch mae_base, finetune_type 'none'), after
an EXTERNAL strict weight-load check (the constructor prints-and-reraises
but the gate must not rely on constructor behaviour alone), eval() mode,
all parameters frozen; setup_deterministic_env() seeds everything, turns
TF32 off on both backends and enables deterministic cudnn. AMP is never
engaged: all forwards run under torch.inference_mode() on float32 inputs
with forward(fp64=False).
"""
import hashlib
import json
import os
import platform
import random
import re
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
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
    for name in ("torchvision", "timm", "einops", "pillow",
                 "huggingface_hub", "sklearn"):
        try:
            mod = __import__(name)
            info[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            info[name] = "MISSING"
    return info


def official_metric_replica(pred, true):
    """Float32 MAE/MSE with np.mean default-dtype accumulation — the same
    metric space as EXP-019's official-entry replica."""
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


def parse_printed_metrics(line):
    """Parse our own runner's metric line: 'mse:<f>, mae:<f>'."""
    hits = re.findall(r"mse:([-\d.eE+]+), mae:([-\d.eE+]+)", line)
    if not hits:
        return None
    mse, mae = hits[-1]
    return {"mse": float(mse), "mae": float(mae)}


def setup_deterministic_env(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)  # bilinear resize has no
    # deterministic kernel everywhere; determinism comes from fixed inputs
    # (fixed mask noise buffer, sequential batches), not autotune


CKPT_DIR = "/home/wlt/VisionTS_Experiments/ckpt"
CKPT_FILENAME = "mae_visualize_vit_base.pth"
WEIGHTS_SHA256 = ("c551c37b63a221a3e6d7d7f49e96f34078c9b183f788bedc"
                  "0387493521bade0e")
WEIGHTS_BYTES = 447725789


def ckpt_path():
    return os.path.join(CKPT_DIR, CKPT_FILENAME)


def assert_ckpt_present():
    """Hard stop if the weight file is missing — the VisionTS constructor
    would otherwise silently auto-download (never allowed)."""
    p = ckpt_path()
    if not os.path.isfile(p):
        sys.exit(f"[weights] HARD STOP: checkpoint missing at {p}")
    return p


def build_visionts(seed=2021):
    """SINGLE inference-configuration path for the whole experiment
    (preflight, select_period, run_all): deterministic env -> isfile
    assert -> VisionTS(mae_base, finetune 'none') strict constructor
    load -> EXTERNAL strict weight-load check -> eval() -> all params
    frozen.  Returns (model, weights_report).  Caller must .to(device)
    afterwards and then call update_config(...)."""
    import torch
    from visionts import VisionTS

    setup_deterministic_env(seed)
    ckpt = assert_ckpt_present()
    digest = sha256_file(ckpt)
    if digest != WEIGHTS_SHA256:
        sys.exit(f"[weights] HARD STOP: checkpoint sha256 mismatch at "
                 f"{ckpt}: got {digest}")
    model = VisionTS(arch="mae_base", finetune_type="none",
                     ckpt_dir=CKPT_DIR, load_ckpt=True)
    report = verify_loaded_weights(model, ckpt)
    model.eval()
    trainable = [n for n, p in model.vision_model.named_parameters()
                 if p.requires_grad]
    if trainable:
        sys.exit(f"[weights] HARD STOP: {len(trainable)} trainable params "
                 f"in frozen model, first={trainable[:5]}")
    return model, report


def model_state_digest(state_dict):
    """Order-stable digest over a state_dict (keys+shapes+dtypes+bytes)."""
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        t = state_dict[k]
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def verify_loaded_weights(model, ckpt_path):
    """EXTERNAL strict weight-load proof (must not rely on the VisionTS
    constructor's own try/except).

    1. torch.load the checkpoint file at the exact path that was passed
       to the constructor;
    2. strict=True load into a FRESH mae_vit_base_patch16();
    3. compare the live model's vision_model.state_dict() against the
       checkpoint tensors key-for-key, bitwise.
    Returns a report dict; raises SystemExit on any failure.
    """
    import torch
    sys.path.insert(0, HERE)
    from visionts import models_mae

    if not os.path.isfile(ckpt_path):
        sys.exit(f"[weights] HARD STOP: checkpoint missing at load path "
                 f"{ckpt_path} (constructor would auto-download — never "
                 f"allowed)")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model" not in ckpt:
        sys.exit("[weights] HARD STOP: checkpoint lacks 'model' entry")
    ref = models_mae.mae_vit_base_patch16()
    try:
        ref.load_state_dict(ckpt["model"], strict=True)
    except Exception as e:
        sys.exit(f"[weights] HARD STOP: fresh-instance strict load failed: "
                 f"{e!r}")
    live = model.vision_model.state_dict()
    ck = ckpt["model"]
    missing = sorted(set(live.keys()) - set(ck.keys()))
    unexpected = sorted(set(ck.keys()) - set(live.keys()))
    if missing or unexpected:
        sys.exit(f"[weights] HARD STOP: key mismatch missing={missing[:5]} "
                 f"unexpected={unexpected[:5]}")
    bad = []
    for k in sorted(live.keys()):
        a, b = live[k], ck[k]
        if a.shape != b.shape or a.dtype != b.dtype:
            bad.append(k)
        elif not torch.equal(a.cpu(), b.cpu()):
            bad.append(k)
    if bad:
        sys.exit(f"[weights] HARD STOP: {len(bad)} tensors differ from "
                 f"checkpoint, first={bad[:5]}")
    del ref
    return {"ok": True, "ckpt_path": ckpt_path,
            "ckpt_sha256": sha256_file(ckpt_path),
            "n_tensors": len(live),
            "fresh_strict_load": "PASS",
            "live_bitwise_match": "PASS"}
