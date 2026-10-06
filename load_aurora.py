"""Self-contained Aurora loader with full integrity and determinism control.

Timing of determinism settings (protocol.json "determinism.timing"):
  - CUBLAS_WORKSPACE_CONFIG is set in process env at import time of this module,
    BEFORE any CUDA initialization.
  - apply_determinism() (deterministic algorithms, TF32 off, cudnn benchmark off)
    must be called BEFORE the first inference call. It is called automatically
    by load_aurora(); drivers that build models another way must call it themselves.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file as safetensors_load_file

PROJECT = Path(__file__).resolve().parent
PROTOCOL = json.loads((PROJECT / "protocol.json").read_text())

EXPECTED_WEIGHTS_SHA256 = PROTOCOL["model"]["weights_sha256"]
EXPECTED_WEIGHTS_BYTES = PROTOCOL["model"]["weights_bytes"]
HF_REPO = PROTOCOL["model"]["hf_repo"]
HF_REVISION = PROTOCOL["model"]["hf_revision"]

# md5 of the key source files of aurora-model==0.2.0, from SOURCE_HASHES.json
SOURCE_HASHES = json.loads((PROJECT / "SOURCE_HASHES.json").read_text())
AURORA_FILES_MD5 = SOURCE_HASHES["sources"]["aurora_package"]["files_md5"]


def apply_env_before_cuda():
    """Idempotent safety net; the real setting happens at import time above."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def apply_determinism():
    """Determinism pack. Call before the first inference; safe to call again."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def md5_file(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_aurora_package():
    """md5-verify key source files of the installed aurora package."""
    import aurora
    pkg_dir = Path(aurora.__file__).resolve().parent
    report = {"package_dir": str(pkg_dir), "version": aurora.__version__, "files": {}, "all_ok": True}
    for name, expect in AURORA_FILES_MD5.items():
        p = pkg_dir / name
        got = md5_file(p) if p.exists() else "MISSING"
        report["files"][name] = {"expected": expect, "got": got, "ok": got == expect}
        report["all_ok"] &= got == expect
    return report


def resolve_weights_path(explicit=None):
    """Return a local weights path. explicit path -> verify; else use HF cache layout
    (already-downloaded snapshot) or hf_hub_download with pinned revision."""
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    # 1) already in HF cache? blob file name == sha256
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    repo_dir = cache / f"models--{HF_REPO.replace('/', '--')}"
    snap = repo_dir / "snapshots" / HF_REVISION / "model.safetensors"
    if snap.exists():
        real = snap.resolve()
        return real

    # 2) download via mirror with pinned revision
    from huggingface_hub import hf_hub_download
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    return Path(hf_hub_download(repo_id=HF_REPO, filename="model.safetensors", revision=HF_REVISION))


def verify_weights(weights_path):
    size = weights_path.stat().st_size
    sha = sha256_file(weights_path)
    ok = (sha == EXPECTED_WEIGHTS_SHA256) and (size == EXPECTED_WEIGHTS_BYTES)
    return {"path": str(weights_path), "bytes": size, "sha256": sha, "ok": ok}


def _config_from_package():
    from aurora import AuroraConfig
    from aurora.utils.path_utils import get_package_file_path
    return AuroraConfig.from_json_file(get_package_file_path("config.json"))


def load_aurora(weights_path=None, device=None, verify_package=True):
    """Integrity-checked, frozen, eval-mode AuroraForPrediction.

    Returns (model, report). Raises on any integrity failure. Determinism pack is
    applied before returning so the caller's first inference is covered.
    """
    apply_env_before_cuda()
    apply_determinism()

    report = {"determinism": {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }}

    if verify_package:
        report["package"] = verify_aurora_package()
        if not report["package"]["all_ok"]:
            raise RuntimeError("aurora package md5 mismatch: " + json.dumps(report["package"]["files"]))

    wp = resolve_weights_path(weights_path)
    wv = verify_weights(wp)
    if not wv["ok"]:
        raise RuntimeError(f"weights sha256/size mismatch: {wv}")
    report["weights"] = wv

    from aurora import AuroraForPrediction
    config = _config_from_package()
    model = AuroraForPrediction(config)

    weights = safetensors_load_file(str(wp), device="cpu")

    model_keys = set(model.state_dict().keys())
    weight_keys = set(weights.keys())
    missing = sorted(model_keys - weight_keys)
    unexpected = sorted(weight_keys - model_keys)
    report["key_diff"] = {"missing": missing, "unexpected": unexpected,
                          "n_model_keys": len(model_keys), "n_weight_keys": len(weight_keys)}
    if missing or unexpected:
        raise RuntimeError(f"key diff non-empty: missing={missing[:10]}... unexpected={unexpected[:10]}...")

    # num_batches_tracked etc. are covered by the strict load itself
    result = model.load_state_dict(weights, strict=True)
    report["strict_load"] = {"missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys)}

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for b in model.buffers():
        b.requires_grad_(False)

    report["mode"] = {"training": model.training, "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)}

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    report["device"] = device

    return model, report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="explicit model.safetensors path (else HF cache/mirror)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-package-check", action="store_true")
    args = ap.parse_args()
    model, rep = load_aurora(args.weights, args.device, verify_package=not args.no_package_check)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
