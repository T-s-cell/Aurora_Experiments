#!/usr/bin/env python3
"""Shared plumbing: paths, read-only references, hashing, state markers,
per-stage input guards, forward fingerprint. The old Aurora project (repo
root) and VisionTS EXP-014 are READ-ONLY."""
import hashlib
import json
import os
import sys

import protocol as P

HERE = P.HERE
REPO_ROOT = P.REPO_ROOT
OLD_DIR = P.OLD_DIR
VTS_DIR = P.VTS_DIR

for p in (OLD_DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

REF_DIR = os.path.join(HERE, "visionts_ref")
CACHE_DIR = os.path.join(HERE, "cache_all")
RESULTS_DIR = os.path.join(HERE, "results")
LOGS_DIR = os.path.join(HERE, "logs")
STATE_DIR = os.path.join(HERE, ".state")

INV_PATH = os.path.join(REF_DIR, "inventory.json")
ARRAYS_PATH = os.path.join(REF_DIR, "frozen_arrays.npz")
VARS_PATH = os.path.join(REF_DIR, "vars.json")
VTS_PREDS_PATH = os.path.join(REF_DIR, "preds.jsonl.gz")
REF_REPORT = os.path.join(REF_DIR, "ref_report.json")
REUSE_PLAN = os.path.join(HERE, "reuse_plan.json")
REUSE_FINAL = os.path.join(HERE, "reuse_final.json")
PREFLIGHT_SAMPLE = os.path.join(HERE, "preflight_sample.json")
MANIFEST_PATH = os.path.join(CACHE_DIR, "manifest.json")


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


def code_md5_map(names, base=HERE):
    return {n: md5_file(os.path.join(base, n)) for n in names
            if os.path.isfile(os.path.join(base, n))}


def new_code_md5():
    return code_md5_map(P.NEW_CODE_ALL, HERE)


def old_code_md5():
    return code_md5_map(P.OLD_CODE_REUSED + P.OLD_CODE_INFO, OLD_DIR)


def dump_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def state_path(stage):
    return os.path.join(STATE_DIR, f"{stage}.done")


def mark_done(stage, payload):
    os.makedirs(STATE_DIR, exist_ok=True)
    payload = dict(payload)
    payload["status"] = "done"
    dump_json(payload, state_path(stage))


def state_of(stage):
    p = state_path(stage)
    if os.path.isfile(p):
        return load_json(p)
    return None


def env_info():
    import platform
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "host": os.uname().nodename,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    except ImportError:
        info["torch"] = None
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except ImportError:
        info["numpy"] = None
    try:
        import aurora
        info["aurora_model"] = getattr(aurora, "__version__", "unknown")
    except ImportError:
        info["aurora_model"] = None
    return info


def old_shard_guards():
    """sha256 of every EXP-010 A48 shard npz (57 files). Used as input guards
    for stages consuming the old cache."""
    out = {}
    for domain in P.EXPECTED_DOMAINS:
        for seed in P.SEEDS:
            name = P.old_shard_name(domain, seed)
            path = os.path.join(P.OLD_PRED_DIR, name)
            if os.path.isfile(path):
                out["old_shard:" + name] = sha256_file(path)
    return out


def stage_input_guards(stage):
    """Declarative per-stage INPUT fingerprint, recorded into the stage marker
    at completion; stage_guard recomputes the same declaration and invalidates
    the stage when anything (incl. protocol.py) changed.
    Keys: relpath under HERE -> sha256; 'code_md5:<name>' -> md5; friendly
    absolute keys -> sha256."""
    g = {}

    def add_rel(rel):
        g[rel] = sha256_file(os.path.join(HERE, rel))

    def add_abs(path, key):
        g[key] = sha256_file(path)

    def add_code(names, base=None):
        for n in names:
            g["code_md5:" + n] = md5_file(os.path.join(base or HERE, n))

    refs = ["visionts_ref/inventory.json", "visionts_ref/frozen_arrays.npz",
            "visionts_ref/vars.json", "visionts_ref/preds.jsonl.gz",
            "visionts_ref/split_manifest.json"]
    if stage == "freeze_ref":
        add_code(["protocol.py", "common.py", "freeze_ref.py"])
    elif stage == "verify":
        for r in refs:
            add_rel(r)
        add_abs(P.LN_MANIFEST, "ln_manifest_sha256")
        add_abs(os.path.join(P.OLD_DATA_DIR, "data_cache.npz"), "old_data_cache_sha256")
        add_abs(os.path.join(P.OLD_DATA_DIR, "split_manifest.json"), "old_split_manifest_sha256")
        add_code(["protocol.py", "common.py", "verify_cache.py"])
        add_code(P.OLD_CODE_REUSED + P.OLD_CODE_INFO, base=OLD_DIR)
        g.update(old_shard_guards())
    elif stage == "sample":
        for r in refs:
            add_rel(r)
        add_rel("reuse_plan.json")
        add_abs(P.LN_MANIFEST, "ln_manifest_sha256")
        add_code(["protocol.py", "common.py", "preflight.py"])
    elif stage == "preflight":
        for r in refs:
            add_rel(r)
        add_rel("reuse_plan.json")
        add_rel("preflight_sample.json")
        add_code(["protocol.py", "common.py", "preflight.py", "infer_fill.py"])
        add_code(P.OLD_CODE_REUSED, base=OLD_DIR)
    elif stage == "fill":
        for r in refs:
            add_rel(r)
        add_rel("reuse_plan.json")
        add_rel("reuse_final.json")
        add_rel("cache_all/manifest.json")
        add_code(["protocol.py", "common.py", "preflight.py", "infer_fill.py"])
        add_code(P.OLD_CODE_REUSED, base=OLD_DIR)
    elif stage == "assemble":
        for r in refs:
            add_rel(r)
        add_rel("reuse_plan.json")
        add_rel("reuse_final.json")
        add_rel("cache_all/manifest.json")
        add_abs(P.LN_MANIFEST, "ln_manifest_sha256")
        add_code(["protocol.py", "common.py", "assemble.py"])
        g.update(old_shard_guards())
    elif stage == "evaluate":
        for r in refs:
            add_rel(r)
        add_rel("results/aurora_preds.jsonl.gz")
        add_rel("reuse_plan.json")
        add_rel("reuse_final.json")
        add_abs(P.LN_MANIFEST, "ln_manifest_sha256")
        add_code(["protocol.py", "common.py", "assemble.py", "evaluate.py"])
    elif stage == "recompute":
        # standalone script: must stay identical to the dict built inside
        # recompute_check.py (which cannot import this module)
        for r in refs:
            add_rel(r)
        for r in ("results/aurora_preds.jsonl.gz", "results/variable_level.csv",
                  "results/domain_level.csv", "results/overall.json"):
            add_rel(r)
        add_code(["recompute_check.py"])
    elif stage == "accept":
        for r in refs:
            add_rel(r)
        for r in ("results/aurora_preds.jsonl.gz", "results/recompute_check.json",
                  "results/subset_2474.json"):
            add_rel(r)
        add_rel("reuse_plan.json")
        add_rel("reuse_final.json")
        add_abs(P.LN_MANIFEST, "ln_manifest_sha256")
        add_code(["protocol.py", "common.py", "verify_acceptance.py"])
    else:
        raise ValueError(f"unknown stage: {stage}")
    return g


def forward_fingerprint():
    """Fingerprint binding GPU forwards to data refs + weights + code + config
    + env + aurora package. Computed ONLY on the GPU host (S3 preflight --run
    writes it into reuse_final.json; S4 infer_fill re-derives and compares)."""
    import load_aurora as LA  # read-only old module; sets CUBLAS env at import

    weights_path = LA.resolve_weights_path(None)
    wv = LA.verify_weights(weights_path)
    if not wv["ok"]:
        raise SystemExit(f"weights sha256/bytes mismatch: {wv}")
    pkg = LA.verify_aurora_package()
    if not pkg["all_ok"]:
        raise SystemExit("aurora package md5 mismatch: "
                         + json.dumps(pkg["files"]))
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == P.CUBLAS_WORKSPACE_CONFIG, \
        "CUBLAS_WORKSPACE_CONFIG not set before CUDA init"
    return {
        "protocol_version": P.PROTOCOL_VERSION,
        "inventory_sha256": sha256_file(INV_PATH),
        "arrays_sha256": sha256_file(ARRAYS_PATH),
        "vars_sha256": sha256_file(VARS_PATH),
        "vts_preds_sha256": sha256_file(VTS_PREDS_PATH),
        "ln_manifest_sha256": sha256_file(P.LN_MANIFEST),
        "old_split_manifest_sha256": P.OLD_SPLIT_MANIFEST_SHA256,
        "old_data_cache_sha256": P.OLD_DATA_CACHE_SHA256,
        "weights_sha256": wv["sha256"],
        "weights_bytes": wv["bytes"],
        "hf_revision": P.HF_REVISION,
        "itl": P.ITL,
        "num_samples": P.NUM_SAMPLES,
        "seeds": list(P.SEEDS),
        "seed_rule": P.SEED_RULE,
        "code_md5_forward": code_md5_map(P.NEW_CODE_FORWARD, HERE),
        "code_md5_reused_old": code_md5_map(P.OLD_CODE_REUSED, OLD_DIR),
        "aurora_package_md5": {k: v["got"] for k, v in pkg["files"].items()},
        "env": env_info(),
    }
