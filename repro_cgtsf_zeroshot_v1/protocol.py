#!/usr/bin/env python3
"""Frozen protocol for the Aurora x CGTSF single-modality zero-shot
experiment (aurora_cgtsf_zeroshot_v1).

Aurora (aurora-model==0.2.0, fully frozen) receives ONLY the Hist series;
the VisionTS reference is the archived EXP-016 run (commit 0932a5ac...,
protocol token a4c01683793a), frozen locally under visionts_ref/.

Freeze flow: `python freeze_ref.py` (builds visionts_ref/ + MANIFEST.json)
-> `python protocol.py freeze` (embeds constants + code fingerprints +
data/weights pins + visionts_ref manifest). preflight.py / run_infer.py
take --i-approve-frozen-protocol <first 12 hex of sha256(protocol.json)>
and re-verify every pinned fingerprint at start; any mismatch is a hard
stop. Data identity = (dataset, Idx, Date); CSV row order is the fixed
window order everywhere. Policy: protocol/source/package/weight/data
fingerprint mismatch = stop; a shard that is merely corrupt/unfinished is
quarantined and recomputed (run_infer.py)."""
import hashlib
import json
import os
import subprocess
import sys

PROTOCOL_NAME = "aurora_cgtsf_zeroshot_v1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# ---- model / inference (frozen Aurora config) ----
AURORA_PACKAGE_VERSION = "0.2.0"
WEIGHTS_FILE = "model.safetensors"
WEIGHTS_SHA256 = ("df2fb96852a59515a14552d5bddc35c"
                  "03588b6a8bea69355984b3dd926a72b58")
WEIGHTS_BYTES = 843564328
HF_REPO = "DecisionIntelligence/Aurora"
HF_REVISION = "c495b02c1b151be52a3c174237ed240aa66e6384"
ITL = 48                         # inference_token_len
NUM_SAMPLES = 100
BATCH_SIZE = 1                   # strictly one window per forward
REVIN = True
TOKEN_LEN = 48                   # package config token_len (internal patch)
SEEDS = (2021, 2022, 2023)
SEED_PREFIX = "aurora-cgtsf-v1"
SEED_MODULUS = 2 ** 31 - 1
SEED_RULE = ("seed = int.from_bytes(sha256(f'{prefix}|{base}|{ds}|{Idx}|"
             "{Date}'.encode()).digest()[:8], 'big') % (2**31 - 1); "
             "torch.manual_seed(seed) immediately before each window's "
             "generate(); no other RNG consumption in between")
AGGREGATION = ("point forecast = out.to(float64).mean(dim=1) over the "
               "100 samples (old predict.py precedent); per-seed scores "
               "only, mean+-std(ddof=1) across seeds; predictions of "
               "different seeds are never averaged")
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
TF32 = False

# ---- data (reused verbatim from VisionTS EXP-016) ----
CSV_SHA256 = {
    "PTF": "2f8a90f977b48e38f7ce0dae3016de21a2b20bb8657fc2ac803607d26fdf9d3f",
    "LEU": "9f181bb8527c1518bb86f016acb224a167e71cce1368135f78767be069ad7b1c",
    "MSPG": "7c484a2005351db7d22a1d8477cd51c7b98010defbdfd89b73cdae951489315a",
}
CSV_BYTES = {"PTF": 18549020, "LEU": 27518933, "MSPG": 48389951}

SUBSETS = {
    "PTF":  {"context": 120, "pred": 24, "n_seq": 32},
    "LEU":  {"context": 240, "pred": 48, "n_seq": 16},
    "MSPG": {"context": 480, "pred": 96, "n_seq": 27},
}
ORDER = ("PTF", "LEU", "MSPG")
EXPECTED_WINDOWS = {"PTF": 11520, "LEU": 11616, "MSPG": 10557}
EXPECTED_PREFLIGHT_WINDOWS = 230   # 75 seqs x first/mid/last + 5 zero-hist extras
EXPECTED_PAD = {ds: (TOKEN_LEN - SUBSETS[ds]["context"] % TOKEN_LEN) % TOKEN_LEN
                for ds in ORDER}          # internal left zero-pad, no truncation
EXPECTED_PAD_LITERAL = {"PTF": 24, "LEU": 0, "MSPG": 0}
PREDICT_TOKEN_NUM = {ds: -(-SUBSETS[ds]["pred"] // TOKEN_LEN)
                     for ds in ORDER}      # ceil(P/48): 1 / 1 / 2
PREDICT_TOKEN_NUM_LITERAL = {"PTF": 1, "LEU": 1, "MSPG": 2}
TOTAL_FORWARDS = sum(EXPECTED_WINDOWS.values()) * len(SEEDS)   # 101,079

SHARD_SIZE = 512
BATCH_SIZE_LITERAL = 1

VTS_REF_RTOL = 1e-12
VTS_REF_ATOL = 1e-12
RECOMPUTE_RTOL = 1e-12
RECOMPUTE_ATOL = 1e-12

# ---- old-root pins (read-only reuse; md5-frozen) ----
OLD_CODE_FILES = ["load_aurora.py", "predict.py", "SOURCE_HASHES.json"]
AURORA_PKG_FILES = {
    "__init__.py": "55bec5171d216e68eeeb59ce675bac6e",
    "configuration_aurora.py": "452cbcf5a954e5a83ef78df5b778df9d",
    "flow_loss.py": "4bc657919426405fa0ce999dc275da53",
    "modality_connector.py": "49fbbbc03575a554c83bc1a107c3bdc8",
    "modeling_aurora.py": "16cc50855050e1d60dc2ae4ce15e3dd0",
    "prototype_retriever.py": "77e0438345684dc26f5b290c6a4a34f0",
    "ts_generation_mixin.py": "faad71b0796b01ea69a3177c32520cd0",
    "util_functions.py": "546a7518b1d36bd57092962adf0fe040",
}

# ---- VisionTS EXP-016 reference pins ----
VTS_EXP016_ARCHIVE_COMMIT = "0932a5ac841ea399271ce88bde5b607ee0bcb831"
VTS_EXP016_CODE_COMMIT = "a169bb39b23c6708048f206e6df9951c7e439e73"
VTS_EXP016_TOKEN = "a4c01683793a"
VTS_EXP016_DIR = os.environ.get(
    "VTS_EXP016_DIR",
    "/home/wlt/MMTS/VisionTS_Experiments/repro_cgtsf_zeroshot_v1")
VTS_REF_DIR = os.path.join(HERE, "visionts_ref")
VTS_REF_MANIFEST_PATH = os.path.join(VTS_REF_DIR, "MANIFEST.json")

CODE_FILES = ["protocol.py", "common.py", "load_cgtsf.py", "infer_aurora.py",
              "preflight.py", "run_infer.py", "verify_cache.py",
              "evaluate.py", "verify_visionts_ref.py", "recompute_check.py",
              "freeze_ref.py", "audit_pack.py"]

DATA_DIR = os.path.join(HERE, "data")
PRED_DIR = os.path.join(HERE, "predictions")
RESULTS_DIR = os.path.join(HERE, "results")
LOGS_DIR = os.path.join(HERE, "logs")
QUARANTINE_DIR = os.path.join(HERE, "quarantine")
PROTO_PATH = os.path.join(HERE, "protocol.json")
COMMIT_PATH = os.path.join(HERE, "COMMIT.txt")


def csv_path(ds):
    return os.path.join(DATA_DIR, f"{ds}.csv")


def shard_paths(ds, seed, shard):
    base = os.path.join(PRED_DIR, f"{ds}__s{seed}__shard{shard:03d}")
    return base + ".npz", base + ".header.json", base + ".done"


def vts_shard_path(rel):
    return os.path.join(VTS_REF_DIR, rel)


def n_shards(ds):
    n = EXPECTED_WINDOWS[ds]
    return (n + SHARD_SIZE - 1) // SHARD_SIZE


def aurora_pkg_dir():
    import importlib.util
    spec = importlib.util.find_spec("aurora")
    if spec is None or spec.origin is None:
        sys.exit("HARD STOP: aurora package not found in current env "
                 "(run under the aurora conda env)")
    return os.path.dirname(spec.origin)


def repo_commit():
    if os.path.isfile(COMMIT_PATH):
        return open(COMMIT_PATH).read().strip()
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def build_protocol_dict():
    import common
    vts_ref_files = {}
    if os.path.isfile(VTS_REF_MANIFEST_PATH):
        vts_ref_files = json.load(open(VTS_REF_MANIFEST_PATH)).get(
            "files", {})
    return {
        "protocol_name": PROTOCOL_NAME,
        "aurora_package_version": AURORA_PACKAGE_VERSION,
        "weights_file": WEIGHTS_FILE,
        "weights_sha256": WEIGHTS_SHA256, "weights_bytes": WEIGHTS_BYTES,
        "hf_repo": HF_REPO, "hf_revision": HF_REVISION,
        "itl": ITL, "num_samples": NUM_SAMPLES, "batch_size": BATCH_SIZE,
        "revin": REVIN, "token_len": TOKEN_LEN,
        "seeds": list(SEEDS), "seed_prefix": SEED_PREFIX,
        "seed_modulus": SEED_MODULUS, "seed_rule": SEED_RULE,
        "aggregation": AGGREGATION,
        "tf32": TF32,
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
        "csv_sha256": dict(CSV_SHA256), "csv_bytes": dict(CSV_BYTES),
        "subsets": {k: dict(v) for k, v in SUBSETS.items()},
        "order": list(ORDER),
        "expected_windows": dict(EXPECTED_WINDOWS),
        "expected_preflight_windows": EXPECTED_PREFLIGHT_WINDOWS,
        "expected_pad": dict(EXPECTED_PAD),
        "predict_token_num": dict(PREDICT_TOKEN_NUM),
        "total_forwards": TOTAL_FORWARDS,
        "shard_size": SHARD_SIZE,
        "vts_ref_rtol": VTS_REF_RTOL, "vts_ref_atol": VTS_REF_ATOL,
        "recompute_rtol": RECOMPUTE_RTOL, "recompute_atol": RECOMPUTE_ATOL,
        "vts_exp016_archive_commit": VTS_EXP016_ARCHIVE_COMMIT,
        "vts_exp016_code_commit": VTS_EXP016_CODE_COMMIT,
        "vts_exp016_token": VTS_EXP016_TOKEN,
        "vts_ref_files": vts_ref_files,
        "code_md5": {n: common.md5_file(os.path.join(HERE, n))
                     for n in CODE_FILES},
        "old_code_md5": {n: common.md5_file(os.path.join(REPO_ROOT, n))
                         for n in OLD_CODE_FILES},
        "aurora_pkg_md5": {n: common.md5_file(os.path.join(aurora_pkg_dir(), n))
                           for n in sorted(AURORA_PKG_FILES)},
        "repo_commit": repo_commit(),
    }


def freeze():
    import common
    if not os.path.isfile(VTS_REF_MANIFEST_PATH):
        sys.exit("visionts_ref/MANIFEST.json missing; run freeze_ref.py "
                 "BEFORE protocol.py freeze")
    d = build_protocol_dict()
    b = json.dumps(d, indent=1, sort_keys=True).encode()
    tmp = PROTO_PATH + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b)
    os.replace(tmp, PROTO_PATH)
    token = hashlib.sha256(open(PROTO_PATH, "rb").read()).hexdigest()[:12]
    print("frozen protocol.json ->", PROTO_PATH)
    print("approval token:", token)


def check_approval(token):
    """Load protocol.json and verify the approval token (= first 12 hex of
    its sha256). Hard stop on mismatch/absence."""
    if not os.path.isfile(PROTO_PATH):
        sys.exit("protocol.json missing; run: python protocol.py freeze")
    digest = hashlib.sha256(open(PROTO_PATH, "rb").read()).hexdigest()
    if token != digest[:12]:
        sys.exit(f"approval token mismatch: got {token!r}, "
                 f"want {digest[:12]!r} (sha256 of protocol.json)")
    return json.load(open(PROTO_PATH))


def verify_frozen(proto):
    """Hard-stop verification of every pin: experiment code, old-root
    code, aurora package sources, CSVs, visionts_ref manifest."""
    import common
    if proto["csv_sha256"] != CSV_SHA256:
        sys.exit("protocol.json/constants drift (csv pins differ)")
    if proto["expected_pad"] != EXPECTED_PAD \
            or proto["expected_pad"] != EXPECTED_PAD_LITERAL:
        sys.exit("protocol.json/constants drift (pad table differs)")
    if proto["predict_token_num"] != PREDICT_TOKEN_NUM \
            or proto["predict_token_num"] != PREDICT_TOKEN_NUM_LITERAL:
        sys.exit("protocol.json/constants drift (token table differs)")
    bad = [n for n, m in proto["code_md5"].items()
           if not os.path.isfile(os.path.join(HERE, n))
           or common.md5_file(os.path.join(HERE, n)) != m]
    bad += [n for n, m in proto["old_code_md5"].items()
            if not os.path.isfile(os.path.join(REPO_ROOT, n))
            or common.md5_file(os.path.join(REPO_ROOT, n)) != m]
    pkg = aurora_pkg_dir()
    bad += [f"aurora:{n}" for n, m in proto["aurora_pkg_md5"].items()
            if not os.path.isfile(os.path.join(pkg, n))
            or common.md5_file(os.path.join(pkg, n)) != m]
    if bad:
        sys.exit(f"FROZEN CODE MISMATCH (re-freeze required, hard stop): {bad}")
    for ds in ORDER:
        p = csv_path(ds)
        if not os.path.isfile(p) or common.sha256_file(p) != CSV_SHA256[ds]:
            sys.exit(f"CSV sha mismatch for {ds}: {p}")
    verify_vts_ref_files(proto["vts_ref_files"])


def verify_vts_ref_files(vts_ref_files):
    """Every frozen visionts_ref file must exist with the pinned sha256."""
    import common
    if not vts_ref_files:
        sys.exit("protocol.json embeds no visionts_ref manifest; "
                 "re-freeze after freeze_ref.py")
    if not os.path.isfile(VTS_REF_MANIFEST_PATH):
        sys.exit(f"visionts_ref missing: {VTS_REF_MANIFEST_PATH}")
    disk = json.load(open(VTS_REF_MANIFEST_PATH)).get("files", {})
    if disk != vts_ref_files:
        sys.exit("visionts_ref/MANIFEST.json differs from the manifest "
                 "embedded in protocol.json (re-freeze or redeploy)")
    bad = [rel for rel, sha in vts_ref_files.items()
           if not os.path.isfile(os.path.join(VTS_REF_DIR, rel))
           or common.sha256_file(os.path.join(VTS_REF_DIR, rel)) != sha]
    if bad:
        sys.exit(f"VISIONTS_REF MISMATCH (hard stop): {bad[:10]} "
                 f"({len(bad)} files)")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "freeze":
        freeze()
    else:
        print(__doc__)
        print("usage: python protocol.py freeze")
