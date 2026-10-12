#!/usr/bin/env python3
"""Build the audit bundle for repro_timemmd_zeroshot_v1.

Stages:
  preflight : frozen code+protocol+diff, verify_refs, preflight report,
              batch decision, probe reports, mini-group evidence, paper
              reference — delivered for USER REVIEW BEFORE the full run.
  final     : everything above plus per-group npz/done, logs, metrics,
              comparison tables, summary.

Hard rules: < 50 MB, redaction scan (secret-looking strings; known public
protocol tokens whitelisted), MANIFEST.sha256 inside, top-level dir named
after the experiment. Output: pack_out/*.tar.gz (+ copy path printed).
"""
import argparse
import hashlib
import io
import os
import re
import shutil
import sys
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C
import protocol as P

PACK_OUT = os.path.join(HERE, "pack_out")
TOP = P.PROTOCOL_NAME
MAX_BYTES = 50 * 1024 * 1024

CODE_FILES = [
    "protocol.py", "common.py", "verify_refs.py", "preflight.py",
    "probe_one_batch.py", "run_all.py", "save_group_meta.py",
    "evaluate.py", "audit_pack.py",
    "wrapper_dt.sh", "watch_and_run_dt.sh",
    "paper_ref.json", "README.md", ".gitignore",
    "patches/exp_main_pred_save.diff",
]

REDACT_PATTERNS = [
    re.compile(rb"(?i)(password|passwd|secret|api[_-]?key)\s*[:=]\s*"
               rb"[\"']?[^\s\"']{6,}"),
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    re.compile(rb"sk-[A-Za-z0-9]{16,}"),
    re.compile(rb"ghp_[A-Za-z0-9]{20,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"(?i)-----begin [a-z ]*private key-----"),
    re.compile(rb"[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s@]{6,}@"),
]
# public approval tokens (sha256 prefixes) are not secrets
WHITELIST = [P.OFFICIAL_REPO_COMMIT.encode(), P.HF_REVISION.encode(),
             P.WEIGHTS_SHA256.encode()]


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def scan_redaction(buf, name):
    low = buf.lower()
    for pat in REDACT_PATTERNS:
        for m in pat.finditer(low):
            s = max(0, m.start() - 30)
            frag = buf[s:m.end() + 10]
            if any(w in frag for w in WHITELIST):
                continue
            if b"token" in m.group(0).lower() and (
                    b"approval" in low[s:m.start()]
                    or b"protocol_token" in frag
                    or b"i-approve" in low[s:m.start()]):
                continue
            sys.exit(f"[audit-pack] REDACTION HIT in {name}: "
                     f"{frag[-80:]!r}")


def gather_files(stage):
    files = []  # (arcname, abs_path or bytes)
    for rel in CODE_FILES:
        p = os.path.join(HERE, rel)
        if not os.path.isfile(p):
            sys.exit(f"[audit-pack] missing code file {rel}")
        files.append((f"{TOP}/{rel}", p))
    for rel in ("protocol.json",):
        p = os.path.join(HERE, rel)
        if not os.path.isfile(p):
            sys.exit("[audit-pack] protocol.json missing — freeze first")
        files.append((f"{TOP}/{rel}", p))
    for rel in ("results/verify_refs.json",
                "results/preflight_report.json",
                "results/batch_decision.json",
                "results/env_report.json",
                "logs/preflight_security_h12.log",
                "logs/env_diff.txt", "logs/pip_freeze_aurora_tmmd_v1.txt",
                "logs/pip_freeze_torch_50_series.txt"):
        p = os.path.join(HERE, rel)
        if os.path.isfile(p):
            files.append((f"{TOP}/{rel}", p))
    probe_dir = os.path.join(C.RESULTS_DIR)
    if os.path.isdir(probe_dir):
        for n in sorted(os.listdir(probe_dir)):
            if n.startswith("probe_") and n.endswith(".json"):
                files.append((f"{TOP}/results/{n}",
                              os.path.join(probe_dir, n)))
    if stage == "final":
        for n in sorted(os.listdir(C.LOGS_DIR)):
            if n.startswith("group_") and n.endswith(".log"):
                files.append((f"{TOP}/logs/{n}",
                              os.path.join(C.LOGS_DIR, n)))
        for rel in ("results/comparison_table_12.csv",
                    "results/comparison_table_1.csv",
                    "results/summary.json",
                    "results/recompute_check.json",
                    "results/weights_load_check.json",
                    "logs/run_all_summary.json",
                    "logs/infer_env.json"):
            p = os.path.join(HERE, rel)
            if os.path.isfile(p):
                files.append((f"{TOP}/{rel}", p))
        for n in sorted(os.listdir(C.PRED_DIR)):
            if n.endswith((".npz", ".done.json")):
                files.append((f"{TOP}/predictions/{n}",
                              os.path.join(C.PRED_DIR, n)))
    return files


def build(stage):
    os.makedirs(PACK_OUT, exist_ok=True)
    files = gather_files(stage)
    manifest = []
    total = 0
    buf_by_name = {}
    for arc, src in files:
        with open(src, "rb") as f:
            buf = f.read()
        scan_redaction(buf, arc)
        buf_by_name[arc] = buf
        total += len(buf)
    if total > MAX_BYTES:
        sys.exit(f"[audit-pack] payload {total} > {MAX_BYTES} bytes; "
                 f"trim before packing")
    stamp = time_tag()
    tar_name = (f"Aurora_TimeMMD_zeroshot_{stage}_{stamp}.tar.gz")
    tar_path = os.path.join(PACK_OUT, tar_name)
    with tarfile.open(tar_path, "w:gz") as tf:
        for arc, buf in buf_by_name.items():
            ti = tarfile.TarInfo(arc)
            ti.size = len(buf)
            ti.mtime = int(time.time())
            tf.addfile(ti, io.BytesIO(buf))
            manifest.append(f"{sha256_bytes(buf)}  {arc}")
        mb = ("\n".join(manifest) + "\n").encode()
        ti = tarfile.TarInfo(f"{TOP}/MANIFEST.sha256")
        ti.size = len(mb)
        ti.mtime = int(time.time())
        tf.addfile(ti, io.BytesIO(mb))
    out_sha = sha256_file(tar_path)
    sz = os.path.getsize(tar_path)
    print(f"[audit-pack] {tar_path} files={len(buf_by_name)+1} "
          f"bytes={sz} sha256={out_sha}")
    if sz > MAX_BYTES:
        sys.exit("[audit-pack] packed tar exceeds 50MB")
    return tar_path, out_sha, sz


def time_tag():
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d_%H%M")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["preflight", "final"], required=True)
    a = ap.parse_args()
    if not os.path.isfile(os.path.join(HERE, "protocol.json")):
        sys.exit("[audit-pack] protocol.json missing — freeze first")
    build(a.stage)
