#!/usr/bin/env python3
"""Build the final audit bundle for repro_timemmd_economy_chrono_v1.

Single-stage FINAL pack (<50MB):
  - all experiment code + the official prediction-saving patch;
  - protocol.json; data_prep.json + sort_mapping.csv; BOTH Economy CSVs
    (sorted + original as read-only base evidence);
  - verify_refs / preflight / batch decision / probe / split_report /
    weights_load_check / env reports;
  - all 4 groups: npz + done.json + captured logs;
  - Economy metrics + comparison_economy_chrono.csv + summary.json;
  - base_evidence/: EXP-019 frozen protocol.json + summary.json + the
    original-row-order Economy__h{6,8,10,12} npz/done (READ-ONLY inputs,
    reproduced for the chrono-vs-orig comparison).

Hard rules: < 50 MB, redaction scan (secret-looking strings; known public
protocol tokens whitelisted), MANIFEST.sha256 inside, top-level dir named
after the experiment. Output: pack_out/*.tar.gz (+ full SHA256 printed).
"""
import argparse
import hashlib
import io
import os
import re
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
    "protocol.py", "common.py", "prepare_data.py", "verify_refs.py",
    "preflight.py", "probe_one_batch.py", "run_all.py",
    "save_group_meta.py", "evaluate.py", "audit_pack.py",
    "wrapper_dt.sh", "run_chrono_dt.sh",
    "paper_ref.json", "README.md", ".gitignore",
    "patches/exp_main_pred_save.diff",
]

HORIZONS = (6, 8, 10, 12)


def BASE_PRED():
    return os.path.join(P.BASE_DIR, "predictions")

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
             P.WEIGHTS_SHA256.encode(), P.BASE_ADD_COMMIT.encode(),
             P.BASE_FREEZE_COMMIT.encode(),
             P.BASE_PROTOCOL_TOKEN.encode(),
             P.ORIGINAL_CSV_SHA256.encode()]


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def add_file(files, arc, abs_path, must=True):
    if os.path.isfile(abs_path):
        files.append((arc, abs_path))
    elif must:
        sys.exit(f"[audit-pack] required file missing: {abs_path}")


def gather_files():
    files = []  # (arcname, abs_path)
    for rel in CODE_FILES:
        add_file(files, f"{TOP}/{rel}", os.path.join(HERE, rel))
    add_file(files, f"{TOP}/protocol.json", os.path.join(HERE, "protocol.json"))

    # data prep evidence + both CSVs
    add_file(files, f"{TOP}/results/data_prep.json",
             os.path.join(C.RESULTS_DIR, "data_prep.json"))
    add_file(files, f"{TOP}/results/sort_mapping.csv",
             os.path.join(C.RESULTS_DIR, "sort_mapping.csv"))
    add_file(files, f"{TOP}/dataset/Economy.csv", P.csv_path("Economy"))
    add_file(files, f"{TOP}/base_evidence/Economy_original.csv",
             os.path.join(P.BASE_DIR, "dataset", "Economy.csv"))

    # pre-run evidence
    for rel in ("results/verify_refs.json",
                "results/preflight_report.json",
                "results/batch_decision.json",
                "results/probe_Economy.json",
                "results/split_report.json",
                "results/weights_load_check.json",
                "logs/env_diff.txt",
                "logs/pip_freeze_aurora_tmmd_v1.txt",
                "logs/pip_freeze_torch_50_series.txt"):
        add_file(files, f"{TOP}/{rel}", os.path.join(HERE, rel), must=False)

    # per-group evidence
    for h in HORIZONS:
        n = f"Economy__h{h}"
        add_file(files, f"{TOP}/predictions/{n}.npz",
                 os.path.join(C.PRED_DIR, f"{n}.npz"))
        add_file(files, f"{TOP}/predictions/{n}.done.json",
                 os.path.join(C.PRED_DIR, f"{n}.done.json"))
        add_file(files, f"{TOP}/logs/group_{n}.log",
                 os.path.join(C.LOGS_DIR, f"group_{n}.log"))

    # results
    for rel in ("results/Economy_metrics.json",
                "results/comparison_economy_chrono.csv",
                "results/summary.json",
                "results/env_report.json",
                "logs/run_all_summary.json"):
        add_file(files, f"{TOP}/{rel}", os.path.join(HERE, rel))

    # base experiment evidence (EXP-019, read-only)
    add_file(files, f"{TOP}/base_evidence/EXP-019_protocol.json",
             P.base_protocol_path())
    add_file(files, f"{TOP}/base_evidence/EXP-019_summary.json",
             os.path.join(P.BASE_DIR, "results", "summary.json"))
    for h in HORIZONS:
        n = f"Economy__h{h}"
        add_file(files, f"{TOP}/base_evidence/{n}.npz",
                 os.path.join(BASE_PRED(), f"{n}.npz"))
        add_file(files, f"{TOP}/base_evidence/{n}.done.json",
                 os.path.join(BASE_PRED(), f"{n}.done.json"))
    return files


def build():
    os.makedirs(PACK_OUT, exist_ok=True)
    files = gather_files()
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
    tar_name = "Aurora_TimeMMD_economy_chrono_final_" + stamp + ".tar.gz"
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
    print(f"[audit-pack] {tar_path}")
    print(f"[audit-pack] files={len(buf_by_name)+1} bytes={sz} "
          f"sha256={out_sha}")
    if sz > MAX_BYTES:
        sys.exit("[audit-pack] packed tar exceeds 50MB")
    return tar_path, out_sha, sz


def time_tag():
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d_%H%M")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["final"], default="final")
    a = ap.parse_args()
    if not os.path.isfile(os.path.join(HERE, "protocol.json")):
        sys.exit("[audit-pack] protocol.json missing — freeze first")
    build()
