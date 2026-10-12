#!/usr/bin/env python3
"""Audit bundles for repro_timemmd_visionts_economy_chrono_v1 (EXP-022).

Two fixed packs per stage (<50MB each, hard gate):
  main: code + vendored visionts/ + protocol.json + all gate reports +
        selected_p.json + (final only) 4 npz/done + logs + comparison
        CSVs + summary + env freezes
        + the ACTUAL ascending dataset/Economy.csv used
        + results/base_evidence/ (EXP-020 sort mapping + reports)
        + reference_evidence/: EXP-020 Economy x4 npz+done, EXP-021
        Economy x4 npz+done, both anchor protocol.json — with a
        sources.sha256 manifest
        weights are NEVER packed
  val_evidence: results/val_evidence/*.npz (every candidate P's
        validation predictions, float32) + selected_p.json

--stage preflight: run after freeze (protocol + selected_p must exist)
--stage final:     run after run_all + evaluate
"""
import argparse
import datetime
import hashlib
import io
import os
import re
import sys
import tarfile
import time

import common as C
import protocol as P

PACK_OUT = os.path.join(C.HERE, "pack_out")
TOP = P.PROTOCOL_NAME
MAX_BYTES = 50 * 1024 * 1024
AURORA_ROOT = "/home/wlt/MMTS/Aurora_Experiments"

CODE_FILES = [
    "protocol.py", "common.py", "data.py", "prepare_data.py",
    "verify_refs.py", "preflight.py", "select_period.py", "run_all.py",
    "save_group_meta.py", "evaluate.py", "audit_pack.py",
    "wrapper_dt.sh", "run_visionts_dt.sh",
    "paper_ref_exp020_exp021.json", "README.md", ".gitignore",
    "visionts/__init__.py", "visionts/model.py", "visionts/models_mae.py",
    "visionts/pos_embed.py", "visionts/util.py",
]

BASE_DATA_DIR = os.path.join(AURORA_ROOT, P.BASE_DATA["dir"])
BASE_METHOD_DIR = os.path.join(AURORA_ROOT, P.BASE_METHOD["dir"])

REFERENCE_EVIDENCE = (
    [(f"reference_evidence/exp020/predictions/Economy__h{h}.npz",
      os.path.join(BASE_DATA_DIR, "predictions", f"Economy__h{h}.npz"))
     for h in P.DATASETS["Economy"]["horizons"]]
    + [(f"reference_evidence/exp020/predictions/Economy__h{h}.done.json",
        os.path.join(BASE_DATA_DIR, "predictions",
                     f"Economy__h{h}.done.json"))
       for h in P.DATASETS["Economy"]["horizons"]]
    + [(f"reference_evidence/exp021/predictions/Economy__h{h}.npz",
        os.path.join(BASE_METHOD_DIR, "predictions", f"Economy__h{h}.npz"))
       for h in P.DATASETS["Economy"]["horizons"]]
    + [(f"reference_evidence/exp021/predictions/Economy__h{h}.done.json",
        os.path.join(BASE_METHOD_DIR, "predictions",
                     f"Economy__h{h}.done.json"))
       for h in P.DATASETS["Economy"]["horizons"]]
    + [("reference_evidence/exp020/protocol.json",
        os.path.join(BASE_DATA_DIR, "protocol.json")),
       ("reference_evidence/exp021/protocol.json",
        os.path.join(BASE_METHOD_DIR, "protocol.json"))]
)

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
WHITELIST = [P.BASE_DATA["add_commit"].encode(),
             P.BASE_DATA["freeze_commit"].encode(),
             P.BASE_DATA["protocol_sha256"].encode(),
             P.BASE_METHOD["add_commit"].encode(),
             P.BASE_METHOD["freeze_commit"].encode(),
             P.BASE_METHOD["protocol_sha256"].encode(),
             P.VISIONTS_REPO_COMMIT.encode(),
             C.WEIGHTS_SHA256.encode(),
             P.DATASETS["Economy"]["csv_sha256"].encode(),
             P.DATASETS["Economy"]["original_csv_sha256"].encode(),
             P.DATASETS["Economy"]["sort_mapping_sha256"].encode()]


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    return C.sha256_file(path)


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
                    or b"selected" in low[s:m.start()]
                    or b"cgtsf_token" in frag):
                continue
            sys.exit(f"[audit-pack] REDACTION HIT in {name}: "
                     f"{frag[-80:]!r}")


def add_file(files, arc, abs_path, must=True):
    if os.path.isfile(abs_path):
        files.append((arc, abs_path))
    elif must:
        sys.exit(f"[audit-pack] required file missing: {abs_path}")


def gather_main(stage):
    files = []
    for rel in CODE_FILES:
        add_file(files, f"{TOP}/{rel}", os.path.join(C.HERE, rel))
    add_file(files, f"{TOP}/protocol.json", C.PROTO_PATH)
    results = ["data_prep.json", "verify_refs.json", "preflight_data.json",
               "preflight_geometry.json", "preflight_smoke.json",
               "batch_decision.json", "weights_load_check.json",
               "weights_digest_baseline.json", "selected_p.json"]
    if stage == "final":
        results += ["comparison_per_h.csv", "comparison_per_mean.csv",
                    "summary.json", "run_all_summary.json"]
    for rel in results:
        add_file(files, f"{TOP}/results/{rel}",
                 os.path.join(C.RESULTS_DIR, rel))
    # the actual ascending CSV used by every window in this experiment
    add_file(files, f"{TOP}/dataset/Economy.csv",
             os.path.join(C.HERE, "dataset", "Economy.csv"))
    # EXP-020 sort provenance evidence
    for rel in ("sort_mapping.csv", "data_prep.json", "split_report.json"):
        add_file(files, f"{TOP}/results/base_evidence/{rel}",
                 os.path.join(C.RESULTS_DIR, "base_evidence", rel))
    if stage == "final":
        for h in P.DATASETS["Economy"]["horizons"]:
            n = f"Economy__h{h}"
            add_file(files, f"{TOP}/predictions/{n}.npz",
                     os.path.join(C.PRED_DIR, f"{n}.npz"))
            add_file(files, f"{TOP}/predictions/{n}.done.json",
                     os.path.join(C.PRED_DIR, f"{n}.done.json"))
            add_file(files, f"{TOP}/logs/{n}.log",
                     os.path.join(C.LOGS_DIR, f"{n}.log"))
        # anchor experiment files (npz identity/comparison + done records
        # + frozen protocols), never modified
        for arc, src in REFERENCE_EVIDENCE:
            add_file(files, f"{TOP}/{arc}", src)
    for rel in ("logs/pip_freeze_diag_vts_env.txt",
                "logs/env_diff.txt"):
        add_file(files, f"{TOP}/{rel}", os.path.join(C.HERE, rel),
                 must=False)
    return files


def gather_val_evidence():
    files = []
    add_file(files, f"{TOP}/results/selected_p.json",
             os.path.join(C.RESULTS_DIR, "selected_p.json"))
    ev_dir = os.path.join(C.RESULTS_DIR, "val_evidence")
    n = 0
    for p_cand in P.DATASETS["Economy"]["candidates"]:
        rel = f"Economy__P{p_cand}.npz"
        add_file(files, f"{TOP}/results/val_evidence/{rel}",
                 os.path.join(ev_dir, rel))
        n += 1
    if n != 4:
        sys.exit(f"[audit-pack] HARD STOP: expected 4 val evidence npz, "
                 f"found {n}")
    return files


def build(stage):
    os.makedirs(PACK_OUT, exist_ok=True)
    packs = []
    for kind, files in (("final" if stage == "final" else "preflight",
                         gather_main(stage)),
                        ("val_evidence", gather_val_evidence())):
        manifest, total, bufs = [], 0, {}
        for arc, src in files:
            with open(src, "rb") as f:
                buf = f.read()
            scan_redaction(buf, arc)
            bufs[arc] = buf
            total += len(buf)
        if total > MAX_BYTES:
            sys.exit(f"[audit-pack] {kind} payload {total} > {MAX_BYTES}")
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        name = f"Aurora_TimeMMD_visionts_economy_chrono_{kind}_{stamp}.tar.gz"
        path = os.path.join(PACK_OUT, name)
        with tarfile.open(path, "w:gz") as tf:
            for arc, buf in bufs.items():
                ti = tarfile.TarInfo(arc)
                ti.size = len(buf)
                ti.mtime = int(time.time())
                tf.addfile(ti, io.BytesIO(buf))
                manifest.append(f"{sha256_bytes(buf)}  {arc}")
            # sources manifest for the anchor evidence (absolute-source
            # sha256, verifying the packed copies are the frozen files)
            src_manifest = []
            for arc, src in (REFERENCE_EVIDENCE
                             + [(f"{TOP}/dataset/Economy.csv",
                                 os.path.join(C.HERE, "dataset",
                                              "Economy.csv"))]
                             + [(f"{TOP}/results/base_evidence/{rel}",
                                 os.path.join(C.RESULTS_DIR,
                                              "base_evidence", rel))
                                for rel in ("sort_mapping.csv",
                                            "data_prep.json",
                                            "split_report.json")]):
                packed_arc = arc if arc.startswith(TOP + "/") \
                    else f"{TOP}/{arc}"
                if packed_arc in bufs:
                    src_manifest.append(f"{sha256_file(src)}  {packed_arc}"
                                        f" <- {src}")
            sm = ("\n".join(src_manifest) + "\n").encode()
            ti = tarfile.TarInfo(f"{TOP}/reference_evidence/"
                                 f"sources.sha256")
            ti.size = len(sm)
            ti.mtime = int(time.time())
            tf.addfile(ti, io.BytesIO(sm))
            mb = ("\n".join(manifest) + "\n").encode()
            ti = tarfile.TarInfo(f"{TOP}/MANIFEST.sha256")
            ti.size = len(mb)
            ti.mtime = int(time.time())
            tf.addfile(ti, io.BytesIO(mb))
        out_sha = sha256_file(path)
        sz = os.path.getsize(path)
        if sz > MAX_BYTES:
            sys.exit(f"[audit-pack] {kind} tar exceeds 50MB")
        packs.append((path, sz, out_sha, len(bufs) + 2))
        print(f"[audit-pack] {path}")
        print(f"[audit-pack]   files={len(bufs)+2} bytes={sz} "
              f"sha256={out_sha}")
    return packs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["preflight", "final"],
                    required=True)
    a = ap.parse_args()
    if not os.path.isfile(C.PROTO_PATH):
        sys.exit("[audit-pack] protocol.json missing — freeze first")
    build(a.stage)
    print("AUDIT_PACK OK")
