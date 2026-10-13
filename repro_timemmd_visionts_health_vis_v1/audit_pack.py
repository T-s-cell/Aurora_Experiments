#!/usr/bin/env python3
"""EXP-023 audit pack: one fixed bundle (<50MB hard gate).

main pack: code + generated README + gates.json + scaler.json +
selection.json + health_window_metrics.csv + series_date_ot.csv (full
1389-row date+OT, user fix #4) + 10 figures + EXP-021 evidence copies
(protocol.json, selected_p.json, comparison_per_h.csv, 4 done.json,
4 frozen npz) + sources.sha256 + MANIFEST.sha256. Weights are never
packed; npz are frozen prediction outputs (~518KB total) for offline
recomputation.
"""
import datetime
import hashlib
import io
import os
import re
import sys
import tarfile
import time

import anchors as A

PACK_OUT = A.PACK_OUT
TOP = A.PROTOCOL_NAME
MAX_BYTES = 50 * 1024 * 1024
HERE = A.HERE

CODE_FILES = [
    "anchors.py", "verify_and_select.py", "plot_windows.py",
    "make_report.py", "audit_pack.py", "run_vis_dt.sh", "wrapper_dt.sh",
    ".gitignore", "README.md",
]

EVIDENCE = (
    [("evidence/exp021_protocol.json", A.PROTO_PATH),
     ("evidence/exp021_selected_p.json", A.SEL_PATH),
     ("evidence/exp021_comparison_per_h.csv", A.PER_H_CSV)]
    + [(f"evidence/exp021_predictions/Health__h{h}.npz", A.npz_path(h))
       for h in A.HORIZONS]
    + [(f"evidence/exp021_predictions/Health__h{h}.done.json",
        A.done_path(h)) for h in A.HORIZONS]
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
WHITELIST = [A.BASE["add_commit"].encode(),
             A.BASE["freeze_commit"].encode(),
             A.BASE["protocol_sha256"].encode(),
             A.BASE["selected_p_sha256"].encode(),
             A.CSV_SHA256.encode()]


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
                    b"protocol_token" in frag
                    or b"cgtsf" in low[s:m.start()]):
                continue
            sys.exit(f"[audit-pack] REDACTION HIT in {name}: "
                     f"{frag[-80:]!r}")


def add_file(files, arc, abs_path, must=True):
    if os.path.isfile(abs_path):
        files.append((arc, abs_path))
    elif must:
        sys.exit(f"[audit-pack] required file missing: {abs_path}")


def gather():
    files = []
    for rel in CODE_FILES:
        add_file(files, f"{TOP}/{rel}", os.path.join(HERE, rel))
    for rel in ("gates.json", "scaler.json", "selection.json",
                "health_window_metrics.csv", "series_date_ot.csv"):
        add_file(files, f"{TOP}/results/{rel}",
                 os.path.join(A.RESULTS_DIR, rel))
    add_file(files, f"{TOP}/figures/plot_manifest.json",
             os.path.join(A.FIG_DIR, "plot_manifest.json"))
    for f in sorted(os.listdir(A.FIG_DIR)):
        if f.endswith(".png"):
            add_file(files, f"{TOP}/figures/{f}",
                     os.path.join(A.FIG_DIR, f))
    n_png = sum(1 for arc, _ in files if arc.endswith(".png"))
    if n_png != 10:
        sys.exit(f"[audit-pack] expected 10 PNGs, found {n_png}")
    for arc, src in EVIDENCE:
        add_file(files, f"{TOP}/{arc}", src)
    for rel in ("logs/pip_freeze_before.txt", "logs/pip_freeze_after.txt",
                "logs/pip_install_matplotlib.log"):
        add_file(files, f"{TOP}/{rel}", os.path.join(HERE, rel),
                 must=False)
    return files


def build():
    os.makedirs(PACK_OUT, exist_ok=True)
    files = gather()
    manifest, src_manifest, bufs, total = [], [], {}, 0
    for arc, src in files:
        with open(src, "rb") as f:
            buf = f.read()
        scan_redaction(buf, arc)
        bufs[arc] = buf
        total += len(buf)
    if total > MAX_BYTES:
        sys.exit(f"[audit-pack] payload {total} > {MAX_BYTES}")
    for arc, src in EVIDENCE + [(f"{TOP}/results/series_date_ot.csv",
                                 os.path.join(A.RESULTS_DIR,
                                              "series_date_ot.csv"))]:
        packed_arc = arc if arc.startswith(TOP + "/") else f"{TOP}/{arc}"
        src_manifest.append(f"{sha256_file(src)}  {packed_arc} <- {src}")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    name = f"Aurora_TimeMMD_visionts_health_vis_final_{stamp}.tar.gz"
    path = os.path.join(PACK_OUT, name)
    with tarfile.open(path, "w:gz") as tf:
        for arc, buf in bufs.items():
            ti = tarfile.TarInfo(arc)
            ti.size = len(buf)
            ti.mtime = int(time.time())
            tf.addfile(ti, io.BytesIO(buf))
            manifest.append(f"{sha256_bytes(buf)}  {arc}")
        for body, arc in ((src_manifest, f"{TOP}/sources.sha256"),
                          (manifest, f"{TOP}/MANIFEST.sha256")):
            b = ("\n".join(body) + "\n").encode()
            ti = tarfile.TarInfo(arc)
            ti.size = len(b)
            ti.mtime = int(time.time())
            tf.addfile(ti, io.BytesIO(b))
    out_sha = sha256_file(path)
    sz = os.path.getsize(path)
    if sz > MAX_BYTES:
        sys.exit(f"[audit-pack] tar exceeds 50MB")
    print(f"[audit-pack] {path}")
    print(f"[audit-pack]   files={len(bufs)+2} bytes={sz} "
          f"sha256={out_sha}")
    print("AUDIT_PACK OK")


if __name__ == "__main__":
    if not os.path.isfile(os.path.join(A.RESULTS_DIR, "gates.json")):
        sys.exit("[audit-pack] results/gates.json missing — run "
                 "verify_and_select.py first")
    build()
