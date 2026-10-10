#!/usr/bin/env python3
"""Freeze the VisionTS EXP-016 reference into visionts_ref/ (local step,
before protocol freeze and before verify_visionts_ref.py / any GPU work).

Copies from the archived EXP-016 directory (commit 0932a5ac..., protocol
token a4c01683793a):
  - predictions/: all 67 shards x (npz + header.json + done) — the frozen
    VisionTS predictions this experiment is compared against;
  - results/: {ds}_metrics.json x3, sequence_level_{ds}.csv x3,
    summary.json, protocol.json (EXP-016 frozen protocol, provenance).
Writes visionts_ref/MANIFEST.json {relpath: sha256} — protocol.py freeze
embeds this manifest and re-verifies every sha at every run. Run order:
freeze_ref.py -> protocol.py freeze -> verify_visionts_ref.py -> GPU."""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for p in (REPO_ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import common as C
import protocol as P

EXPECTED_NPZ = {"PTF": 23, "LEU": 23, "MSPG": 21}


def main():
    src = P.VTS_EXP016_DIR
    if not os.path.isdir(src):
        sys.exit(f"EXP-016 directory not found: {src} "
                 f"(set VTS_EXP016_DIR to override)")
    src_proto = os.path.join(src, "protocol.json")
    tok = C.sha256_file(src_proto)[:12]
    if tok != P.VTS_EXP016_TOKEN:
        sys.exit(f"HARD STOP: EXP-016 protocol token {tok} != "
                 f"{P.VTS_EXP016_TOKEN} (wrong or drifted source)")
    src_commit = open(os.path.join(src, "COMMIT.txt")).read().strip() \
        if os.path.isfile(os.path.join(src, "COMMIT.txt")) else "unknown"
    if src_commit != P.VTS_EXP016_CODE_COMMIT:
        sys.exit(f"HARD STOP: EXP-016 COMMIT.txt {src_commit!r} != "
                 f"{P.VTS_EXP016_CODE_COMMIT!r} (frozen code commit)")
    # soft provenance: the archive commit containing the frozen tree
    try:
        import subprocess
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=src,
            capture_output=True, text=True).stdout.strip()
        if head != P.VTS_EXP016_ARCHIVE_COMMIT:
            print(f"[freeze_ref] WARNING: EXP-016 repo HEAD {head[:12]} != "
                  f"archive pin {P.VTS_EXP016_ARCHIVE_COMMIT[:12]}")
    except Exception:
        pass

    pred_dst = os.path.join(P.VTS_REF_DIR, "predictions")
    res_dst = os.path.join(P.VTS_REF_DIR, "results")
    os.makedirs(pred_dst, exist_ok=True)
    os.makedirs(res_dst, exist_ok=True)

    files = []
    for ds in P.ORDER:
        n = 0
        for shard in range(1000):
            base = os.path.join(src, "predictions",
                                f"{ds}__shard{shard:03d}")
            if not os.path.isfile(base + ".npz"):
                break
            for ext in (".npz", ".header.json", ".done"):
                files.append((os.path.join(src, "predictions",
                                           f"{ds}__shard{shard:03d}{ext}"),
                              os.path.join("predictions",
                                           f"{ds}__shard{shard:03d}{ext}")))
            n += 1
        if n != EXPECTED_NPZ[ds]:
            sys.exit(f"{ds}: found {n} EXP-016 shards != {EXPECTED_NPZ[ds]}")
    for name in (["PTF_metrics.json", "LEU_metrics.json",
                  "MSPG_metrics.json"]
                 + [f"sequence_level_{ds}.csv" for ds in P.ORDER]
                 + ["summary.json"]):
        files.append((os.path.join(src, "results", name),
                      os.path.join("results", name)))
    files.append((src_proto, "protocol.json"))

    for src_path, rel in files:
        dst = os.path.join(P.VTS_REF_DIR, rel)
        if os.path.abspath(src_path) != os.path.abspath(dst):
            shutil.copy2(src_path, dst)
    manifest = {
        "source": src,
        "exp016_code_commit": src_commit,
        "exp016_archive_commit": P.VTS_EXP016_ARCHIVE_COMMIT,
        "exp016_token": P.VTS_EXP016_TOKEN,
        "frozen": C.load_json(src_proto).get("finished", "unknown"),
        "files": {rel: C.sha256_file(os.path.join(P.VTS_REF_DIR, rel))
                  for _s, rel in files},
    }
    C.dump_json(manifest, P.VTS_REF_MANIFEST_PATH)
    print(f"froze {len(files)} files -> {P.VTS_REF_DIR}")
    print(f"manifest -> {P.VTS_REF_MANIFEST_PATH}")
    print("next: python protocol.py freeze && python verify_visionts_ref.py")


if __name__ == "__main__":
    main()
