#!/usr/bin/env python3
"""Build the review tar.gz into /home/wlt/MMTS/temp/ with a MANIFEST.sha256.

Included: all code, protocol/provenance JSONs, README, requirements, reference/,
preflight evidence (assertion JSONs + small sample .npy + run logs),
environment_lock.txt, results/ CSVs if present.

Excluded: large data (data/*.zip, data/*.npz), model weights, official-eval
prediction shards (predictions/*.npz), quarantine shards, __pycache__,
resume-drill scratch (its assertions live in preflight/L_resume.json).
"""
import hashlib
import os
import tarfile
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
TEMP = Path("/home/wlt/MMTS/temp")

INCLUDE_FILES = [
    ".gitignore", "README.md", "protocol.json", "SOURCE_HASHES.json",
    "requirements.txt", "environment_lock.txt",
    "load_aurora.py", "data_loader.py", "predict.py", "aggregate.py",
    "verify_aggregate.py", "run_preflight.py", "run_eval.py", "audit_pack.py",
    "data/split_manifest.json",
]
INCLUDE_DIRS = ["reference", "preflight"]
INCLUDE_GLOBS = ["preflight_run*.log", "preflight_final*.log", "logs/preflight_*.log",
                 "results/*.csv", "results/*.json", "results/*.md"]


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    stamp = time.strftime("%Y%m%d")
    name = f"Aurora_TimesX_code_preflight_{stamp}"
    work = PROJECT / f"_pack_{name}"
    if work.exists():
        import shutil
        shutil.rmtree(work)
    (work / name).mkdir(parents=True)
    dst_root = work / name

    for rel in INCLUDE_FILES:
        src = PROJECT / rel
        if src.exists():
            dst = dst_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
    for d in INCLUDE_DIRS:
        src = PROJECT / d
        if not src.exists():
            continue
        dst = dst_root / d
        dst.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copytree(src, dst, ignore=_ignore)
    for g in INCLUDE_GLOBS:
        for src in PROJECT.glob(g):
            dst = dst_root / src.relative_to(PROJECT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())

    # manifest over the payload
    manifest_lines = []
    for p in sorted(dst_root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(dst_root)
            manifest_lines.append(f"{sha256_file(p)}  {rel}")
    (dst_root / "MANIFEST.sha256").write_text("\n".join(manifest_lines) + "\n")

    TEMP.mkdir(exist_ok=True)
    out = TEMP / f"{name}.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        tar.add(dst_root, arcname=name)
    import shutil
    shutil.rmtree(work)
    print(f"[audit_pack] wrote {out} ({out.stat().st_size/1e6:.1f} MB, "
          f"{len(manifest_lines)} files)")


def _ignore(dirpath, names):
    ignored = set()
    for n in names:
        p = Path(dirpath) / n
        if n == "__pycache__" or p.suffix in (".pyc", ".npz"):
            ignored.add(n)
    # drop import-side-effect dirs of vendored exp004
    if str(dirpath).endswith("exp004"):
        ignored |= {"audit", "checkpoints", "dataset", "logs", "manifests", "predictions"}
    return ignored


if __name__ == "__main__":
    main()
