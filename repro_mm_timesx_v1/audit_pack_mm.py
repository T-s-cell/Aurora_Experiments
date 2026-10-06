#!/usr/bin/env python3
"""Build the M48T512 review tar.gz into /home/wlt/MMTS/temp/.

Preflight edition (S6.5 gate): lets a reviewer verify the frozen protocol_mm,
the text cache (token IDs + decoded text + block boundaries), and every
preflight assertion — plus independently re-derive token IDs and re-run the
dry-run plan. No prediction shards exist yet (zero test-window inference has
been performed).

Included: all mm code, the reused root sources it depends on, protocol_mm +
SOURCE_MM_HASHES, the full text cache (npz + meta jsonl + build stats),
preflight evidence JSONs + run log + wrapper + theta dry-run log + env lock,
data protocol artifacts (split_manifest / data_cache / Z0 reference).

Excluded: model weights, data/*.zip, __pycache__, mm_e scratch.
"""
import hashlib
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SUBDIR = Path(__file__).resolve().parent
TEMP = Path("/home/wlt/MMTS/temp")

INCLUDE_MM_FILES = [
    "protocol_mm.json", "SOURCE_MM_HASHES.json",
    "text_builder.py", "text_cache.py", "predict_mm.py", "run_eval_mm.py",
    "aggregate_mm.py", "run_preflight_mm.py", "audit_pack_mm.py",
]
INCLUDE_ROOT_FILES = [
    ".gitignore", "README.md", "protocol.json", "SOURCE_HASHES.json",
    "requirements.txt", "environment_lock.txt", "environment_lock_theta.txt",
    "load_aurora.py", "data_loader.py", "predict.py", "aggregate.py",
    "run_eval.py", "data/split_manifest.json",
    "data/data_cache.npz",
    "data/Z0__test.npz",
]
INCLUDE_MM_GLOBS = [
    "cache/text_tokens_M48T512.npz", "cache/text_meta_M48T512.jsonl",
    "cache/build_stats_M48T512.json",
    "preflight/*.json", "preflight/mm_c_child_pred.npy",
    "logs/preflight_mm.log", "logs/preflight_wrapper_theta.sh",
    "logs/dry_run_mm.log", "logs/environment_lock_theta_mm.txt",
]
INCLUDE_REFERENCE = PROJECT / "reference" / "exp004" / "timesx_data.py"


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
    name = f"Aurora_TimesX_mm_preflight_{stamp}"
    work = SUBDIR / f"_pack_{name}"
    if work.exists():
        shutil.rmtree(work)
    dst_root = work / name
    dst_root.mkdir(parents=True)

    for rel in INCLUDE_MM_FILES:
        src = SUBDIR / rel
        if src.exists():
            dst = dst_root / "repro_mm_timesx_v1" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
    for rel in INCLUDE_ROOT_FILES:
        src = PROJECT / rel
        if src.exists():
            dst = dst_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
    for g in INCLUDE_MM_GLOBS:
        for src in SUBDIR.glob(g):
            dst = dst_root / "repro_mm_timesx_v1" / src.relative_to(SUBDIR)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
    dst = dst_root / "reference" / "exp004" / "timesx_data.py"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(INCLUDE_REFERENCE.read_bytes())

    def _g(*a):
        r = subprocess.run(["git", *a], cwd=PROJECT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else f"(git {' '.join(a)} failed)"
    git_state = "\n".join([
        f"HEAD: {_g('rev-parse', 'HEAD')}",
        f"branch: {_g('rev-parse', '--abbrev-ref', 'HEAD')}",
        f"remote origin: {_g('remote', 'get-url', origin_name())}",
        f"HEAD subject: {_g('log', '-1', '--pretty=%s')}",
        "working tree:",
        _g("status", "--short") or "(clean)",
    ]) + "\n"
    (dst_root / "GIT_STATE.txt").write_text(git_state)

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
    shutil.rmtree(work)
    print(f"[audit_pack_mm] wrote {out} ({out.stat().st_size/1e6:.1f} MB, "
          f"{len(manifest_lines)} files)")


def origin_name():
    r = subprocess.run(["git", "remote"], cwd=PROJECT, capture_output=True, text=True)
    return r.stdout.split()[0] if r.stdout.split() else "origin"


if __name__ == "__main__":
    main()
