#!/usr/bin/env python3
"""Compare two trace output dirs: npz arrays bitwise + jsonl byte equality."""
import sys
from pathlib import Path

import numpy as np

SCHEMES = ("D0", "D1", "D2")


def main(a_dir, b_dir):
    a_dir, b_dir = Path(a_dir), Path(b_dir)
    ok = True
    for s in SCHEMES:
        za = np.load(a_dir / f"trace_{s}.npz", allow_pickle=False)
        zb = np.load(b_dir / f"trace_{s}.npz", allow_pickle=False)
        arrays_eq = all(np.array_equal(za[k], zb[k])
                        for k in ("ids", "mask", "var_keys", "sample_ids"))
        ja = (a_dir / f"trace_{s}.jsonl").read_bytes()
        jb = (b_dir / f"trace_{s}.jsonl").read_bytes()
        jsonl_eq = ja == jb
        print(f"[{s}] npz_arrays_equal={arrays_eq} jsonl_bytes_equal="
              f"{jsonl_eq} (npz {za['ids'].shape[0]} rows, jsonl "
              f"{len(ja)} bytes)")
        ok &= arrays_eq and jsonl_eq
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
