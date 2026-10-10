#!/usr/bin/env python3
"""Verify the frozen VisionTS reference (visionts_ref/) BEFORE any GPU work
(user ruling 2026-10-11: runs right after freeze_ref.py, ahead of
preflight). This round's scoring code (evaluate.ladder, identical code
path to EXP-016) recomputes the subset and per-sequence metrics from the
frozen EXP-016 prediction shards, then compares against EXP-016's frozen
results ({ds}_metrics.json + sequence_level_{ds}.csv) at the DOUBLE
tolerance rtol=atol=1e-12: an entry passes iff
    |got - ref| <= atol + rtol * |ref|.
Also re-verifies: EXP-016 protocol.json approval token
(a4c01683793a), the archived commit pin, and every manifest sha256.
Nonzero exit on ANY failure. Writes results/visionts_ref_check.json."""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for p in (REPO_ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import common as C
import protocol as P
from evaluate import ladder, load_vts_shards

RTOL = P.VTS_REF_RTOL
ATOL = P.VTS_REF_ATOL


def close(got, ref):
    got, ref = float(got), float(ref)
    ad = abs(got - ref)
    rd = ad / abs(ref) if ref != 0 else ad
    return (ad <= ATOL + RTOL * abs(ref)), ad, rd


def main():
    man = C.load_json(P.VTS_REF_MANIFEST_PATH)
    if man.get("exp016_code_commit") != P.VTS_EXP016_CODE_COMMIT:
        sys.exit(f"HARD STOP: visionts_ref code-commit pin "
                 f"{man.get('exp016_code_commit')!r} != "
                 f"{P.VTS_EXP016_CODE_COMMIT!r}")
    if man.get("exp016_archive_commit") != P.VTS_EXP016_ARCHIVE_COMMIT:
        sys.exit(f"HARD STOP: visionts_ref archive-commit pin "
                 f"{man.get('exp016_archive_commit')!r} != "
                 f"{P.VTS_EXP016_ARCHIVE_COMMIT!r}")
    proto_ref = C.load_json(os.path.join(P.VTS_REF_DIR, "protocol.json"))
    tok_ok = proto_ref.get("protocol_name") == "cgtsf_zeroshot_v1"
    proto_sha = C.sha256_file(os.path.join(P.VTS_REF_DIR, "protocol.json"))
    if proto_sha[:12] != P.VTS_EXP016_TOKEN:
        sys.exit(f"HARD STOP: visionts_ref protocol.json token "
                 f"{proto_sha[:12]} != {P.VTS_EXP016_TOKEN}")
    bad_files = [rel for rel, sha in man.get("files", {}).items()
                 if not os.path.isfile(os.path.join(P.VTS_REF_DIR, rel))
                 or C.sha256_file(os.path.join(P.VTS_REF_DIR, rel)) != sha]
    if bad_files:
        sys.exit(f"HARD STOP: visionts_ref manifest sha mismatches: "
                 f"{bad_files[:10]} ({len(bad_files)})")

    rep = {"rtol": RTOL, "atol": ATOL,
           "exp016_code_commit": man.get("exp016_code_commit"),
           "exp016_archive_commit": man.get("exp016_archive_commit"),
           "exp016_token_ok": bool(tok_ok),
           "manifest_files": len(man.get("files", {})),
           "manifest_sha_ok": True,
           "subsets": {}}
    all_ok, max_ad, max_rd = True, 0.0, 0.0
    for ds in P.ORDER:
        rows = load_vts_shards(ds)
        if len(rows) != P.EXPECTED_WINDOWS[ds]:
            sys.exit(f"{ds}: visionts_ref rows {len(rows)} != "
                     f"{P.EXPECTED_WINDOWS[ds]}")
        mse, mae, _point, seqs = ladder(rows)
        ref = C.load_json(os.path.join(P.VTS_REF_DIR, "results",
                                       f"{ds}_metrics.json"))
        with open(os.path.join(P.VTS_REF_DIR, "results",
                               f"sequence_level_{ds}.csv")) as f:
            r = list(csv.reader(f))[1:]
        ref_seq = {int(row[0]): (int(row[1]), float(row[2]), float(row[3]))
                   for row in r}
        seq_bad, seq_max_ad = 0, 0.0
        for ix, n, m, a in seqs:
            if ref_seq[ix][0] != n:
                seq_bad += 1
                continue
            for got, expect in ((m, ref_seq[ix][1]), (a, ref_seq[ix][2])):
                ok, ad, rd = close(got, expect)
                seq_max_ad = max(seq_max_ad, ad)
                max_rd = max(max_rd, rd)
                if not ok:
                    seq_bad += 1
        m_ok, m_ad, m_rd = close(mse, ref["mse_seq_equal_weight"])
        a_ok, a_ad, a_rd = close(mae, ref["mae_seq_equal_weight"])
        ok = (m_ok and a_ok and seq_bad == 0
              and len(seqs) == ref["n_sequences"]
              and len(rows) == ref["n_windows"])
        all_ok &= ok
        max_ad = max(max_ad, m_ad, a_ad, seq_max_ad)
        max_rd = max(max_rd, m_rd, a_rd)
        rep["subsets"][ds] = {
            "mse": {"got": mse, "ref": ref["mse_seq_equal_weight"],
                    "abs_diff": m_ad, "rel_diff": m_rd, "ok": bool(m_ok)},
            "mae": {"got": mae, "ref": ref["mae_seq_equal_weight"],
                    "abs_diff": a_ad, "rel_diff": a_rd, "ok": bool(a_ok)},
            "sequence_entries_mismatched": seq_bad,
            "max_seq_abs_diff": seq_max_ad,
            "n_sequences": len(seqs), "n_windows": len(rows),
            "ok": bool(ok),
        }
        print(f"[vts-ref] {ds}: dMSE_abs={m_ad:.3e} dMAE_abs={a_ad:.3e} "
              f"seq_mismatch={seq_bad} -> {'OK' if ok else 'FAIL'}")
    rep["max_abs_diff_overall"] = max_ad
    rep["max_rel_diff_overall"] = max_rd
    rep["ok"] = bool(all_ok)
    os.makedirs(P.RESULTS_DIR, exist_ok=True)
    C.dump_json(rep, os.path.join(P.RESULTS_DIR, "visionts_ref_check.json"))
    print(f"[vts-ref] max abs diff = {max_ad:.3e}, max rel diff = "
          f"{max_rd:.3e} (double tolerance rtol=atol={ATOL:.0e}) -> "
          f"{'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
