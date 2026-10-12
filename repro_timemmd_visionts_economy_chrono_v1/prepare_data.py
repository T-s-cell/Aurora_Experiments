#!/usr/bin/env python3
"""Stage 1 (EXP-022): copy the EXP-020 ascending Economy.csv byte-identically
from the frozen EXP-020 base into dataset/, plus the sort provenance
evidence (sort_mapping.csv, data_prep.json, split_report.json) into
results/base_evidence/. Gates: ascending/original/sort-mapping sha256 ==
EXP-020 protocol pins, row count, finite OT, parseable dates, ascending
order (first 1989-01-01, last 2024-03-01). Writes results/data_prep.json."""
import datetime
import os
import shutil
import sys

import common as C
import protocol as P

BASE_DIR = "/home/wlt/MMTS/Aurora_Experiments/repro_timemmd_economy_chrono_v1"
RAW_BASE_DIR = "/home/wlt/MMTS/Aurora_Experiments/repro_timemmd_zeroshot_v1"


def main():
    try:
        import sklearn  # noqa: F401
    except Exception as e:
        sys.exit(f"[prepare] HARD STOP: scikit-learn unavailable in this "
                 f"env ({e!r}) — the StandardScaler replication requires "
                 f"real sklearn, not a hand-rolled mean/std")

    spec = P.DATASETS["Economy"]

    # EXP-020 protocol file gate (data-protocol anchor)
    base_proto_path = os.path.join(BASE_DIR, "protocol.json")
    base_proto = C.load_json(base_proto_path)
    if C.sha256_file(base_proto_path) != P.BASE_DATA["protocol_sha256"]:
        sys.exit("[prepare] HARD STOP: EXP-020 protocol.json file sha256 "
                 "differs from pinned value")

    os.makedirs(os.path.join(C.HERE, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(C.RESULTS_DIR, "base_evidence"), exist_ok=True)

    # 1. ascending CSV copy + sha gates
    src = os.path.join(BASE_DIR, "dataset", "Economy.csv")
    dst = os.path.join(C.HERE, "dataset", "Economy.csv")
    if not os.path.isfile(src):
        sys.exit(f"[prepare] HARD STOP: base ascending CSV missing: {src}")
    if C.sha256_file(src) != spec["csv_sha256"]:
        sys.exit("[prepare] HARD STOP: base ascending CSV sha256 != pin")
    if base_proto["datasets"]["Economy"]["csv_sha256"] != spec["csv_sha256"]:
        sys.exit("[prepare] HARD STOP: EXP-020 protocol csv pin differs "
                 "from this protocol's pin")
    shutil.copyfile(src, dst)
    if C.sha256_file(dst) != spec["csv_sha256"]:
        sys.exit("[prepare] HARD STOP: copied ascending CSV sha256 != pin")

    # 2. provenance evidence copies (EXP-020 sort mapping + reports)
    evidence = {}
    for rel, want_key in (
            ("results/sort_mapping.csv", "sort_mapping_sha256"),
            ("results/data_prep.json", None),
            ("results/split_report.json", None)):
        s = os.path.join(BASE_DIR, rel)
        d = os.path.join(C.RESULTS_DIR, "base_evidence",
                         os.path.basename(rel))
        if not os.path.isfile(s):
            sys.exit(f"[prepare] HARD STOP: base evidence missing: {s}")
        shutil.copyfile(s, d)
        evidence[os.path.basename(rel)] = C.sha256_file(d)
    if evidence["sort_mapping.csv"] != spec["sort_mapping_sha256"]:
        sys.exit("[prepare] HARD STOP: sort_mapping.csv sha256 != "
                 "EXP-020 protocol pin")

    # 3. original (raw-row-order) CSV gate against the EXP-019 base
    orig = os.path.join(RAW_BASE_DIR, "dataset", "Economy.csv")
    if not os.path.isfile(orig):
        sys.exit(f"[prepare] HARD STOP: original CSV missing: {orig}")
    if C.sha256_file(orig) != spec["original_csv_sha256"]:
        sys.exit("[prepare] HARD STOP: original EXP-019 Economy.csv sha256 "
                 "!= pinned original_csv_sha256")

    # 4. parse the ascending copy and gate boundaries
    from data import load_csv
    dates, ot = load_csv("Economy")
    n = len(ot)
    if n != 423:
        sys.exit(f"[prepare] HARD STOP: Economy rows {n} != 423")
    try:
        parsed = [datetime.date.fromisoformat(d) for d in dates]
    except ValueError as e:
        sys.exit(f"[prepare] HARD STOP: Economy unparseable date: {e!r}")
    if parsed != sorted(parsed):
        sys.exit("[prepare] HARD STOP: Economy dates not ascending")
    oa = spec["order_assert"]
    if dates[0] != oa["first_date"] or dates[-1] != oa["last_date"]:
        sys.exit(f"[prepare] HARD STOP: Economy order assert failed: "
                 f"first={dates[0]} last={dates[-1]} (expected ascending "
                 f"{oa['first_date']} -> {oa['last_date']})")
    n_train = int(n * 0.7)
    n_test = int(n * 0.2)
    val_end = n - n_test
    out = {
        "exp_id": P.EXP_ID,
        "base_dir": BASE_DIR,
        "csv_sha256": spec["csv_sha256"],
        "original_csv_sha256": spec["original_csv_sha256"],
        "sort_mapping_sha256": spec["sort_mapping_sha256"],
        "base_evidence_sha256": evidence,
        "order_assert": ("Economy whole-row stable ascending sort "
                         "(first=1989-01-01, last=2024-03-01), per the "
                         "frozen EXP-020 base protocol"),
        "datasets": {
            "Economy": {
                "n_rows": n,
                "n_train": n_train, "num_val": val_end - n_train,
                "n_test": n_test, "val_end": val_end,
                "dates": {"first": dates[0], "last": dates[-1],
                          "val_end": dates[val_end],
                          "test_first": dates[val_end],
                          "test_last": dates[-1],
                          "val_first": dates[n_train]},
            },
        },
    }
    C.dump_json(out, os.path.join(C.RESULTS_DIR, "data_prep.json"))
    print(f"[prepare] Economy: rows={n} split={n_train}/"
          f"{val_end - n_train}/{n_test} dates {dates[0]}..{dates[-1]} "
          f"sha OK (ascending + original + sort_mapping)")
    print("[prepare] data_prep.json written")


if __name__ == "__main__":
    main()
