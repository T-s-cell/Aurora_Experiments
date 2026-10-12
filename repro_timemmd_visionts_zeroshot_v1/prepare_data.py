#!/usr/bin/env python3
"""Stage 1 (EXP-021): copy the 9 raw-row-order CSVs byte-identically from
the frozen EXP-019 base into dataset/, gate them (sha256 == base pin ==
protocol pin, row counts, finite OT, parseable dates, Economy
year-descending order preserved), write results/data_prep.json."""
import datetime
import os
import shutil
import sys

import common as C
import protocol as P

BASE_DIR = "/home/wlt/MMTS/Aurora_Experiments/repro_timemmd_zeroshot_v1"


def main():
    try:
        import sklearn  # noqa: F401
    except Exception as e:
        sys.exit(f"[prepare] HARD STOP: scikit-learn unavailable in this "
                 f"env ({e!r}) — the StandardScaler replication requires "
                 f"real sklearn, not a hand-rolled mean/std")
    base_proto_path = os.path.join(BASE_DIR, "protocol.json")
    base_proto = C.load_json(base_proto_path)
    if C.sha256_file(base_proto_path) != P.BASE["protocol_sha256"]:
        sys.exit("[prepare] HARD STOP: base protocol.json file sha256 "
                 "differs from pinned EXP-019 value")

    os.makedirs(os.path.join(C.HERE, "dataset"), exist_ok=True)
    out = {"exp_id": P.EXP_ID, "base_dir": BASE_DIR, "datasets": {}}

    for ds in P.DOMAIN_ORDER:
        spec = P.DATASETS[ds]
        src = os.path.join(BASE_DIR, "dataset", f"{ds}.csv")
        dst = os.path.join(C.HERE, "dataset", f"{ds}.csv")
        if not os.path.isfile(src):
            sys.exit(f"[prepare] HARD STOP: base CSV missing: {src}")
        if C.sha256_file(src) != spec["csv_sha256"]:
            sys.exit(f"[prepare] HARD STOP: {ds} base CSV sha256 != pin")
        if base_proto["datasets"][ds]["csv_sha256"] != spec["csv_sha256"]:
            sys.exit(f"[prepare] HARD STOP: {ds} base protocol csv pin "
                     f"differs from this protocol's pin")
        shutil.copyfile(src, dst)
        if C.sha256_file(dst) != spec["csv_sha256"]:
            sys.exit(f"[prepare] HARD STOP: {ds} copied CSV sha256 != pin")

        from data import load_csv
        dates, ot = load_csv(ds)
        n = len(ot)
        if n < spec["seq_len"] + spec["horizons"][-1] + 2:
            sys.exit(f"[prepare] HARD STOP: {ds} too few rows: {n}")
        try:
            for d in (dates[0], dates[-1]):
                datetime.date.fromisoformat(d)
        except ValueError as e:
            sys.exit(f"[prepare] HARD STOP: {ds} unparseable date: {e!r}")
        if ds == "Economy":
            if dates[0] != "2024-01-01" or dates[-1] != "1989-12-01":
                sys.exit(f"[prepare] HARD STOP: Economy order assert "
                         f"failed: first={dates[0]} last={dates[-1]} "
                         f"(expected year-descending blocks 2024-01-01 -> "
                         f"1989-12-01)")
        n_train = int(n * 0.7)
        n_test = int(n * 0.2)
        val_end = n - n_test
        blk = {
            "n_rows": n,
            "n_train": n_train, "num_val": val_end - n_train,
            "n_test": n_test, "val_end": val_end,
            "dates": {"first": dates[0], "last": dates[-1],
                      "val_end": dates[val_end],
                      "test_first": dates[val_end],
                      "test_last": dates[-1],
                      "val_first": dates[n_train]},
        }
        out["datasets"][ds] = blk
        print(f"[prepare] {ds}: rows={n} split={n_train}/"
              f"{val_end - n_train}/{n_test} dates {dates[0]}.."
              f"{dates[-1]} sha OK")

    out["order_assert"] = ("Economy kept original year-descending block "
                           "order (first=2024-01-01, last=1989-12-01)")
    C.dump_json(out, os.path.join(C.RESULTS_DIR, "data_prep.json"))
    print("[prepare] data_prep.json written")


if __name__ == "__main__":
    main()
