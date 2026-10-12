#!/usr/bin/env python3
"""EXP-020 data preparation: sort Economy.csv chronologically.

Reads the frozen EXP-019 original Economy.csv (sha256 verified against
the EXP-019 pin), sorts whole rows stable-ascending by the parsed
`date` column, writes this experiment's dataset/Economy.csv plus
results/sort_mapping.csv (new_row_index,original_row_index,date; 0-based).

Validation (all must pass before results/data_prep.json is written):
  - 423 rows; dates parse, are unique, strictly increasing after sort;
  - OT finite everywhere;
  - the written CSV is RE-READ and the inverse permutation maps every
    row cell-for-cell back to the original rows (fact / missing values
    / preds included);
  - sha256 of original / sorted / mapping recorded.
"""
import csv
import math
import os
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C
import protocol as P

N_EXPECT = 423


def read_rows(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = list(r)
    return header, rows


def main():
    src = os.path.join(P.BASE_DIR, "dataset", "Economy.csv")
    src_sha = C.sha256_file(src)
    if src_sha != P.ORIGINAL_CSV_SHA256:
        sys.exit(f"[prepare] source CSV sha mismatch: {src_sha[:12]} != "
                 f"EXP-019 pin {P.ORIGINAL_CSV_SHA256[:12]}")
    header, orig = read_rows(src)
    di = header.index("date")
    oi = header.index("OT")
    if len(orig) != N_EXPECT:
        sys.exit(f"[prepare] {len(orig)} rows != {N_EXPECT}")
    ds = []
    for i, row in enumerate(orig):
        try:
            d = date.fromisoformat(row[di])
        except ValueError:
            sys.exit(f"[prepare] row {i}: unparseable date {row[di]!r}")
        ds.append(d)
        try:
            v = float(row[oi])
            if not math.isfinite(v):
                raise ValueError
        except ValueError:
            sys.exit(f"[prepare] row {i}: OT not finite: {row[oi]!r}")
    if len(set(ds)) != len(ds):
        sys.exit("[prepare] duplicate dates")
    order = sorted(range(len(orig)), key=lambda i: (ds[i], i))
    # new_row j comes from orig row order[j]
    sorted_rows = [orig[j] for j in order]
    sorted_dates = [ds[j] for j in order]
    if any(b <= a for a, b in zip(sorted_dates, sorted_dates[1:])):
        sys.exit("[prepare] sorted dates not strictly increasing")

    os.makedirs(C.DATASET_DIR, exist_ok=True)
    dst = P.csv_path("Economy")
    with open(dst, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(sorted_rows)

    mapping_path = os.path.join(C.RESULTS_DIR, "sort_mapping.csv")
    with open(mapping_path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["new_row_index", "original_row_index", "date"])
        for j, (si, sd) in enumerate(zip(order, sorted_dates)):
            w.writerow([j, si, sd.isoformat()])

    # RE-READ the written CSV and verify inverse permutation cell-for-cell
    back_header, back = read_rows(dst)
    if back_header != header:
        sys.exit("[prepare] header changed by round-trip")
    if len(back) != N_EXPECT:
        sys.exit(f"[prepare] re-read row count {len(back)} != {N_EXPECT}")
    for j in range(N_EXPECT):
        if back[j] != orig[order[j]]:
            k = next(c for c in range(len(header))
                     if back[j][c] != orig[order[j]][c])
            sys.exit(f"[prepare] inverse-perm mismatch at new row {j} "
                     f"(orig {order[j]}, col {header[c]!r})")
    back_dates = [date.fromisoformat(r[di]) for r in back]
    if any(b <= a for a, b in zip(back_dates, back_dates[1:])):
        sys.exit("[prepare] re-read CSV not strictly increasing")

    rep = {
        "ok": True,
        "exp_id": P.EXP_ID,
        "sort_rule": P.SORT_RULE,
        "n_rows": N_EXPECT,
        "columns": header,
        "original_csv_sha256": src_sha,
        "sorted_csv_sha256": C.sha256_file(dst),
        "sort_mapping_sha256": C.sha256_file(mapping_path),
        "first_date": sorted_dates[0].isoformat(),
        "last_date": sorted_dates[-1].isoformat(),
        "inverse_perm_check": "re-read CSV row[j] == original row[order[j]] "
                              "cell-for-cell: PASS",
    }
    C.dump_json(rep, os.path.join(C.RESULTS_DIR, "data_prep.json"))
    print(f"[prepare] sorted {N_EXPECT} rows by date: "
          f"{rep['first_date']} .. {rep['last_date']}")
    print(f"[prepare] sorted csv sha256 {rep['sorted_csv_sha256']}")
    print(f"[prepare] mapping sha256 {rep['sort_mapping_sha256']}")
    print("[prepare] inverse-perm round-trip check: PASS")
    print("[prepare] RESULT: PASS")


if __name__ == "__main__":
    main()
