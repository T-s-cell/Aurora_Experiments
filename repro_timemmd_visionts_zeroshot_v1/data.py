#!/usr/bin/env python3
"""Data layer for EXP-021: raw-row-order TimeMMD CSVs (byte-identical to
the EXP-019 base), sklearn StandardScaler fit on the first 70% of OT rows
(official-loader replica — never hand-rolled mean/std), and window
construction identical to EXP-019: test prediction starts at val_end,
history [s-L, s), label [s, s+H), stride 1; validation windows
s in [max(n_train, L), val_end - H] closed.
"""
import csv
import os

import numpy as np
from sklearn.preprocessing import StandardScaler

import protocol as P

HERE = os.path.dirname(os.path.abspath(__file__))


def csv_path(ds):
    return os.path.join(HERE, "dataset", f"{ds}.csv")


def load_csv(ds):
    """stdlib csv only: raw date strings + OT as float64, row order
    preserved exactly as on disk."""
    path = csv_path(ds)
    dates, ot = [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        if "date" not in (r.fieldnames or []) or "OT" not in r.fieldnames:
            raise SystemExit(f"[data] HARD STOP: {path} lacks date/OT "
                             f"columns (header={r.fieldnames})")
        for row in r:
            dates.append(row["date"])
            ot.append(float(row["OT"]))
    if not dates:
        raise SystemExit(f"[data] HARD STOP: {path} has zero rows")
    return dates, np.asarray(ot, dtype=np.float64)


class Dataset:
    """One domain: raw rows, official split, scaler, scaled float32 series.

    scaled32 = float32((OT - mean_) / scale_) with mean_/scale_ from
    sklearn StandardScaler fit on rows [0, n_train) — bitwise-identical
    to the official loader path verified in EXP-020.
    """

    def __init__(self, ds):
        self.ds = ds
        spec = P.DATASETS[ds]
        self.spec = spec
        self.dates, self.ot = load_csv(ds)
        n = len(self.ot)
        self.n_rows = n
        self.n_train = int(n * 0.7)
        self.n_test = int(n * 0.2)
        self.val_end = n - self.n_test
        self.num_val = self.val_end - self.n_train
        declared = spec["split_rows"]
        if (self.n_train, self.num_val, self.n_test) != tuple(declared):
            raise SystemExit(
                f"[data] HARD STOP: {ds} split {self.n_train}/"
                f"{self.num_val}/{self.n_test} != declared {declared}")
        if not np.all(np.isfinite(self.ot)):
            raise SystemExit(f"[data] HARD STOP: {ds} OT has non-finite "
                             f"values")
        self.scaler = StandardScaler()
        self.scaler.fit(self.ot[:self.n_train].reshape(-1, 1))
        self.scaled32 = self.scaler.transform(
            self.ot.reshape(-1, 1)).astype(np.float32).ravel()

    def _starts(self, first, last):
        return list(range(first, last + 1))

    def test_windows(self, H):
        """EXP-019-identical test set: starts at val_end, ends at N-H."""
        L = self.spec["seq_len"]
        stops = self._starts(self.val_end, self.n_rows - H)
        declared = self.spec["windows"][str(H)]
        if len(stops) != declared:
            raise SystemExit(f"[data] HARD STOP: {self.ds}_h{H} test windows "
                             f"{len(stops)} != declared {declared}")
        if stops[0] != self.val_end or stops[-1] + H != self.n_rows:
            raise SystemExit(f"[data] HARD STOP: {self.ds} h{H} test "
                             f"boundaries {stops[0]}..{stops[-1]} violate "
                             f"val_end={self.val_end} N={self.n_rows} H={H}")
        return self._gather(stops, L, H)

    def val_windows(self, H):
        """Validation windows s in [max(n_train, L), val_end - H] closed —
        no test row is ever touched (s + H <= val_end)."""
        L = self.spec["seq_len"]
        first = max(self.n_train, L)
        last = self.val_end - H
        if last < first:
            raise SystemExit(f"[data] HARD STOP: {self.ds} h{H} empty val "
                             f"range [{first},{last}]")
        stops = self._starts(first, last)
        return self._gather(stops, L, H)

    def _gather(self, stops, L, H):
        arr = self.scaled32
        hist = np.stack([arr[s - L:s] for s in stops]).astype(np.float32)
        true = np.stack([arr[s:s + H] for s in stops]).astype(np.float32)
        rows = np.asarray(stops, dtype=np.int64)
        dts = np.asarray([self.dates[s] for s in stops])
        return (hist[..., None], true[..., None], rows, dts)


def val_window_count(ds, H):
    spec = P.DATASETS[ds]
    n = spec["split_rows"][0] + spec["split_rows"][1] + spec["split_rows"][2]
    val_end = n - spec["split_rows"][2]
    first = max(spec["split_rows"][0], spec["seq_len"])
    return val_end - H - first + 1
