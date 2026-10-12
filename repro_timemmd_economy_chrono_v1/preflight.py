#!/usr/bin/env python3
"""Preflight for repro_timemmd_economy_chrono_v1 (run on dt BEFORE freeze).

  1. data gate   : data_prep.json present/ok (sorted CSV sha + original
                   CSV sha == EXP-019 pin); 423 rows -> window counts
                   79/77/75/73 == protocol pin; OT finite; fact column
                   present; chronological order assert (first/last Month
                   AND date); SPLIT_CLAIMS date boundaries asserted on
                   the sorted CSV.
  2. batch probe : ONE batch at the max horizon (H=12) through the
                   official path; on CUDA OOM walk the ladder down. The
                   winning batch is written to results/batch_decision.json
                   BEFORE protocol freeze.

There is NO sandbox mini group in this experiment: H=12 is the first
official group and its full gate suite (save_group_meta) validates the
pipeline before H=6/8/10 start. The watcher chains: freeze -> run_all
--only Economy:12 -> run_all (rest) -> evaluate -> audit pack.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C
import protocol as P


def read_csv_rows(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = list(r)
    return header, rows


def data_gate():
    bad = []
    counts = {}
    miss = {}
    try:
        dp = C.load_json(os.path.join(C.RESULTS_DIR, "data_prep.json"))
    except FileNotFoundError:
        dp = None
        bad.append("results/data_prep.json missing — run prepare_data.py")
    for ds, cfg in P.DATASETS.items():
        header, rows = read_csv_rows(P.csv_path(ds))
        col = {name: i for i, name in enumerate(header)}
        for need in ("fact", "OT", "date", "prior_history_avg",
                     "start_date", "end_date"):
            if need not in col:
                bad.append(f"{ds}: column {need} missing")
        n = len(rows)
        counts[ds] = n
        oi = col.get("OT")
        if oi is not None:
            for i, row in enumerate(rows):
                try:
                    v = float(row[oi])
                    if not math.isfinite(v):
                        raise ValueError
                except Exception:
                    bad.append(f"{ds}: OT row {i} not finite: {row[oi]!r}")
                    break
        fi = col.get("fact")
        miss[ds] = sum(1 for row in rows if not row[fi].strip())
        if ds == "Economy":
            oa = cfg["order_assert"]
            first, last = rows[0], rows[-1]
            if first[col["Month"]] != oa["Month"]:
                bad.append(f"Economy first Month={first[col['Month']]!r} "
                           f"!= {oa['Month']!r}")
            if first[col["date"]] != oa["date"]:
                bad.append(f"Economy first date={first[col['date']]!r} "
                           f"!= {oa['date']!r}")
            if last[col["Month"]] != oa["last_Month"]:
                bad.append(f"Economy last Month={last[col['Month']]!r} "
                           f"!= {oa['last_Month']!r}")
            if last[col["date"]] != oa["last_date"]:
                bad.append(f"Economy last date={last[col['date']]!r} "
                           f"!= {oa['last_date']!r}")
            # split claims on the sorted CSV
            sc = P.SPLIT_CLAIMS
            if n != sc["n_rows"]:
                bad.append(f"Economy rows {n} != claim {sc['n_rows']}")
            else:
                for i, want in ((0, sc["train_dates"][0]),
                                (sc["num_train"] - 1, sc["train_dates"][1]),
                                (sc["num_train"], sc["val_dates"][0]),
                                (sc["num_train"] + sc["num_val"] - 1,
                                 sc["val_dates"][1]),
                                (sc["num_train"] + sc["num_val"],
                                 sc["test_dates"][0]),
                                (n - 1, sc["test_dates"][1])):
                    got_d = rows[i][col["date"]]
                    if got_d != want:
                        bad.append(f"Economy row {i} date={got_d!r} != "
                                   f"claim {want!r}")
        for h in cfg["horizons"]:
            want = cfg["windows"][h]
            got = C.expected_windows(n, h)
            if got != want:
                bad.append(f"{ds} H={h}: windows {got} != expected {want}")
    if dp is not None:
        got_sha = C.sha256_file(P.csv_path("Economy"))
        if got_sha != dp.get("sorted_csv_sha256"):
            bad.append("Economy csv sha drift vs data_prep.json")
    rep = {"rows": counts, "missing_fact_counts": miss,
           "data_prep_ok": bool(dp and dp.get("ok")),
           "ok": not bad, "bad": bad[:50]}
    print(f"[preflight] data gate: {'PASS' if not bad else bad[:10]}")
    return rep


def probe_domain(ds, tag):
    """Walk the batch ladder; return the winning batch and probe report."""
    cfg = P.DATASETS[ds]
    h_max = max(cfg["horizons"])
    hb = os.path.join(C.RESULTS_DIR, f"probe_{tag}.json")
    tried = []
    for batch in P.BATCH_LADDER:
        r = subprocess.run(
            [sys.executable, os.path.join(HERE, "probe_one_batch.py"),
             "--ds", ds, "--pred-len", str(h_max), "--batch", str(batch),
             "--hb", hb],
            cwd=HERE, capture_output=True, text=True)
        if r.returncode == 0:
            rep = json.load(open(hb))
            tried.append({"batch": batch, "ok": True, "report": rep})
            print(f"[preflight] probe {ds} H={h_max} batch={batch}: OK "
                  f"peak={rep['peak_cuda_gib']}GiB wall={rep['wall_s']}s")
            return batch, tried
        tail = (r.stdout + r.stderr)[-400:]
        oom = ("OutOfMemoryError" in tail or "CUDA out of memory" in tail
               or "NotEnoughMemory" in tail)
        tried.append({"batch": batch, "ok": False,
                      "oom": bool(oom), "tail": tail[-200:]})
        print(f"[preflight] probe {ds} H={h_max} batch={batch}: FAIL "
              f"(oom={oom})")
        if not oom:
            break  # non-OOM failure: stop climbing down, report
    return None, tried


def batch_probes():
    plan = {}
    reports = {}
    for ds in P.DOMAIN_ORDER:
        batch, tried = probe_domain(ds, ds)
        if batch is None:
            sys.exit(f"[preflight] no feasible batch for {ds} -> HARD STOP")
        plan[ds] = batch
        reports[ds] = tried
    C.dump_json({"ok": True, "batch_per_domain": plan,
                 "probes": reports, "probe_horizon_rule": "max horizon of domain",
                 "ladder": list(P.BATCH_LADDER)},
                C.BATCH_DECISION_PATH)
    print(f"[preflight] batch_decision: {plan}")
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gpu", action="store_true")
    a = ap.parse_args()

    rep = {"stages": {}}
    rep["stages"]["data_gate"] = data_gate()
    if not rep["stages"]["data_gate"]["ok"]:
        C.dump_json(rep, os.path.join(C.RESULTS_DIR, "preflight_report.json"))
        sys.exit("[preflight] RESULT: FAIL (data gate)")

    if a.skip_gpu:
        rep["ok"] = False
        rep["note"] = "skipped GPU stages"
        C.dump_json(rep, os.path.join(C.RESULTS_DIR, "preflight_report.json"))
        sys.exit(0)

    plan = batch_probes()
    rep["stages"]["batch_probes"] = {"plan": plan}

    check = subprocess.run([sys.executable, os.path.join(HERE, "protocol.py"),
                            "check_official"], cwd=HERE)
    if check.returncode != 0:
        sys.exit("[preflight] official md5 changed -> HARD STOP")

    # protocol.freeze() re-validates data_prep.json, batch_decision.json,
    # the official md5 set and the base EXP-019 protocol token
    fr = subprocess.run([sys.executable, os.path.join(HERE, "protocol.py"),
                         "freeze"], cwd=HERE, capture_output=True, text=True)
    if fr.returncode != 0:
        sys.exit(f"[preflight] freeze failed:\n{fr.stdout}{fr.stderr}")
    token = fr.stdout.strip().splitlines()[-1].split()[-1]
    print(f"[preflight] frozen token: {token}")

    rep["ok"] = True
    rep["protocol_token"] = token
    C.dump_json(rep, os.path.join(C.RESULTS_DIR, "preflight_report.json"))
    print(f"[preflight] RESULT: PASS (token {token}; next: first official "
          f"group Economy:12)")


if __name__ == "__main__":
    main()
