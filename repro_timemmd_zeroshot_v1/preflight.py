#!/usr/bin/env python3
"""Preflight for repro_timemmd_zeroshot_v1 (run on dt BEFORE freeze).

  1. data gate   : 9 CSV row counts -> 36 window counts == protocol pin;
                   OT finite; Economy original-order assert on BOTH the
                   Month column ('January 2024') and the date column
                   ('2024-01-01'); fact column present; missing-text count.
  2. batch probe : per domain, ONE batch at its max horizon through the
                   official path; on CUDA OOM walk the ladder down. The
                   winning per-domain batch is written to
                   results/batch_decision.json BEFORE protocol freeze.
  3. mini group  : Security h12 FULL run (48 windows, one incomplete
                   batch) in a sandbox predictions dir through run_all.py
                   --only (fresh process), exercising save+recompute gates
                   and token_len=24; sandbox results are NOT final output.
  4. summary     : PASS verdict + crude ETA from the mini group.

The full 36-group run is NOT started here; the watcher freezes the
protocol, packs the preflight audit bundle and STOPS for user review.
"""
import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C
import protocol as P

MISSING_TEXT = "No information available"


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
        for h in cfg["horizons"]:
            want = cfg["windows"][h]
            got = C.expected_windows(n, h)
            if got != want:
                bad.append(f"{ds} H={h}: windows {got} != expected {want}")
    rep = {"rows": counts, "missing_fact_counts": miss,
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


def mini_group(token):
    sandbox = os.path.join(HERE, "preflight_tmp", "predictions")
    shutil.rmtree(os.path.join(HERE, "preflight_tmp"), ignore_errors=True)
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "run_all.py"),
         "--i-approve-frozen-protocol", token, "--only", "Security:12",
         "--pred-root", sandbox],
        cwd=HERE, capture_output=True, text=True)
    wall = time.time() - t0
    out = (r.stdout + r.stderr)
    if r.returncode != 0:
        sys.exit(f"[preflight] Security:12 mini group FAILED:\n{out[-1500:]}")
    src_log = os.path.join(C.LOGS_DIR, "group_Security__h12.log")
    keep_log = os.path.join(C.LOGS_DIR, "preflight_security_h12.log")
    if os.path.isfile(src_log):
        shutil.copyfile(src_log, keep_log)
    done = C.load_json(os.path.join(sandbox, "Security__h12.done.json"))
    rep = {"wall_s": round(wall, 1), "windows": done["windows"],
           "batch": done["batch"], "official_printed": done["official_printed"],
           "recompute_float32": done["recompute_float32"],
           "metrics_float64": done["metrics_float64"], "ok": True}
    print(f"[preflight] mini group Security:12 OK in {wall:.0f}s "
          f"(batch {done['batch']}, official mse="
          f"{done['official_printed']['mse']:.6f})")
    return rep


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

    # freeze with the measured batch plan so the mini group runs under the
    # exact frozen config that the full run will use
    fr = subprocess.run([sys.executable, os.path.join(HERE, "protocol.py"),
                         "freeze"], cwd=HERE, capture_output=True, text=True)
    if fr.returncode != 0:
        sys.exit(f"[preflight] freeze failed:\n{fr.stdout}{fr.stderr}")
    token = fr.stdout.strip().splitlines()[-1].split()[-1]
    print(f"[preflight] frozen token: {token}")

    rep["stages"]["mini_group"] = mini_group(token)
    mg = rep["stages"]["mini_group"]
    wps = mg["windows"] / mg["wall_s"]
    eta_s = P.TOTAL_WINDOWS / wps
    rep["eta"] = {"sec_per_window": round(1 / wps, 3),
                  "total_windows": P.TOTAL_WINDOWS,
                  "eta_hours_crude": round(eta_s / 3600, 2),
                  "note": ("ETA from Security h12 (batch "
                           f"{mg['batch']}); other domains differ in "
                           "seq_len/token_len/horizon, treat as order-of-"
                           "magnitude only")}
    print(f"[preflight] crude ETA: {rep['eta']['eta_hours_crude']}h "
          f"({1/wps:.3f}s/window)")

    rep["ok"] = True
    rep["protocol_token"] = token
    C.dump_json(rep, os.path.join(C.RESULTS_DIR, "preflight_report.json"))
    print(f"[preflight] RESULT: PASS (token {token}; AWAITING USER REVIEW "
          f"BEFORE FULL RUN)")


if __name__ == "__main__":
    main()
