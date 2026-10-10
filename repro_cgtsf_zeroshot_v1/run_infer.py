#!/usr/bin/env python3
"""Full inference PTF -> LEU -> MSPG x seeds 2021/2022/2023
(3 x 33,693 = 101,079 windows), sharded + resumable, batch=1.

Resume policy (tiered, per the EXP-016 ruling):
  - stored guard set (protocol sha / code md5 / old-root md5 / package md5 /
    weights sha / csv sha / vts_ref manifest sha) differing from current
    ->  HARD STOP (tier-1: fingerprint mismatch);
  - guards match but the shard npz is missing/corrupt/unfinished
    ->  quarantine the old files (reason + timestamp), recompute the shard,
        log the event (tier-2: shard damage only, not a stop).
A .done marker alone never skips a shard: payload sha256 is re-verified.
Any NaN/Inf in a forward output = immediate hard stop (nothing is written).
Per (ds, seed): in-memory model state digest taken before and after must be
identical; pad/token tables re-asserted. Preflight binding:
results/preflight_report.json must exist with ok=true before anything runs;
the run's batch is pinned to 1 and every shard's guards bind batch + env +
sha256 of the preflight report; the resolved weights file is hash-verified
before model build and again after the run."""
import argparse
import datetime
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for p in (REPO_ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

import common as C
import infer_aurora as IA
import load_aurora as LA
import protocol as P
from load_cgtsf import load_windows


def setup_determinism():
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == P.CUBLAS_WORKSPACE_CONFIG, \
        "CUBLAS_WORKSPACE_CONFIG not set (use wrapper_theta.sh)"
    LA.apply_determinism()
    torch.manual_seed(min(P.SEEDS))
    np.random.seed(min(P.SEEDS))


def quarantine(ds, seed, shard, reason):
    os.makedirs(P.QUARANTINE_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    npz, hdr, done = P.shard_paths(ds, seed, shard)
    tag = f"{ds}__s{seed}__shard{shard:03d}.{ts}"
    for p in (npz, hdr, done):
        if os.path.isfile(p):
            shutil.move(p, os.path.join(P.QUARANTINE_DIR,
                                        os.path.basename(p) + "." + tag))
    entry = {"ds": ds, "seed": seed, "shard": shard, "reason": reason,
             "timestamp": ts}
    C.dump_json(entry, os.path.join(
        P.QUARANTINE_DIR, f"{tag}.reason.json"))
    print(f"[infer] QUARANTINED {ds} s{seed} shard {shard}: {reason}",
          flush=True)
    return entry


def shard_valid(ds, seed, shard, guards):
    npz, hdr, done = P.shard_paths(ds, seed, shard)
    if not os.path.isfile(done):
        return False, "no .done"
    d = C.load_json(done)
    if d.get("guards") != guards:
        sys.exit(f"HARD STOP {ds} s{seed} shard {shard}: stored guards differ "
                 f"from current (protocol/source/package/weight/data "
                 f"fingerprint mismatch). Re-freeze and rerun; do NOT "
                 f"overwrite.")
    if not os.path.isfile(npz):
        return False, ".done present but npz missing"
    if C.sha256_file(npz) != d.get("payload_sha256"):
        return False, "payload sha256 mismatch (corrupt/unfinished)"
    return True, "ok"


def compute_shard(model, windows, device, base_seed, Pl):
    preds = np.zeros((len(windows), Pl), dtype=np.float64)
    ms = np.zeros(len(windows), dtype=np.float64)
    for i, w in enumerate(windows):
        seed_i = C.derive_seed(base_seed, w["ds"], w["idx"], w["date"])
        t0 = time.perf_counter()
        a = IA.predict_window(model, w["hist"], Pl, seed_i)
        torch.cuda.synchronize()
        ms[i] = (time.perf_counter() - t0) * 1000.0
        if not C.all_finite(a):
            sys.exit(f"HARD STOP: non-finite (NaN/Inf) prediction at {w['ds']} "
                     f"s{base_seed} window {i} (Idx={w['idx']} "
                     f"Date={w['date']}) — nothing is written beyond this "
                     f"point in the shard")
        preds[i] = a
    return preds, ms


def write_shard(ds, seed, shard, windows, preds, ms, guards):
    npz, hdr, done = P.shard_paths(ds, seed, shard)
    os.makedirs(P.PRED_DIR, exist_ok=True)
    idx = np.array([w["idx"] for w in windows], dtype=np.int64)
    dates = np.array([w["date"] for w in windows])
    seed_base = np.full(len(windows), seed, dtype=np.int64)
    seeds_w = np.array([C.derive_seed(seed, w["ds"], w["idx"], w["date"])
                        for w in windows], dtype=np.int64)
    target = np.stack([w["target"] for w in windows]).astype(np.float64)
    tmp = npz + ".tmp.npz"
    np.savez(tmp, idx=idx, date=dates, seed_base=seed_base, seed=seeds_w,
             pred=preds, target=target, ms=ms, ds=np.array(ds))
    os.replace(tmp, npz)
    payload = {"payload_sha256": C.sha256_file(npz),
               "guards": guards, "ds": ds, "seed": seed, "shard": shard,
               "n": len(windows), "written": time.strftime("%F %T")}
    C.dump_json(payload, done)
    C.dump_json(payload, hdr)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-approve-frozen-protocol", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", default=None,
                    help="explicit weights path (default: old-root HF-cache "
                         "resolution via load_aurora)")
    ap.add_argument("--only-ds", choices=P.ORDER, default=None)
    ap.add_argument("--only-seed", type=int, choices=P.SEEDS, default=None)
    args = ap.parse_args()

    proto = P.check_approval(args.i_approve_frozen_protocol)
    P.verify_frozen(proto)
    setup_determinism()
    device = args.device
    assert device == "cuda" and torch.cuda.is_available()
    assert P.BATCH_SIZE == 1

    # --- preflight gate: no report, or ok!=true -> no full run ---
    rep_path = os.path.join(P.RESULTS_DIR, "preflight_report.json")
    if not os.path.isfile(rep_path):
        sys.exit("HARD STOP: results/preflight_report.json missing — "
                 "run preflight first")
    pre = C.load_json(rep_path)
    if not pre.get("ok"):
        sys.exit("HARD STOP: preflight report ok!=true — full run blocked")
    rep_sha = C.sha256_file(rep_path)
    print(f"[infer] preflight ok; batch={P.BATCH_SIZE}, "
          f"preflight_report sha256 {rep_sha[:16]}...")

    # --- resolve + verify the weights file the model will ACTUALLY load ---
    w_path = IA.resolve_weights(args.weights)
    w_sha_0 = C.sha256_file(w_path)
    if w_sha_0 != P.WEIGHTS_SHA256:
        sys.exit(f"HARD STOP: resolved weights sha mismatch at {w_path}")
    if pre.get("weights_sha256") not in (None, w_sha_0):
        sys.exit("HARD STOP: preflight weights sha differs from the "
                 "resolved weights file")

    model, _ = IA.build_model(w_path, device)

    guards = C.current_guards(batch=P.BATCH_SIZE,
                              preflight_report_sha256=rep_sha)
    quarantined = []
    t_start = time.perf_counter()

    for ds in P.ORDER:
        if args.only_ds and ds != args.only_ds:
            continue
        cfg = P.SUBSETS[ds]
        assert IA.expected_pad(cfg["context"]) == P.EXPECTED_PAD[ds] \
            == P.EXPECTED_PAD_LITERAL[ds], f"{ds}: pad table drift"
        assert IA.predict_token_num(cfg["pred"]) == P.PREDICT_TOKEN_NUM[ds] \
            == P.PREDICT_TOKEN_NUM_LITERAL[ds], f"{ds}: token table drift"
        windows = load_windows(ds)
        Pl = cfg["pred"]
        nsh = P.n_shards(ds)

        for seed in P.SEEDS:
            if args.only_seed and seed != args.only_seed:
                continue
            digest_a = C.model_state_digest(model)
            print(f"[infer] === {ds} s{seed}: {len(windows)} windows, "
                  f"{nsh} shards, batch={P.BATCH_SIZE}", flush=True)

            done_cnt = 0
            for shard in range(nsh):
                lo = shard * P.SHARD_SIZE
                hi = min(lo + P.SHARD_SIZE, len(windows))
                chunk = windows[lo:hi]
                valid, why = shard_valid(ds, seed, shard, guards)
                if valid:
                    done_cnt += 1
                    continue
                base = f"{ds}__s{seed}__shard{shard:03d}"
                if os.path.isfile(os.path.join(P.PRED_DIR, base + ".done")) or \
                   os.path.isfile(os.path.join(P.PRED_DIR, base + ".npz")):
                    quarantined.append(quarantine(ds, seed, shard, why))
                preds, ms = compute_shard(model, chunk, device, seed, Pl)
                write_shard(ds, seed, shard, chunk, preds, ms, guards)
                done_cnt += 1
                el = time.perf_counter() - t_start
                done_w = done_cnt * P.SHARD_SIZE
                eta_s = el / done_w * (P.EXPECTED_WINDOWS[ds] - done_w)
                print(f"[infer] {ds} s{seed} shard {shard + 1}/{nsh} done "
                      f"({len(chunk)} windows, {ms.mean():.1f} ms/win) "
                      f"elapsed {el:.0f}s, ETA(this ds x seed) "
                      f"{eta_s / 60:.1f} min", flush=True)

            digest_b = C.model_state_digest(model)
            if digest_a != digest_b:
                sys.exit(f"HARD STOP {ds} s{seed}: in-memory model state "
                         f"changed during inference")
            print(f"[infer] {ds} s{seed}: model state unchanged; complete "
                  f"({done_cnt}/{nsh} shards)", flush=True)

    w = C.sha256_file(w_path)
    if w != P.WEIGHTS_SHA256:
        sys.exit("HARD STOP: weights file changed during run")
    summary = {"finished": time.strftime("%F %T"),
               "wall_seconds": time.perf_counter() - t_start,
               "batch": P.BATCH_SIZE,
               "seeds": list(P.SEEDS),
               "preflight_report_sha256": rep_sha,
               "weights_load_path": w_path,
               "quarantined": quarantined,
               "weights_sha256": w, "env": C.env_info()}
    C.dump_json(summary, os.path.join(P.LOGS_DIR, "infer_summary.json"))
    print(f"[infer] ALL DONE in {summary['wall_seconds'] / 60:.1f} min; "
          f"quarantined shards: {len(quarantined)}", flush=True)


if __name__ == "__main__":
    main()
