#!/usr/bin/env python3
"""Official Aurora x TimesX test-set evaluation driver — DUAL GATE, NOT STARTED.

Gate 1 (approval): requires --i-approve-frozen-protocol <first 12 hex of
protocol.json sha256>. Without it only --dry-run works and no inference runs.

Gate 2 (runtime): before any window is touched, ALL fingerprints are
re-verified (protocol data sha256s, weights sha256, aurora package md5s,
determinism settings live), test window count must equal 2474 and the
composite-key set must equal Z0's. GPU free-memory gate >= 20 GB.

This round ships the entry point only; do not run without explicit approval.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))


def protocol_sha():
    return hashlib.sha256((PROJECT / "protocol.json").read_bytes()).hexdigest()


def gpu_free_gb():
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    vals = [int(x) for x in q.stdout.strip().splitlines() if x.strip().isdigit()]
    return max(vals) / 1024.0 if vals else 0.0


def runtime_gate(protocol, itl, need_gpu=True):
    """Re-verify every frozen fingerprint. Returns report; raises on failure."""
    import data_loader
    from load_aurora import verify_aurora_package, verify_weights, resolve_weights_path

    rep = {"itl": itl}
    fp = data_loader.verify_data_fingerprints(protocol)
    rep["data"] = {k: v["ok"] for k, v in fp.items()}
    if not all(rep["data"].values()):
        raise RuntimeError(f"data fingerprint mismatch: {fp}")

    wp = resolve_weights_path()
    wv = verify_weights(wp)
    rep["weights"] = {"ok": wv["ok"], "sha256": wv["sha256"]}
    if not wv["ok"]:
        raise RuntimeError(f"weights fingerprint mismatch: {wv}")

    pkg = verify_aurora_package()
    rep["package_all_ok"] = pkg["all_ok"]
    if not pkg["all_ok"]:
        raise RuntimeError(f"aurora package md5 mismatch: {pkg['files']}")

    data = data_loader.TimesXData()
    rows = data.test_rows()
    rep["test_windows"] = len(rows)
    if len(rows) != data_loader.EXPECTED_TEST_TOTAL:
        raise RuntimeError(f"test windows {len(rows)} != 2474")

    z0 = np.load(PROJECT / "data" / "Z0__test.npz", allow_pickle=False)
    z0_keys = set(zip([str(x) for x in z0["var_keys"]], [str(x) for x in z0["sample_ids"]]))
    rows_keys = {(vk, sid) for vk, sid, *_ in rows}
    rep["keys_equal_z0"] = rows_keys == z0_keys
    if not rep["keys_equal_z0"]:
        raise RuntimeError("composite-key set differs from Z0")

    if need_gpu:
        free = gpu_free_gb()
        rep["gpu_free_gb"] = free
        if free < 20.0:
            raise RuntimeError(f"GPU free {free:.1f}GB < 20GB gate")

    return rep


def dry_run(protocol, itl):
    rep = runtime_gate(protocol, itl, need_gpu=False)
    n = 2474 * len(protocol["seeds"]["base_seeds"])
    plan = {"mode": "dry-run (no inference performed)",
            "runtime_gate": rep,
            "would_run": {"windows": n, "num_samples": protocol["inference"]["num_samples"],
                          "shards": f"19 domains x {len(protocol['seeds']['base_seeds'])} seeds x itl={itl}",
                          "method_tag": f"A{itl}"}}
    print(json.dumps(plan, indent=2))
    return 0


def approved_run(protocol, itl, seeds, device):
    import torch
    import data_loader
    from load_aurora import load_aurora
    from predict import ShardStore, derive_seed, predict_window, shard_fingerprint
    import aggregate

    method = f"A{itl}"
    model, load_rep = load_aurora(device=device)
    print(json.dumps({"load_report": {k: load_rep[k] for k in
                                      ("weights", "mode", "device", "determinism")}}, indent=2))

    gate = runtime_gate(protocol, itl, need_gpu=True)
    print("[gate] runtime checks OK:", json.dumps(gate))

    data = data_loader.TimesXData()
    store = ShardStore()

    for base_seed in seeds:
        rows_all = data.test_rows()
        fp = shard_fingerprint(protocol, itl, base_seed, load_rep["weights"]["sha256"])
        by_domain = {}
        for r in rows_all:
            by_domain.setdefault(r[2], []).append(r)

        for domain in data_loader.DOMAIN_NAMES:
            dom_rows = by_domain[domain]
            if store.reuse_or_init(method, domain, base_seed, fp):
                print(f"[eval] reuse {method} {domain} s{base_seed} ({len(dom_rows)} rows)")
                continue
            t0 = time.time()
            out_rows = []
            for vk, sid, d, past, target, dv in dom_rows:
                w = derive_seed(base_seed, vk, sid)
                pred = predict_window(model, past, itl, protocol["inference"]["num_samples"], w)
                out_rows.append({"var_key": vk, "sample_id": sid, "domain": d,
                                 "pred": pred, "target": target, "d": dv})
            # coverage: recomputed target/d are the frozen values themselves; verify row count
            assert len(out_rows) == len(dom_rows)
            store.save(method, domain, base_seed, fp, out_rows)
            dt = time.time() - t0
            print(f"[eval] {method} {domain} s{base_seed}: {len(out_rows)} windows "
                  f"in {dt:.1f}s ({dt / len(out_rows) * 1000:.0f} ms/window)")

    # full coverage check across shards vs Z0 (keys + target/d bitwise)
    z0 = np.load(PROJECT / "data" / "Z0__test.npz", allow_pickle=False)
    z0_order = {(str(v), str(s)): i for i, (v, s) in
                enumerate(zip(z0["var_keys"], z0["sample_ids"]))}
    for base_seed in seeds:
        acc_v, acc_t, acc_d, acc_p = [], [], [], []
        for domain in data_loader.DOMAIN_NAMES:
            z = store.load(method, domain, base_seed)
            for i in range(len(z["var_keys"])):
                key = (str(z["var_keys"][i]), str(z["sample_ids"][i]))
                j = z0_order[key]
                acc_v.append(key)
                acc_t.append(z["target"][i])
                acc_d.append(z["d"][i])
                acc_p.append(z["pred"][i])
        assert len(acc_v) == 2474
        if not (np.array_equal(np.array(acc_t), z0["target"][np.array([z0_order[k] for k in acc_v])])
                and np.array_equal(np.array(acc_d), z0["d"][np.array([z0_order[k] for k in acc_v])])):
            raise RuntimeError(f"s{base_seed}: target/d mismatch vs frozen data")
        print(f"[eval] s{base_seed} coverage OK: 2474 keys, target/d bitwise equal to frozen source")

    aggregate.main([method])
    print("[eval] DONE — official evaluation complete (statement: Aurora is NOT "
          "trained or fine-tuned on TimesX).")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--itl", type=int, required=True, choices=[48, 9])
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--i-approve-frozen-protocol", default=None,
                    help="first 12 hex of protocol.json sha256")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    protocol = json.loads((PROJECT / "protocol.json").read_text())
    seeds = args.seeds or protocol["seeds"]["base_seeds"]
    token = args.i_approve_frozen_protocol
    if token:
        expect = protocol_sha()[:12]
        if token != expect:
            print(f"[gate] REJECTED: token {token!r} != protocol sha prefix {expect!r}")
            return 2
        print("[gate] protocol approval token accepted")
        return approved_run(protocol, args.itl, seeds, args.device)
    if args.dry_run:
        return dry_run(protocol, args.itl)
    print("[gate] No approval token. Official evaluation is locked; "
          "use --dry-run for a no-inference plan, or pass "
          "--i-approve-frozen-protocol <12-hex> to unlock.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
