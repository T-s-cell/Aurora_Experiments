#!/usr/bin/env python3
"""Official M48T512 (fixed-text-budget multimodal) evaluation driver.

Dual gate mirroring run_eval.py:
  Gate 1: --i-approve-frozen-protocol <first 12 hex of sha256(protocol_mm.json)>.
          The EXP-010 token is explicitly rejected. --dry-run never infers.
  Gate 2: runtime — all fingerprints re-verified (root runtime_gate on
          protocol_mm), text-cache sha + composite-key set equality with the
          2,474 frozen test rows, GPU free >= 20 GB.

Shards go to repro_mm_timesx_v1/predictions/ (method tag M48T512); the frozen
EXP-010 root predictions/ and results/ are never touched.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

SUBDIR = Path(__file__).resolve().parent
PROJECT = SUBDIR.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(SUBDIR))

from run_eval import gpu_free_gb, runtime_gate  # noqa: E402
import data_loader  # noqa: E402
from predict import (ShardStore, build_expectation, derive_seed,  # noqa: E402
                     verify_shard_coverage)
from predict_mm import (TextTokenStore, predict_window_mm,  # noqa: E402
                        protocol_mm_sha, shard_fingerprint_mm)
from text_cache import sha256_file  # noqa: E402

EXP010_TOKEN = "580eb3c3f51d"  # must NEVER unlock this driver


def dry_run(protocol, itl):
    rep = runtime_gate(protocol, itl, need_gpu=False)
    tstore = TextTokenStore(protocol)
    rows = data_loader.TimesXData(PROJECT / "data").test_rows()
    rows_keys = {(vk, sid) for vk, sid, *_ in rows}
    if tstore.keys() != rows_keys:
        raise RuntimeError("text cache key set != frozen test rows")
    n = 2474 * len(protocol["seeds"]["base_seeds"])
    plan = {"mode": "dry-run (no inference performed)",
            "runtime_gate": rep,
            "text_cache": {"npz_sha256_ok": True, "keys_equal_test_rows": True},
            "would_run": {"windows": n,
                          "num_samples": protocol["inference"]["num_samples"],
                          "shards": f"19 domains x {len(protocol['seeds']['base_seeds'])} seeds x itl={itl}",
                          "method_tag": protocol["text"]["method_tag"]}}
    print(json.dumps(plan, indent=2))
    return 0


def approved_run(protocol, itl, seeds, device):
    from load_aurora import load_aurora

    method = protocol["text"]["method_tag"]
    model, load_rep = load_aurora(device=device)
    print(json.dumps({"load_report": {k: load_rep[k] for k in
                                      ("weights", "mode", "device", "determinism")}}, indent=2))

    gate = runtime_gate(protocol, itl, need_gpu=True)
    print("[gate] runtime checks OK:", json.dumps(gate))

    tstore = TextTokenStore(protocol)
    device0 = next(model.parameters()).device
    data = data_loader.TimesXData(PROJECT / "data")
    rows_all = data.test_rows()
    rows_keys = {(vk, sid) for vk, sid, *_ in rows_all}
    if tstore.keys() != rows_keys:
        raise RuntimeError("text cache key set != frozen test rows")
    print(f"[gate] text cache OK: sha {protocol['text']['cache']['npz_sha256'][:12]}…, "
          f"{len(tstore.keys())} keys == test rows")

    expect_full = build_expectation(rows_all)
    by_domain = {}
    for r in rows_all:
        by_domain.setdefault(r[2], []).append(r)
    expect_by_domain = {d: build_expectation(rs) for d, rs in by_domain.items()}

    store = ShardStore(out_dir=SUBDIR / "predictions")

    for base_seed in seeds:
        fp = shard_fingerprint_mm(protocol, itl, base_seed, load_rep["weights"]["sha256"])
        for domain in data_loader.DOMAIN_NAMES:
            dom_rows = by_domain[domain]
            if store.reuse_or_init(method, domain, base_seed, fp,
                                   expect=expect_by_domain[domain]):
                print(f"[eval] reuse {method} {domain} s{base_seed} ({len(dom_rows)} rows)")
                continue
            t0 = time.time()
            out_rows = []
            for vk, sid, d, past, target, dv in dom_rows:
                ids, mask, ttids = tstore.get(vk, sid, device0)
                w = derive_seed(base_seed, vk, sid)
                pred = predict_window_mm(model, past, itl,
                                         protocol["inference"]["num_samples"], w,
                                         text_ids=ids, text_mask=mask,
                                         text_typeids=ttids)
                out_rows.append({"var_key": vk, "sample_id": sid, "domain": d,
                                 "pred": pred, "target": target, "d": dv})
            assert len(out_rows) == len(dom_rows)
            store.save(method, domain, base_seed, fp, out_rows)
            dt = time.time() - t0
            print(f"[eval] {method} {domain} s{base_seed}: {len(out_rows)} windows "
                  f"in {dt:.1f}s ({dt / len(out_rows) * 1000:.0f} ms/window)")

    z0 = np.load(PROJECT / "data" / "Z0__test.npz", allow_pickle=False)
    z0_keys = set(zip([str(x) for x in z0["var_keys"]], [str(x) for x in z0["sample_ids"]]))
    for base_seed in seeds:
        acc = []
        preds_by_key = {}
        for domain in data_loader.DOMAIN_NAMES:
            npz, _, _ = store._paths(method, domain, base_seed)
            with np.load(npz, allow_pickle=False) as z:
                for i in range(len(z["var_keys"])):
                    key = (str(z["var_keys"][i]), str(z["sample_ids"][i]))
                    acc.append((key[0], key[1], z["target"][i], float(z["d"][i])))
                    preds_by_key[key] = z["pred"][i]
        rep = verify_shard_coverage(acc, expect_full)
        if z0_keys != set(expect_full):
            raise RuntimeError("expectation key set != Z0 key set")
        if not all(np.isfinite(p).all() for p in preds_by_key.values()):
            raise RuntimeError(f"s{base_seed}: non-finite predictions")
        print(f"[eval] s{base_seed} coverage OK: {rep['n_rows']} rows, "
              f"{rep['n_unique_keys']} unique keys == Z0 key set, "
              f"target/d bitwise equal to frozen source, preds finite")

    import aggregate_mm
    aggregate_mm.main((method,))
    print("[eval] DONE — M48T512 official evaluation complete (statement: Aurora "
          "is NOT trained or fine-tuned on TimesX; pre-training overlap unaudited).")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--itl", type=int, default=48)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--i-approve-frozen-protocol", default=None,
                    help="first 12 hex of sha256(protocol_mm.json)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    protocol = json.loads((SUBDIR / "protocol_mm.json").read_text())
    if args.itl != protocol["inference"]["itl_main"]:
        print(f"[gate] REJECTED: itl={args.itl} != frozen itl_main "
              f"{protocol['inference']['itl_main']}")
        return 2
    seeds = args.seeds or protocol["seeds"]["base_seeds"]
    token = args.i_approve_frozen_protocol
    if token and args.dry_run:
        print("[gate] REJECTED: --dry-run and --i-approve-frozen-protocol are "
              "mutually exclusive; dry-run never performs inference.")
        return 2
    if token == EXP010_TOKEN:
        print("[gate] REJECTED: this is the EXP-010 (A48) token; it must never "
              "unlock the M48T512 protocol.")
        return 2
    if token:
        expect = protocol_mm_sha()[:12]
        if token != expect:
            print(f"[gate] REJECTED: token {token!r} != protocol_mm sha prefix {expect!r}")
            return 2
        print("[gate] protocol_mm approval token accepted")
        return approved_run(protocol, args.itl, seeds, args.device)
    if args.dry_run:
        return dry_run(protocol, args.itl)
    print("[gate] No approval token. Official evaluation is locked; "
          "use --dry-run for a no-inference plan, or pass "
          "--i-approve-frozen-protocol <12-hex> to unlock.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
