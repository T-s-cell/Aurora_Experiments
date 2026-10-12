#!/usr/bin/env python3
"""Drive all 4 EXP-022 groups (Economy, H ascending 6/8/10/12), one fresh
process per group via save_group_meta.py.

Done markers are bound to the frozen protocol sha256 AND the batch AND
the selected periodicity AND the window count; a skip additionally
requires the npz on disk with the pinned sha256 and expected shapes.
Any mismatch -> stale marker -> group reruns. Requires frozen
protocol.json + selected_p.json."""
import os
import subprocess
import sys
import time

import numpy as np

import common as C
import protocol as P


def done_is_valid(name, ds, h, proto, proto_sha, batch, p_ds):
    done_path = os.path.join(C.PRED_DIR, f"{name}.done.json")
    try:
        d = C.load_json(done_path)
    except Exception as e:
        return False, f"done.json unreadable: {e!r}"
    if not d.get("ok"):
        return False, "done.ok is not true"
    if d.get("protocol_sha256") != proto_sha:
        return False, "protocol sha mismatch"
    if d.get("batch") != batch:
        return False, "batch mismatch"
    if d.get("periodicity") != p_ds:
        return False, f"periodicity {d.get('periodicity')} != {p_ds}"
    want_n = proto["datasets"][ds]["windows"][str(h)]
    if d.get("windows") != want_n:
        return False, f"windows {d.get('windows')} != {want_n}"
    if d.get("digest_ok") is not True:
        return False, "digest_ok not true"
    npz_path = os.path.join(C.PRED_DIR, f"{name}.npz")
    if not os.path.isfile(npz_path):
        return False, "npz missing"
    if d.get("npz_sha256") != C.sha256_file(npz_path):
        return False, "npz sha256 mismatch vs done pin"
    try:
        z = np.load(npz_path, allow_pickle=False)
        for key in ("pred", "true"):
            if z[key].shape != (want_n, h, 1):
                return False, f"npz {key} shape {z[key].shape} != " \
                              f"{(want_n, h, 1)}"
    except Exception as e:
        return False, f"npz unreadable: {e!r}"
    return True, None


def main():
    if not os.path.isfile(C.PROTO_PATH):
        sys.exit("[run_all] HARD STOP: protocol.json missing — freeze "
                 "first")
    import hashlib
    token = hashlib.sha256(open(C.PROTO_PATH, "rb").read()).hexdigest()[:12]
    proto_sha = C.sha256_file(C.PROTO_PATH)
    proto = C.load_json(C.PROTO_PATH)
    sp_path = P.selected_p_path()
    if C.sha256_file(sp_path) != proto["selected_p_sha256"]:
        sys.exit("[run_all] HARD STOP: selected_p.json sha256 != protocol "
                 "pin")

    bd = C.load_json(os.path.join(C.RESULTS_DIR, "batch_decision.json"))
    batch = int(bd["chosen"]["min_over_domains"])
    sel = C.load_json(sp_path)
    os.makedirs(C.PRED_DIR, exist_ok=True)

    env = dict(os.environ)
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env.setdefault("PYTHONHASHSEED", "0")

    results, t_all = {}, time.time()
    for ds in P.DOMAIN_ORDER:
        p_ds = int(sel["datasets"][ds]["selected"])
        if p_ds != int(proto["datasets"][ds]["selected_periodicity"]):
            sys.exit(f"[run_all] HARD STOP: {ds} selected P {p_ds} != "
                     f"protocol pin")
        for h in P.DATASETS[ds]["horizons"]:
            name = f"{ds}__h{h}"
            ok, why = done_is_valid(name, ds, h, proto, proto_sha, batch,
                                    p_ds)
            if ok:
                d = C.load_json(os.path.join(C.PRED_DIR,
                                             f"{name}.done.json"))
                results[name] = {"skipped": "valid done marker",
                                 "wall_s": d.get("wall_s")}
                print(f"[run_all] {name}: valid done marker, skip")
                continue
            log = os.path.join(C.LOGS_DIR, f"{name}.log")
            cmd = [sys.executable, os.path.join(C.HERE, "save_group_meta.py"),
                   "--ds", ds, "--pred-len", str(h), "--batch", str(batch),
                   "--protocol-token", token]
            t0 = time.time()
            with open(log, "w") as lf:
                r = subprocess.run(cmd, cwd=C.HERE, env=env, stdout=lf,
                                   stderr=subprocess.STDOUT)
            wall = time.time() - t0
            if r.returncode != 0:
                sys.exit(f"[run_all] HARD STOP: {name} failed rc="
                         f"{r.returncode} — see {log}")
            d = C.load_json(os.path.join(C.PRED_DIR, f"{name}.done.json"))
            ok2, why2 = done_is_valid(name, ds, h, proto, proto_sha, batch,
                                      p_ds)
            if not ok2:
                sys.exit(f"[run_all] HARD STOP: {name} done marker invalid "
                         f"after run: {why2}")
            results[name] = {"wall_s": round(wall, 1),
                             "mse32": d["metrics_float32"]["mse"],
                             "mae32": d["metrics_float32"]["mae"]}
            print(f"[run_all] {name}: OK {results[name]}")

    total_wall = round(time.time() - t_all, 1)
    summary = {"exp_id": P.EXP_ID, "groups_expected": len(results),
               "groups_done": len(results), "batch": batch,
               "total_wall_s": total_wall,
               "protocol_token": token, "protocol_sha256": proto_sha}
    C.dump_json(summary, os.path.join(C.RESULTS_DIR, "run_all_summary.json"))
    print(f"RUN_ALL OK: {len(results)}/{len(P.DOMAIN_ORDER) * 4} groups, "
          f"wall {total_wall}s")


if __name__ == "__main__":
    main()
