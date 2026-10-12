#!/usr/bin/env python3
"""Drive the 4 zero-shot groups (H=12 first, then 6/8/10), one fresh
process per group, official entry only. Done markers are bound to the
protocol sha256 AND the batch; a skip additionally requires the npz to
exist with the pinned sha256 and the expected (windows, H, 1) shapes —
any mismatch means the marker is stale and the group reruns.
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C
import protocol as P


def done_is_valid(name, ds, h, proto, proto_sha, batch):
    """A done marker may only be skipped on: ok flag + current protocol
    sha + same batch + window count + npz present with matching sha256
    and expected shapes. Returns (True, None) or (False, reason)."""
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
    want_n = proto["datasets"][ds]["windows"][str(h)]
    if d.get("windows") != want_n:
        return False, f"windows {d.get('windows')} != {want_n}"
    npz_path = os.path.join(C.PRED_DIR, f"{name}.npz")
    if not os.path.isfile(npz_path):
        return False, "npz missing"
    got_sha = C.sha256_file(npz_path)
    if d.get("npz_sha256") != got_sha:
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


def group_cmd(ds, h, batch, log_path):
    cfg = P.DATASETS[ds]
    py = sys.executable
    cmd = [
        py, "run_longExp.py",
        "--is_training", "0",
        "--features", P.FEATURES,
        "--target", P.TARGET,
        "--label_len", str(P.LABEL_LEN),
        "--random_seed", str(P.SEED),
        "--seq_len", str(cfg["seq_len"]),
        "--pred_len", str(h),
        "--inference_token_len", str(cfg["token_len"]),
        "--data", ds,
        "--data_path", f"{ds}.csv",
        "--batch_size", str(batch),
        "--model_path", os.environ["AURORA_HF_SNAPSHOT"],
        "--root_path", C.DATASET_DIR,
        "--gpu", "0",
        "--num_workers", str(P.NUM_WORKERS),
    ]
    return cmd


def run_group(ds, h, batch, token, proto_sha, log_path, env):
    stage = os.path.join(C.PRED_DIR, f"{ds}__h{h}.stage")
    genv = dict(env)
    genv["TMMD_PRED_DIR"] = stage
    os.makedirs(stage, exist_ok=True)
    cmd = group_cmd(ds, h, batch, log_path)
    t0 = time.time()
    with open(log_path, "w") as lf:
        r = subprocess.run(cmd, cwd=C.OFFICIAL_TMM, env=genv,
                           stdout=lf, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    if r.returncode != 0:
        return {"ok": False, "why": f"official entry rc={r.returncode}",
                "wall_s": wall}
    meta = subprocess.run(
        [sys.executable, os.path.join(HERE, "save_group_meta.py"),
         "--ds", ds, "--pred-len", str(h), "--stage-dir", stage,
         "--log", log_path, "--batch", str(batch),
         "--protocol-token", token, "--pred-root", C.PRED_DIR],
        cwd=HERE, env=env, capture_output=True, text=True)
    if meta.returncode != 0:
        return {"ok": False,
                "why": f"save_group_meta failed:\n{meta.stdout}\n{meta.stderr}",
                "wall_s": wall}
    print(meta.stdout.strip())
    for f in ("preds.npy", "trues.npy"):
        p = os.path.join(stage, f)
        if os.path.isfile(p):
            os.remove(p)
    os.rmdir(stage)
    return {"ok": True, "wall_s": wall}


def weights_load_check():
    """Vendored model class + pinned snapshot must load with empty
    missing/unexpected/mismatched key lists and no error msgs."""
    cwd0 = os.getcwd()
    os.chdir(C.OFFICIAL_TMM)
    if C.OFFICIAL_TMM not in sys.path:
        sys.path.insert(0, C.OFFICIAL_TMM)
    from aurora.modeling_aurora import AuroraForPrediction
    model, info = AuroraForPrediction.from_pretrained(
        os.environ["AURORA_HF_SNAPSHOT"], output_loading_info=True)
    del model
    rep = {k: list(info.get(k, []) or []) for k in
           ("missing_keys", "unexpected_keys", "mismatched_keys",
            "error_msgs")}
    rep["ok"] = all(len(v) == 0 for v in rep.values())
    C.dump_json(rep, os.path.join(C.RESULTS_DIR, "weights_load_check.json"))
    os.chdir(cwd0)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-approve-frozen-protocol", required=True)
    ap.add_argument("--only", default=None,
                    help="comma list of DS:H to run a subset (preflight)")
    ap.add_argument("--pred-root", default=None,
                    help="redirect predictions dir (preflight sandbox)")
    a = ap.parse_args()

    proto = P.verify_frozen(a.i_approve_frozen_protocol)
    wlc = weights_load_check()
    if not wlc["ok"]:
        sys.exit(f"[run-all] weights load check FAIL: {wlc}")
    print("[run-all] weights load check OK (missing/unexpected/"
          "mismatched/error all empty)", flush=True)
    proto_sha = C.sha256_file(C.PROTO_PATH)
    plan = proto["batch_plan"]
    if a.pred_root:
        C.PRED_DIR = os.path.abspath(a.pred_root)
        os.makedirs(C.PRED_DIR, exist_ok=True)

    if a.only:
        groups = []
        for item in a.only.split(","):
            ds, h = item.split(":")
            groups.append((ds, int(h)))
    else:
        groups = [(ds, h) for ds in proto["domain_order"]
                  for h in proto["datasets"][ds]["horizons"]]
    print(f"[run-all] {len(groups)} group(s); batch_plan={plan}")

    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONHASHSEED", "0")

    summary = {"protocol_token": a.i_approve_frozen_protocol,
               "protocol_sha256": proto_sha, "groups": {}, "stopped": None}
    for gi, (ds, h) in enumerate(groups, 1):
        batch = plan[ds]
        name = f"{ds}__h{h}"
        done_path = os.path.join(C.PRED_DIR, f"{name}.done.json")
        if os.path.isfile(done_path):
            valid, why = done_is_valid(name, ds, h, proto, proto_sha, batch)
            if valid:
                print(f"[run-all] [{gi}/{len(groups)}] {name}: done "
                      f"(protocol+batch+npz verified) -> skip")
                summary["groups"][name] = {"skipped": True}
                continue
            print(f"[run-all] {name}: stale/incomplete done ({why}) -> rerun")
        want_n = proto["datasets"][ds]["windows"][str(h)]
        log_path = os.path.join(C.LOGS_DIR, f"group_{name}.log")
        print(f"[run-all] [{gi}/{len(groups)}] {name}: batch={batch} "
              f"windows={want_n} seq_len={proto['datasets'][ds]['seq_len']} "
              f"token_len={proto['datasets'][ds]['token_len']}", flush=True)
        res = run_group(ds, h, batch, a.i_approve_frozen_protocol,
                        proto_sha, log_path, env)
        res["wall_s"] = round(res["wall_s"], 1)
        summary["groups"][name] = res
        if not res["ok"]:
            summary["stopped"] = name
            C.dump_json(summary, os.path.join(C.LOGS_DIR,
                                              "run_all_summary.json"))
            sys.exit(f"[run-all] HARD STOP at {name}: {res['why']}")
    C.dump_json(summary, os.path.join(C.LOGS_DIR, "run_all_summary.json"))
    print(f"[run-all] ALL {len(groups)} GROUPS OK")


if __name__ == "__main__":
    main()
