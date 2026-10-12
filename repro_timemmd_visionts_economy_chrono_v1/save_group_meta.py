#!/usr/bin/env python3
"""One group (ds, H) of EXP-022, run as a fresh process per group.

Frozen protocol + selected_p must already exist. Single inference path:
common.build_visionts(); update_config(L, H, P_sel) with THIS group's H;
digest gate (vision_model unchanged within group AND equal to the
preflight baseline); batch inference (inference_mode, fp64=False,
float32 input); identity gates vs the frozen EXP-020 ascending npz
(true bitwise in float32, row_pred_start, date_pred_start, window
count, shapes); float32 metric replica recomputed from the reloaded
npz must match the pre-save computation at 1e-12; then predictions/
{ds}__h{H}.npz (EXP-020 key layout, float64 upcast) + done.json.
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np

import common as C
import data
import protocol as P

BASE_PRED_DIR = os.path.join("/home/wlt/MMTS/Aurora_Experiments",
                             P.BASE_DATA["dir"], "predictions")


def base_npz_path(ds, h):
    return os.path.join(BASE_PRED_DIR, f"{ds}__h{h}.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--pred-len", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--protocol-token", required=True)
    a = ap.parse_args()

    if not os.path.isfile(C.PROTO_PATH):
        sys.exit("[group] HARD STOP: protocol.json missing — freeze first")
    token = hashlib.sha256(open(C.PROTO_PATH, "rb").read()).hexdigest()[:12]
    if token != a.protocol_token:
        sys.exit(f"[group] HARD STOP: protocol token {token} != "
                 f"{a.protocol_token}")
    proto_sha = C.sha256_file(C.PROTO_PATH)
    proto = C.load_json(C.PROTO_PATH)

    sp_path = P.selected_p_path()
    if C.sha256_file(sp_path) != proto["selected_p_sha256"]:
        sys.exit("[group] HARD STOP: selected_p.json sha256 != protocol "
                 "pin — selection changed after freeze")
    sel = C.load_json(sp_path)
    p_ds = int(sel["datasets"][a.ds]["selected"])
    if p_ds != int(proto["datasets"][a.ds]["selected_periodicity"]):
        sys.exit(f"[group] HARD STOP: selected P {p_ds} != protocol "
                 f"pin {proto['datasets'][a.ds]['selected_periodicity']}")

    spec = P.DATASETS[a.ds]
    H = a.pred_len
    if H not in spec["horizons"]:
        sys.exit(f"[group] HARD STOP: H={H} not in {spec['horizons']}")
    want_n = spec["windows"][str(H)]
    name = f"{a.ds}__h{H}"

    import torch
    if not torch.cuda.is_available():
        sys.exit("[group] HARD STOP: CUDA unavailable")
    device = torch.device("cuda")
    t0 = time.time()

    model, wrep = C.build_visionts(seed=P.RUN_SETTINGS["seed"])
    model.to(device)
    baseline = C.load_json(os.path.join(C.RESULTS_DIR,
                                        "weights_digest_baseline.json"))
    dg_before = C.model_state_digest(model.vision_model.state_dict())
    if dg_before != baseline["vision_model_digest"]:
        sys.exit("[group] HARD STOP: vision_model digest != preflight "
                 "baseline")

    model.update_config(spec["seq_len"], H, p_ds,
                        norm_const=P.RUN_SETTINGS["norm_const"],
                        align_const=P.RUN_SETTINGS["align_const"],
                        interpolation=P.RUN_SETTINGS["interpolation"])

    hist, true32, rows, dts = data.Dataset(a.ds).test_windows(H)
    if len(hist) != want_n:
        sys.exit(f"[group] HARD STOP: windows {len(hist)} != {want_n}")

    outs = []
    with torch.inference_mode():
        for i in range(0, len(hist), a.batch):
            x = torch.from_numpy(hist[i:i + a.batch]).to(device)
            y = model.forward(x, fp64=False)
            outs.append(y.detach().to("cpu").numpy().astype(np.float32))
    pred32 = np.concatenate(outs, axis=0)

    dg_after = C.model_state_digest(model.vision_model.state_dict())
    digest_ok = dg_after == dg_before

    # ---- identity gates vs frozen EXP-020 ascending npz ----
    bz = np.load(base_npz_path(a.ds, H), allow_pickle=False)
    base_true32 = bz["true"].astype(np.float32)
    if not np.array_equal(true32, base_true32):
        bad = np.argwhere(true32 != base_true32)
        i0, t0_ = (int(bad[0][0]), int(bad[0][1])) if len(bad) else (0, 0)
        sys.exit(f"[group] HARD STOP: true mismatch vs EXP-020 npz at "
                 f"[{i0},{t0_}]: {true32[i0, t0_]} vs "
                 f"{base_true32[i0, t0_]}")
    if not np.array_equal(rows, bz["row_pred_start"]):
        sys.exit(f"[group] HARD STOP: row_pred_start mismatch vs EXP-020 "
                 f"npz")
    if not np.array_equal(dts.astype(str),
                          bz["date_pred_start"].astype(str)):
        sys.exit(f"[group] HARD STOP: date_pred_start mismatch vs EXP-020 "
                 f"npz")
    if pred32.shape != (want_n, H, 1) or true32.shape != (want_n, H, 1):
        sys.exit(f"[group] HARD STOP: shapes pred={pred32.shape} "
                 f"true={true32.shape} != {(want_n, H, 1)}")
    if not (np.isfinite(pred32).all() and np.isfinite(true32).all()):
        sys.exit(f"[group] HARD STOP: non-finite values in {name}")

    # ---- metrics: official float32 replica primary, float64 appendix ----
    mae32, mse32 = C.official_metric_replica(pred32, true32)
    mae64, mse64 = C.metrics_float64(pred32, true32)
    print(f"{name} mse:{mse32}, mae:{mae32}  (P={p_ds}, windows={want_n}, "
          f"batch={a.batch})")

    out_path = os.path.join(C.PRED_DIR, f"{name}.npz")
    np.savez(out_path,
             pred=pred32.astype(np.float64),
             true=true32.astype(np.float64),
             row_pred_start=rows.astype(np.int64),
             date_pred_start=dts,
             ds=np.array(a.ds),
             pred_len=np.int64(H))
    npz_sha = C.sha256_file(out_path)

    # ---- recompute gate from the reloaded npz ----
    z = np.load(out_path, allow_pickle=False)
    mae_r, mse_r = C.official_metric_replica(z["pred"].astype(np.float32),
                                             z["true"].astype(np.float32))
    if not (C.close(mse_r, mse32) and C.close(mae_r, mae32)):
        sys.exit(f"[group] HARD STOP: recompute mismatch {name}: "
                 f"mse {mse_r} vs {mse32}, mae {mae_r} vs {mae32}")
    if not digest_ok:
        sys.exit(f"[group] HARD STOP: digest changed within group {name}")

    done = {"ok": True, "exp_id": P.EXP_ID, "group": name, "ds": a.ds,
            "pred_len": H, "periodicity": p_ds, "batch": a.batch,
            "windows": want_n, "protocol_token": token,
            "protocol_sha256": proto_sha,
            "selected_p_sha256": proto["selected_p_sha256"],
            "npz_sha256": npz_sha,
            "weights_load_check": wrep,
            "digest_ok": True,
            "identity_gates": {
                "true_bitwise_vs_exp020": "PASS",
                "row_pred_start_vs_exp020": "PASS",
                "date_pred_start_vs_exp020": "PASS"},
            "metrics_float32": {"mse": mse32, "mae": mae32},
            "metrics_float64": {"mse": mse64, "mae": mae64},
            "recompute_float32": {"mse": mse_r, "mae": mae_r},
            "wall_s": round(time.time() - t0, 1)}
    C.dump_json(done, os.path.join(C.PRED_DIR, f"{name}.done.json"))
    print(f"[group] {name} DONE wall={done['wall_s']}s npz_sha="
          f"{npz_sha[:12]}…")


if __name__ == "__main__":
    main()
