#!/usr/bin/env python3
"""GPU step (EXP-022): select periodicity P for Economy on the ASCENDING
validation split by the VisionTS paper's own method — standardized-space
validation MSE for every candidate P at all 4 horizons, equal-weight mean
over H, argmin, ties to the smaller P. 16 combinations (4 P x 4 H) share
the same validation windows (38/36/34/32 for H=6/8/10/12, s in
[296, 339-H]).

Single inference path: common.build_visionts() (deterministic env,
strict external weight check, eval(), all frozen). No weight updates.
Writes results/val_evidence/Economy__P{p}.npz (float32 pred/true, row
starts, raw date strings) and results/selected_p.json.
"""
import os
import sys

import numpy as np

import common as C
import data
import protocol as P


def forward_windows(model, device, hist, batch):
    """hist (n, L, 1) float32 -> pred (n, H, 1) float32, sequential
    batches, torch.inference_mode, fp64=False."""
    import torch
    outs = []
    with torch.inference_mode():
        for i in range(0, len(hist), batch):
            x = torch.from_numpy(hist[i:i + batch]).to(device)
            y = model.forward(x, fp64=False)
            outs.append(y.detach().to("cpu").numpy().astype(np.float32))
    return np.concatenate(outs, axis=0)


def main():
    import torch
    if not torch.cuda.is_available():
        sys.exit("[select] HARD STOP: CUDA unavailable")
    device = torch.device("cuda")

    model, wrep = C.build_visionts(seed=P.RUN_SETTINGS["seed"])
    model.to(device)
    C.dump_json(wrep, os.path.join(C.RESULTS_DIR, "weights_load_check.json"))
    print(f"[select] weights verified: sha {wrep['ckpt_sha256'][:12]}…, "
          f"{wrep['n_tensors']} tensors bitwise")

    digest_before = C.model_state_digest(model.vision_model.state_dict())

    ev_dir = os.path.join(C.RESULTS_DIR, "val_evidence")
    os.makedirs(ev_dir, exist_ok=True)

    batch = P.RUN_SETTINGS["preferred_batch"]
    bd = os.path.join(C.RESULTS_DIR, "batch_decision.json")
    if os.path.isfile(bd):
        bdec = C.load_json(bd)
        batch = int(bdec["chosen"]["min_over_domains"])
        print(f"[select] using probed batch {batch} from batch_decision")

    ds = "Economy"
    spec = P.DATASETS[ds]
    dset = data.Dataset(ds)
    sel = {"exp_id": P.EXP_ID, "rule": dict(P.SELECTION_RULE),
           "seed": P.RUN_SETTINGS["seed"], "batch": batch,
           "device": torch.cuda.get_device_name(0),
           "exp021_reference_P_raw_order":
               spec["exp021_selected_P_raw_order"],
           "datasets": {}}
    blk = {"seq_len": spec["seq_len"], "candidates": {}}
    print(f"[select] === {ds} (L={spec['seq_len']}, ascending val split "
          f"2013-09..2017-03) ===")
    for p_cand in sorted(spec["candidates"]):
        arrays, per_h_f32, per_h_mae_f32, per_h_f64, counts = \
            {}, {}, {}, {}, {}
        for H in spec["horizons"]:
            hist, true, rows, dts = dset.val_windows(H)
            counts[str(H)] = int(len(hist))
            model.update_config(
                spec["seq_len"], H, p_cand,
                norm_const=P.RUN_SETTINGS["norm_const"],
                align_const=P.RUN_SETTINGS["align_const"],
                interpolation=P.RUN_SETTINGS["interpolation"])
            pred = forward_windows(model, device, hist, batch)
            if pred.shape != true.shape:
                sys.exit(f"[select] HARD STOP: {ds} P={p_cand} h{H} "
                         f"pred shape {pred.shape} != true {true.shape}")
            if not (np.isfinite(pred).all() and np.isfinite(true).all()):
                sys.exit(f"[select] HARD STOP: {ds} P={p_cand} h{H} "
                         f"non-finite values")
            mae32, mse32 = C.official_metric_replica(pred, true)
            _, mse64 = C.metrics_float64(pred, true)
            per_h_f32[str(H)] = mse32
            per_h_mae_f32[str(H)] = mae32
            per_h_f64[str(H)] = mse64
            arrays[f"pred_h{H}"] = pred
            arrays[f"true_h{H}"] = true
            arrays[f"row_pred_start_h{H}"] = rows
            arrays[f"date_pred_start_h{H}"] = dts
        mean32 = float(np.mean([per_h_f32[str(H)]
                                for H in spec["horizons"]]))
        path = os.path.join(ev_dir, f"{ds}__P{p_cand}.npz")
        np.savez(path, **arrays)
        sha = C.sha256_file(path)
        blk["candidates"][str(p_cand)] = {
            "per_H_mse_float32": per_h_f32,
            "per_H_mae_float32": per_h_mae_f32,
            "per_H_mse_float64": per_h_f64,
            "per_H_mean_mse_float32": mean32,
            "val_windows": counts,
            "val_npz_sha256": sha,
        }
        print(f"[select] {ds} P={p_cand:>3}: 4H-mean val MSE(f32)="
              f"{mean32:.6f} per-H={per_h_f32} evidence={sha[:12]}…")
    sel["datasets"][ds] = blk

    digest_after = C.model_state_digest(model.vision_model.state_dict())
    if digest_before != digest_after:
        sys.exit("[select] HARD STOP: vision_model weights changed during "
                 "selection")
    print("[select] weight digest unchanged across selection")

    scored = sorted(
        ((c["per_H_mean_mse_float32"], int(cs))
         for cs, c in blk["candidates"].items()),
        key=lambda t: (t[0], t[1]))
    best = scored[0]
    tie = [str(t[1]) for t in scored if t[0] == best[0]]
    blk["selected"] = best[1]
    blk["tied_candidates"] = tie
    blk["selected_mean_mse_float32"] = best[0]
    print(f"[select] {ds}: SELECTED P={best[1]} "
          f"(4H-mean {best[0]:.6f}, ties={tie}; EXP-021 raw-order "
          f"reference P={spec['exp021_selected_P_raw_order']})")

    C.dump_json(sel, os.path.join(C.RESULTS_DIR, "selected_p.json"))
    print("SELECT OK: selected_p.json written")


if __name__ == "__main__":
    main()
