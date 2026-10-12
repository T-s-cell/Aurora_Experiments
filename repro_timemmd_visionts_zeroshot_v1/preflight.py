#!/usr/bin/env python3
"""Stage A preflight (EXP-021). Order:

1. data gates: all 9 domains construct via data.Dataset (split rows,
   finite OT, scaler), test windows for every (ds, H) match the frozen
   counts and boundary invariants, val window counts recorded
2. weights gate: common.build_visionts() — isfile assert, sha pin,
   constructor strict load, EXTERNAL strict check (fresh
   mae_vit_base_patch16 strict load + per-tensor bitwise compare),
   eval(), all params frozen -> results/weights_load_check.json,
   vision_model digest baseline
3. static geometry assert: every (ds, P, H) combination after
   update_config satisfies output_width * P >= pad_left + L + H
   (guards the forward's slicing against degenerate resize outputs)
4. batch ladder probe 256->16 on real first test windows at max-H
   (largest candidate P; Environment additionally at both extremes
   P=1 and P=365), results/batch_decision.json
5. smoke forward: one batch, finite + shape
"""
import os
import sys

import numpy as np

import common as C
import data
import protocol as P

LADDER = P.RUN_SETTINGS["batch_ladder"]


def geometry_ok(model, ds, p_cand, H):
    spec = P.DATASETS[ds]
    L = spec["seq_len"]
    model.update_config(L, H, p_cand,
                        norm_const=P.RUN_SETTINGS["norm_const"],
                        align_const=P.RUN_SETTINGS["align_const"],
                        interpolation=P.RUN_SETTINGS["interpolation"])
    out_w = model.output_resize.size[-1]
    need = model.pad_left + L + H
    return out_w * p_cand >= need, {"padded_out_width": out_w,
                                    "required": need,
                                    "pad_left": model.pad_left,
                                    "pad_right": model.pad_right,
                                    "num_patch_input": model.num_patch_input}


def main():
    import torch
    if not torch.cuda.is_available():
        sys.exit("[preflight] HARD STOP: CUDA unavailable")
    device = torch.device("cuda")

    # 1. data gates (CPU)
    data_rep = {"datasets": {}}
    for ds in P.DOMAIN_ORDER:
        spec = P.DATASETS[ds]
        dset = data.Dataset(ds)
        row = {"n_rows": dset.n_rows, "split": [dset.n_train, dset.num_val,
                                                dset.n_test],
               "val_end": dset.val_end, "test_windows": {},
               "val_windows": {}}
        for H in spec["horizons"]:
            hist, true, rows, dts = dset.test_windows(H)
            row["test_windows"][str(H)] = int(len(hist))
            assert len(hist) == spec["windows"][str(H)]
            assert rows[0] == dset.val_end and rows[-1] + H == dset.n_rows
            vh, vt, vrows, _ = dset.val_windows(H)
            row["val_windows"][str(H)] = int(len(vh))
        data_rep["datasets"][ds] = row
        print(f"[preflight] data {ds}: rows={dset.n_rows} "
              f"test={row['test_windows']} val={row['val_windows']}")
    total = sum(sum(r["test_windows"].values())
                for r in data_rep["datasets"].values())
    if total != P.TOTAL_TEST_WINDOWS:
        sys.exit(f"[preflight] HARD STOP: total test windows {total} != "
                 f"{P.TOTAL_TEST_WINDOWS}")
    data_rep["total_test_windows"] = total
    C.dump_json(data_rep, os.path.join(C.RESULTS_DIR, "preflight_data.json"))

    # 2. weights gate (single build path)
    model, wrep = C.build_visionts(seed=P.RUN_SETTINGS["seed"])
    model.to(device)
    C.dump_json(wrep, os.path.join(C.RESULTS_DIR, "weights_load_check.json"))
    print(f"[preflight] weights: strict load PASS, {wrep['n_tensors']} "
          f"tensors bitwise, sha {wrep['ckpt_sha256'][:12]}…")
    baseline = C.model_state_digest(model.vision_model.state_dict())
    C.dump_json({"vision_model_digest": baseline,
                 "note": "mask buffer excluded (re-registered per "
                         "update_config); digest covers vision_model only"},
                os.path.join(C.RESULTS_DIR, "weights_digest_baseline.json"))

    # 3. static geometry over ALL (ds, P, H)
    geom = {"datasets": {}}
    for ds in P.DOMAIN_ORDER:
        spec = P.DATASETS[ds]
        g = {}
        for p_cand in spec["candidates"]:
            for H in spec["horizons"]:
                ok, info = geometry_ok(model, ds, p_cand, H)
                if not ok:
                    sys.exit(f"[preflight] HARD STOP: geometry {ds} "
                             f"P={p_cand} h{H}: {info}")
                g[f"P{p_cand}_h{H}"] = info
        geom["datasets"][ds] = g
        print(f"[preflight] geometry {ds}: {len(g)} combos OK")
    C.dump_json(geom, os.path.join(C.RESULTS_DIR, "preflight_geometry.json"))

    # 4. batch ladder probe on real windows at max-H
    probe = {"ladder": LADDER, "configs": {}, "chosen": {}}
    for ds in P.DOMAIN_ORDER:
        spec = P.DATASETS[ds]
        dset = data.Dataset(ds)
        Hmax = max(spec["horizons"])
        p_list = [max(spec["candidates"])]
        if ds == "Environment":
            p_list = [1, max(spec["candidates"])]
        ds_batch = LADDER[0]
        for p_cand in p_list:
            hist, _, _, _ = dset.test_windows(Hmax)
            chosen = None
            for b in LADDER:
                try:
                    model.update_config(
                        spec["seq_len"], Hmax, p_cand,
                        norm_const=P.RUN_SETTINGS["norm_const"],
                        align_const=P.RUN_SETTINGS["align_const"],
                        interpolation=P.RUN_SETTINGS["interpolation"])
                    x = torch.from_numpy(hist[:b]).to(device)
                    with torch.inference_mode():
                        y = model.forward(x, fp64=False)
                    _ = y.shape
                    chosen = b
                    break
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        sys.exit(f"[preflight] HARD STOP: probe {ds} "
                                 f"P={p_cand} b={b}: {e!r}")
                    torch.cuda.empty_cache()
                    print(f"[preflight] probe {ds} P={p_cand}: batch {b} "
                          f"OOM, stepping down")
            if chosen is None:
                sys.exit(f"[preflight] HARD STOP: {ds} P={p_cand} OOM even "
                         f"at batch {LADDER[-1]}")
            key = f"{ds}__P{p_cand}_h{Hmax}"
            probe["configs"][key] = {"batch": chosen, "windows_avail":
                                     int(len(hist))}
            ds_batch = min(ds_batch, chosen)
            print(f"[preflight] probe {key}: batch {chosen}")
        probe["chosen"][ds] = ds_batch
    probe["chosen"]["min_over_domains"] = min(probe["chosen"].values())
    C.dump_json(probe, os.path.join(C.RESULTS_DIR, "batch_decision.json"))
    print(f"[preflight] batch decision: {probe['chosen']}")

    # 5. smoke: one batch, first domain, smallest candidate P
    ds0 = P.DOMAIN_ORDER[0]
    spec0 = P.DATASETS[ds0]
    dset0 = data.Dataset(ds0)
    H0 = max(spec0["horizons"])
    b = probe["chosen"]["min_over_domains"]
    model.update_config(spec0["seq_len"], H0, spec0["candidates"][0],
                        norm_const=P.RUN_SETTINGS["norm_const"],
                        align_const=P.RUN_SETTINGS["align_const"],
                        interpolation=P.RUN_SETTINGS["interpolation"])
    hist, true, _, _ = dset0.test_windows(H0)
    x = torch.from_numpy(hist[:b]).to(device)
    with torch.inference_mode():
        y = model.forward(x, fp64=False)
    ynp = y.detach().to("cpu").numpy()
    if ynp.shape != true[:b].shape:
        sys.exit(f"[preflight] HARD STOP: smoke shape {ynp.shape} != "
                 f"{true[:b].shape}")
    if not np.isfinite(ynp).all():
        sys.exit("[preflight] HARD STOP: smoke non-finite output")
    C.dump_json({"ok": True, "ds": ds0, "H": H0, "batch": b,
                 "shape": list(ynp.shape)},
                os.path.join(C.RESULTS_DIR, "preflight_smoke.json"))
    print(f"[preflight] smoke {ds0} h{H0} b={b}: shape {ynp.shape} finite "
          f"OK")
    print("PREFLIGHT OK")


if __name__ == "__main__":
    main()
