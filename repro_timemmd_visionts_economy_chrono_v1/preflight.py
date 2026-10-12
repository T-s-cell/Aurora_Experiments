#!/usr/bin/env python3
"""Preflight (EXP-022, single stage — no mid-run approval). Order:

1. data gates: Economy constructs via data.Dataset (EXP-020 ascending
   protocol, split rows, scaler bitwise gate vs EXP-020 split_report,
   ascending order_assert), test windows for every H match the frozen
   counts and boundary invariants (first history rows 147..338, last
   pred starts 417/415/413/411), val window counts recorded (first
   history rows 104..295)
2. weights gate: common.build_visionts() — isfile assert, sha pin,
   constructor strict load, EXTERNAL strict check (fresh
   mae_vit_base_patch16 strict load + per-tensor bitwise compare),
   eval(), all params frozen -> results/weights_load_check.json,
   vision_model digest baseline
3. static geometry assert: all 16 (P, H) combinations after
   update_config satisfy output_width * P >= pad_left + L + H
4. batch ladder probe 256->16 on real first test windows at H=12 with
   both extreme candidate P (1 and 12), results/batch_decision.json
5. smoke forward: one batch, finite + shape
"""
import os
import sys

import numpy as np

import common as C
import data
import protocol as P

LADDER = P.RUN_SETTINGS["batch_ladder"]
DS = "Economy"


def geometry_ok(model, p_cand, H):
    spec = P.DATASETS[DS]
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
    spec = P.DATASETS[DS]

    # 1. data gates (CPU)
    dset = data.Dataset(DS)
    row = {"n_rows": dset.n_rows,
           "split": [dset.n_train, dset.num_val, dset.n_test],
           "val_end": dset.val_end, "test_windows": {}, "val_windows": {}}
    for H in spec["horizons"]:
        hist, true, rows, dts = dset.test_windows(H)
        row["test_windows"][str(H)] = int(len(hist))
        assert len(hist) == spec["windows"][str(H)]
        assert rows[0] == dset.val_end and rows[-1] + H == dset.n_rows
        assert rows[0] - spec["seq_len"] == 147  # first test history 147..338
        vh, vt, vrows, _ = dset.val_windows(H)
        row["val_windows"][str(H)] = int(len(vh))
        assert vrows[0] - spec["seq_len"] == 104  # first val history 104..295
        assert vrows[-1] + H <= dset.val_end
        assert data.val_window_count(DS, H) == len(vh)
    if sum(row["test_windows"].values()) != P.TOTAL_TEST_WINDOWS:
        sys.exit(f"[preflight] HARD STOP: total test windows "
                 f"{sum(row['test_windows'].values())} != "
                 f"{P.TOTAL_TEST_WINDOWS}")
    C.dump_json({"datasets": {DS: row},
                 "total_test_windows": P.TOTAL_TEST_WINDOWS},
                os.path.join(C.RESULTS_DIR, "preflight_data.json"))
    print(f"[preflight] data {DS}: rows={dset.n_rows} "
          f"test={row['test_windows']} val={row['val_windows']}")

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

    # 3. static geometry over all 16 (P, H)
    geom = {}
    for p_cand in spec["candidates"]:
        for H in spec["horizons"]:
            ok, info = geometry_ok(model, p_cand, H)
            if not ok:
                sys.exit(f"[preflight] HARD STOP: geometry P={p_cand} "
                         f"h{H}: {info}")
            geom[f"P{p_cand}_h{H}"] = info
    print(f"[preflight] geometry: {len(geom)} combos OK")
    C.dump_json(geom, os.path.join(C.RESULTS_DIR, "preflight_geometry.json"))

    # 4. batch ladder probe on real windows at H=12, extreme P values
    probe = {"ladder": LADDER, "configs": {}, "chosen": {}}
    Hmax = max(spec["horizons"])
    ds_batch = LADDER[0]
    for p_cand in (min(spec["candidates"]), max(spec["candidates"])):
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
                    sys.exit(f"[preflight] HARD STOP: probe P={p_cand} "
                             f"b={b}: {e!r}")
                torch.cuda.empty_cache()
                print(f"[preflight] probe P={p_cand}: batch {b} OOM, "
                      f"stepping down")
        if chosen is None:
            sys.exit(f"[preflight] HARD STOP: P={p_cand} OOM even at "
                     f"batch {LADDER[-1]}")
        key = f"{DS}__P{p_cand}_h{Hmax}"
        probe["configs"][key] = {"batch": chosen,
                                 "windows_avail": int(len(hist))}
        ds_batch = min(ds_batch, chosen)
        print(f"[preflight] probe {key}: batch {chosen}")
    probe["chosen"][DS] = ds_batch
    probe["chosen"]["min_over_domains"] = ds_batch
    C.dump_json(probe, os.path.join(C.RESULTS_DIR, "batch_decision.json"))
    print(f"[preflight] batch decision: {probe['chosen']}")

    # 5. smoke: one batch at the probed batch
    b = probe["chosen"]["min_over_domains"]
    model.update_config(spec["seq_len"], Hmax, spec["candidates"][0],
                        norm_const=P.RUN_SETTINGS["norm_const"],
                        align_const=P.RUN_SETTINGS["align_const"],
                        interpolation=P.RUN_SETTINGS["interpolation"])
    hist, true, _, _ = dset.test_windows(Hmax)
    x = torch.from_numpy(hist[:b]).to(device)
    with torch.inference_mode():
        y = model.forward(x, fp64=False)
    ynp = y.detach().to("cpu").numpy()
    if ynp.shape != true[:b].shape:
        sys.exit(f"[preflight] HARD STOP: smoke shape {ynp.shape} != "
                 f"{true[:b].shape}")
    if not np.isfinite(ynp).all():
        sys.exit("[preflight] HARD STOP: smoke non-finite output")
    C.dump_json({"ok": True, "ds": DS, "H": Hmax, "batch": b,
                 "shape": list(ynp.shape)},
                os.path.join(C.RESULTS_DIR, "preflight_smoke.json"))
    print(f"[preflight] smoke {DS} h{Hmax} b={b}: shape {ynp.shape} "
          f"finite OK")
    print("PREFLIGHT OK")


if __name__ == "__main__":
    main()
