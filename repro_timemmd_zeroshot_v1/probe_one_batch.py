#!/usr/bin/env python3
"""Single-batch GPU probe used by preflight (NOT part of the 36 groups).

Loads the model via the official Exp_Main._build_model, digests the
IN-MEMORY state_dict (includes BatchNorm running buffers), runs the
official test-batch inference path on the FIRST batch only, then
re-digests. Any digest change, non-finite output or wrong shape fails.
Reports peak CUDA memory. OOM exits nonzero (caller walks the ladder).
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C
import protocol as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--pred-len", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--hb", default="dump")
    a = ap.parse_args()

    cfg = P.DATASETS[a.ds]
    os.chdir(C.OFFICIAL_TMM)
    if C.OFFICIAL_TMM not in sys.path:
        sys.path.insert(0, C.OFFICIAL_TMM)

    import torch
    from einops import rearrange
    from exp.exp_main import Exp_Main

    args = SimpleNamespace(
        use_gpu=True, use_multi_gpu=False, gpu=0, devices="0",
        model="Aurora", model_path=os.environ["AURORA_HF_SNAPSHOT"],
        data=a.ds, data_path=f"{a.ds}.csv", root_path=C.DATASET_DIR,
        features=P.FEATURES, target=P.TARGET, embed="timeF", freq="h",
        seq_len=cfg["seq_len"], label_len=P.LABEL_LEN,
        pred_len=a.pred_len, inference_token_len=cfg["token_len"],
        batch_size=a.batch, num_workers=P.NUM_WORKERS,
        checkpoints=os.path.join(HERE, "logs", "probe_ckpt"),
        few_shot_ratio=0.1)
    exp = Exp_Main(args)

    sd_before = C.state_dict_digest(exp.model.state_dict())

    _, loader = exp._get_data(flag="test")
    n_total = len(loader.dataset)
    batch = next(iter(loader))
    (batch_x, batch_y, ids, mask, ttids) = batch
    dev = exp.device
    batch_x = batch_x.float().to(dev)
    batch_y = batch_y.float().to(dev)
    ids = ids.to(dev)
    mask = mask.to(dev)
    ttids = ttids.to(dev)
    n_vars = batch_x.shape[-1]
    xb = rearrange(batch_x, "b l c -> (b c) l")
    ids = ids.repeat(n_vars, 1)
    mask = mask.repeat(n_vars, 1)
    ttids = ttids.repeat(n_vars, 1)

    t0 = time.time()
    with torch.no_grad():
        exp.model.eval()
        pred_y = exp.model.generate(
            inputs=xb, text_input_ids=ids, text_attention_mask=mask,
            text_token_type_ids=ttids,
            inference_token_len=cfg["token_len"],
            max_output_length=a.pred_len, max_text_token_length=P.MAX_TEXT_TOKEN,
            num_samples=P.NUM_SAMPLES)
        output = rearrange(pred_y, "(b c) s l -> s b l c", c=n_vars).mean(0)
    dt = time.time() - t0

    out = output.detach().cpu().numpy()
    assert out.shape == (batch_y.shape[0], a.pred_len, n_vars), \
        f"point-pred shape {out.shape} != {(batch_y.shape[0], a.pred_len, n_vars)}"
    assert tuple(pred_y.shape) == (batch_y.shape[0] * n_vars, P.NUM_SAMPLES,
                                   a.pred_len), \
        f"sample shape {tuple(pred_y.shape)}"
    assert np.isfinite(out).all(), "non-finite probe output -> HARD STOP"

    sd_after = C.state_dict_digest(exp.model.state_dict())
    assert sd_before == sd_after, \
        "in-memory state_dict changed during inference (BN buffers?)"

    rep = {
        "ds": a.ds, "pred_len": a.pred_len, "batch": a.batch,
        "probe_windows_in_loader": int(n_total),
        "batch_rows": int(batch_y.shape[0]),
        "sample_shape": list(pred_y.shape),
        "wall_s": round(dt, 3),
        "peak_cuda_gib": round(torch.cuda.max_memory_allocated() / (1 << 30), 3),
        "state_dict_digest_unchanged": True,
        "ok": True,
    }
    print("[probe] " + json.dumps(rep))
    with open(a.hb, "w") as f:
        json.dump(rep, f)


if __name__ == "__main__":
    main()
