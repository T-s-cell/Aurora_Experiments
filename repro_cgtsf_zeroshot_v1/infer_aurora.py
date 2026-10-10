#!/usr/bin/env python3
"""Aurora inference wrapper (single-modality, Hist-only).

Delegates loading to the OLD-ROOT load_aurora.py (read-only reuse, md5-
pinned by the protocol): package md5 gate -> weights sha256+bytes gate ->
strict key-exact state_dict load -> eval() -> all params+buffers frozen ->
apply_determinism(). No model source is modified anywhere.

predict_window() parameterizes max_output_length (24/48/96) — the old
96->12 hard-coded wrapper is NOT reused. Internal behavior of the pinned
package (verified + recorded in preflight, never bypassed):
  - left zero-pad to a multiple of token_len=48: (48 - L % 48) % 48
    => PTF 120 -> pad 24, LEU 240 -> pad 0, MSPG 480 -> pad 0; history is
    never truncated;
  - revin statistics computed on the UNPADDED input (ts_generation_mixin),
    so manually prepended zeros change the output — recorded demo, no gate;
  - output tokens ceil(P/48) (1/1/2), clipped to max_output_length in the
    stream head => exact output shape [B, num_samples, P]."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for p in (REPO_ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

# importing the old-root loader pins CUBLAS_WORKSPACE_CONFIG into the
# process env BEFORE any CUDA initialization (os.environ.setdefault at its
# import time).
import load_aurora as LA  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import protocol as P  # noqa: E402


def expected_pad(context_len):
    return (P.TOKEN_LEN - context_len % P.TOKEN_LEN) % P.TOKEN_LEN


def predict_token_num(pred_len):
    return -(-pred_len // P.TOKEN_LEN)


def resolve_weights(explicit=None):
    """Resolve the weights file (HF cache snapshot / mirror) and verify
    sha256+bytes against the frozen pins. Returns the load path."""
    wp = LA.resolve_weights_path(explicit)
    rep = LA.verify_weights(wp)
    if not rep.get("ok"):
        sys.exit(f"HARD STOP: weights verification failed at {rep['path']}: "
                 f"sha {rep['sha256']} bytes {rep['bytes']}")
    return str(rep["path"])


def build_model(weights_path, device):
    model, report = LA.load_aurora(weights_path=weights_path, device=device,
                                   verify_package=True)
    if not report.get("weights", {}).get("ok", True):
        sys.exit("HARD STOP: loader weight report not ok")
    return model, report


def predict_window(model, hist, pred_len, seed):
    """hist: float64 (C,) -> float64 (P,) point forecast (float64 sample
    mean over num_samples; exact old predict.py aggregation). Resets the
    global RNG to the per-window seed immediately before generate()."""
    x = torch.from_numpy(np.asarray(hist, dtype=np.float32)).unsqueeze(0)
    x = x.to(next(model.parameters()).device)
    torch.manual_seed(int(seed))
    with torch.inference_mode():
        out = model.generate(
            inputs=x,
            text_inputs=None,
            vision_inputs=None,
            revin=P.REVIN,
            num_samples=P.NUM_SAMPLES,
            max_output_length=int(pred_len),
            inference_token_len=P.ITL,
        )
    if tuple(out.shape) != (1, P.NUM_SAMPLES, int(pred_len)):
        sys.exit(f"HARD STOP: generate() output shape {tuple(out.shape)} "
                 f"!= (1, {P.NUM_SAMPLES}, {pred_len}); internal "
                 f"pad/clip behavior differs from the frozen analysis")
    return out.to(torch.float64).mean(dim=1).squeeze(0).cpu().numpy()
