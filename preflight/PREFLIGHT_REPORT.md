# Preflight Report A–L — Aurora × TimesX zero-shot baseline

Run: 2026-10-06, single coherent run on **eta L20** (`cuda`), env
`/dev_data/wlt/conda/envs/aurora` (clone of `pytorch_init`: torch 2.12.0+cu126 /
numpy 2.4.4; installed `aurora-model==0.2.0`, `transformers==4.57.6`,
`safetensors==0.8.0`, `huggingface_hub==0.36.2`; full lock in
`environment_lock.txt`). Driver: `run_preflight.py --itl 48`.
Log: `preflight_final_itl48.log`. Assertion evidence: `preflight/*.json`,
small raw samples under `preflight/samples/`.

**Zero inference on test windows.** All model calls use synthetic windows or
train/val windows only (`preflight_report.json.test_window_inference = 0`).

## Result table

| section | what | result | key evidence |
|---|---|---|---|
| A | load integrity | OK | weights sha256 `df2fb968…72b58` == protocol; package md5 8/8; key diff bidirectionally empty (896 == 896); `strict=True` clean; `training=False`; trainable params 0 |
| B | BN eval-state | OK | 29 BatchNorm modules; running stats + counters bitwise unchanged across real inference |
| C | shape/finite | OK | every window: raw output `[1,100,12]`, point forecast `(12,)`, all finite |
| D | determinism | OK | same-seed twice bitwise on 3 windows; different seed differs; settings live (TF32 off ×2, benchmark off, det-algos on, CUBLAS=:4096:8); **cross-process bitwise (max_abs_diff = 0.0)** via two fresh subprocesses |
| E | immutability | OK | full `state_dict` 896 tensors (incl. `num_batches_tracked`) bitwise unchanged after inference |
| F | window identity | OK | data_cache slices vs TimesX zip JSON originals exact (max abs diff 0.0) on spot checks; 2,474 unique composite keys `(var_key, sample_id)` |
| G | RevIN A/B | OK | official `revin=True` vs manual-normalize→`revin=False`→manual-restore: **bitwise equal on all 4 windows** (max_abs = max_rel = 0.0; declared fallback tolerances abs≤1e-5 / rel≤1e-6 not needed) |
| H | metrics | OK | `window_d` closed form; constant window → `fallback_std`; std-MSE closed form |
| I | aggregation vs ORIGINAL | OK | 840 comparisons (190 vars × 4 metrics + 19 domains × 4 + overall) vs EXP-004 `aggregate.py` at full precision: **all bitwise (max_abs_diff = 0.0)**; corroboration vs archived 6-dp CSV: MSE diff 1.3e-7, MAE diff 2.3e-7 ≤ 5e-7 |
| J | itl=48 vs itl=9 | OK | parallel evidence below |
| K | performance | OK | 325 ms/window, peak VRAM 0.83 GB, extrapolated 2,474 × 3 ≈ 40 min |
| L | resume drill | OK | continuous vs interrupted-resume bitwise; compatible shard reused bitwise; stale fingerprint (itl changed) rejected and moved to `quarantine/`; duplicate & missing key detection triggered |

## G — scale semantics (protocol note)

- The scoring denominator `d = pstdev(past96, ddof=0)` (fallback `fallback_std`
  below 1e-8) and Aurora's internal RevIN scale `std(unbiased=False)+1e-5` are
  **distinct quantities**. Scoring uses only `d`; inference uses only the
  official RevIN path. They are never interchanged or blended. No negligibility
  claim is made for small-scale sequences.
- Degenerate (constant) windows: scoring falls back to `fallback_std`; inference
  RevIN divides by 1e-5. Different mechanisms for different purposes — both kept
  exactly as defined (H verifies the scoring rule; C/G verify inference).

## J — itl evidence (parallel presentation; freeze is a user decision)

| scheme | geometry | ms/window | notes |
|---|---|---|---|
| itl=48 (main) | `ceil(12/48)=1` token, pad 0, output 48→slice 12 | 320 | official TimeMMD zero-shot precedent: Health/Traffic `seq_len=96, inference_token_len=48`; README quickstart default |
| itl=9 (candidate) | `ceil(12/9)=2` tokens, pad 3, output 18→slice 12 (no misalignment observed: finite, correct shape) | 351 | matches the paper's "11 past tokens" description (96/9≈10.7); LightGTS-style periodic patching would suggest itl=period |

Both schemes produce finite, correctly-shaped predictions on the same windows.
**No recommendation is made from token counts**; the main/candidate framing
follows the official script precedent. The final freeze is the user's call.

## Dispositions / anomalies encountered (no silent fallbacks)

- `transformers<5` is a hard requirement: the shipped weights use
  transformers-4.x ViT module names (`…encoder.layer.N.attention.attention.key`);
  under 5.x the vision encoder builds `…layers.N.attention.k_proj`, breaking
  strict loading (100+ key mismatch). Verified working: 4.57.6 (local + eta).
- `torch.use_deterministic_algorithms(True)` produced **no operator errors** on
  the full inference path — no op required disposition.
- Preflight iterations fixed driver-side bugs only (window-tuple unpacking,
  tensor cast, vendored import path, resume-drill assertions); the model/
  protocol/scoring code was fixed once (`predict.py` float64 cast) and the
  **final single-run A–L evidence** above supersedes all earlier partial runs
  (logs of those iterations retained in `logs/preflight_run_itl48*.log`).

## Approval gate status

`run_eval.py` dual gate verified:
- no token → locked (exit 1); wrong token → rejected (exit 2);
- `--dry-run` performs full fingerprint checks + plan (7,422 windows = 2,474 × 3
  seeds, 19×3 shards, method tag `A48`) with zero inference.
- Official run additionally gates on GPU free ≥ 20 GB (observed 23.5 GB idle).
