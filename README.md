# Aurora × TimesX — Unimodal Zero-Shot Baseline

Aurora (`aurora-model==0.2.0`, ICLR 2026, DecisionIntelligence/Aurora) evaluated
zero-shot on TimesX (190 variables / 19 domains / 2,474 native test windows,
history 96 -> forecast 12) under the frozen EXP-004 evaluation protocol.

**Scope statement:** Aurora is *not trained or fine-tuned on TimesX*. Pre-training
data overlap with TimesX is not audited, so no stronger claim is made.

Reference point: VisionTS Z0 (EXP-004, commit dcd2425), std-MSE overall
3.6547 / std-MAE 1.0491 (display precision).

## Layout

| file | role |
|---|---|
| `protocol.json` | frozen protocol fingerprint: data/weights sha256, inference config, seed rule, determinism, shard fingerprint & resume policy, eval gate |
| `SOURCE_HASHES.json` | provenance of every upstream artifact (hashes) |
| `load_aurora.py` | integrity-checked loader: sha256 -> package md5 -> bidirectional key diff -> `strict=True` -> `eval()` -> all params frozen; determinism pack applied before first inference |
| `data_loader.py` | read-only EXP-004 window logic; composite key `(var_key, sample_id)`; 2,474-window hard check |
| `predict.py` | per-window inference (seeded, sample-mean of 100), fingerprint-bound sharded npz, verified resume + quarantine |
| `aggregate.py` | port of EXP-004 aggregation (window -> variable -> 19-domain equal weight -> overall; 3-seed mean±std(ddof=1)); best seed never selected |
| `verify_aggregate.py` | preflight I: ported aggregation vs ORIGINAL EXP-004 `aggregate.py` at full precision (840 comparisons, bitwise) + 6-dp archive corroboration |
| `run_preflight.py` | preflight sections A–L driver (synthetic + train/val windows only; zero test-window inference) |
| `run_eval.py` | official evaluation driver — **dual gate, not started** |
| `reference/exp004/` | read-only vendored EXP-004 sources for machines without the VisionTS tree |
| `data/` | read-only copies: split_manifest.json, data_cache.npz, Z0__test.npz, TimesX zip (npz/zip git-ignored) |
| `results/`, `predictions/`, `preflight/`, `logs/` | outputs (predictions/logs git-ignored) |

## Key protocol facts

- Inference: batch=1, `num_samples=100` averaged to a point forecast, `max_output_length=12`,
  `revin=True`, text/vision None, `inference_token_len` main scheme **48**
  (official TimeMMD precedent: Health/Traffic use seq_len=96 + itl=48);
  itl=9 remains a frozen-decision candidate. Never infer the choice from token counts.
- Seeds: base 2021/2022/2023; per-window
  `w = int.from_bytes(sha256("aurora-timesx-v1|{base}|{var_key}|{sample_id}")[:8]) % (2**31-1)`.
- Scoring: `d = pstdev(past96, ddof=0)` with manifest `fallback_std` below 1e-8;
  window -> variable -> within-domain equal weight -> 19-domain equal weight.
  Scoring denominator `d` and Aurora's internal RevIN scale (`std(unbiased=False)+1e-5`)
  are distinct and never interchanged.
- Determinism: `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA init; TF32 off (both),
  `cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True)` before first
  inference. Determinism errors must be located and dispositioned, never silenced.
- Shards: `predictions/{method}__{domain}__s{seed}__test.npz` + fingerprint header +
  `.done`; any mismatch on resume moves the stale shard to `predictions/quarantine/`.
- transformers: `<5` is a HARD requirement (weights use transformers-4.x ViT module
  naming; 5.x renames them and strict loading fails). Verified working: 4.57.6.

## Reproduce

```bash
# preflight (GPU machine, no test-window inference)
python run_preflight.py --itl 48

# dry-run of the official evaluation (fingerprint checks + plan, no inference)
python run_eval.py --itl 48 --dry-run

# official evaluation — LOCKED behind explicit approval
# token = first 12 hex of sha256(protocol.json)
python run_eval.py --itl 48 --i-approve-frozen-protocol <12-hex>
```

## Status

- Code + eta environment (L20) + preflight A–L: complete (see `preflight/` evidence).
- Official 2,474-window × 3-seed evaluation: **implemented, not started** — awaits
  explicit protocol approval.
