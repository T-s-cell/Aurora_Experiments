# Aurora TimeMMD official-public-implementation zero-shot reproduction (original row order) — repro_timemmd_zeroshot_v1

One question: running the Aurora paper's TimeMMD multimodal zero-shot
benchmark (paper Table 12: 9 domains × 4 horizons = 36 groups; Table 1:
domain averages) with the OFFICIAL public pipeline at commit
`a247760abbc9d17a861bc365c032368d317815f2` (`TimeMMD/scripts/
run_aurora_timemmd_zero_shot.sh` → `run_longExp.py` →
`exp/exp_main.py::test`), on the official benchmark CSVs IN THEIR ORIGINAL
ROW ORDER, how close do we get to the published numbers? Single seed 2021,
`generate(num_samples=100)` mean as point prediction, metrics in the
standardized space per the official entry.

- The official source is vendored byte-exact under `official/TimeMMD/`
  (42 files md5-pinned in `protocol.py`). The ONLY source change is
  `patches/exp_main_pred_save.diff` (~6 lines saving preds/trues via
  `TMMD_PRED_DIR`); both pristine and patched exp_main md5s are pinned.
  Paths/devices/dependencies are adapted via CLI args and a cloned conda
  env, never via source edits.
- Data: the 9 domain CSVs (`dataset/`, sha256-pinned) exactly as shipped
  in the official benchmark pack, original row order preserved —
  Economy's year-descending blocks stay as-is; results are labeled
  "original row order". This does not endorse the ordering; it replicates
  what the official loader consumes.
- Known paper-vs-artifact deviations (recorded, not resolved): paper A.4
  says past tokens are fixed at 11 while the script pins per-dataset
  seq_len (192/1056/528/220/96); Environment CSV has 15,248 rows vs
  Table 7's 11,102; SocialGood 892 vs 900. Security Table-12 values use
  commas as decimal-point artifacts (67.572 etc.).
- Execution (dt, RTX 5060 Ti / WSL2, env `aurora_tmmd_v1` cloned from
  `torch_50_series` + pandas/scikit-learn/matplotlib; torch/torchvision/
  numpy versions must NOT change — hard gate): wait GPU → env snapshot →
  `verify_refs.py` (CSV sha / official md5 / HF weights sha256+revision /
  offline tokenizer) → `preflight.py` (data gate incl. Economy
  Month='January 2024' & date='2024-01-01' asserts; per-domain batch
  probe with OOM ladder 256→16 through `probe_one_batch.py` which digests
  the in-memory state_dict incl. BatchNorm buffers before/after; full
  Security h12 mini group with save+recompute gates; ETA) → protocol
  freeze (batch plan included; token = sha256[:12]) → PREFLIGHT audit
  pack → **STOP for user review**. Only after explicit approval:
  `run_all.py --i-approve-frozen-protocol <token>` (36 groups, one fresh
  process each, done markers bound to protocol sha + actual batch,
  NaN/Inf and window-count hard stops) → `evaluate.py` → final pack.
- Recompute gate: the official float32 metric path is replicated
  byte-faithfully (`common.official_metric_replica`) and must match the
  entry's printed mse/mae within rtol=atol=1e-12 (identical computation,
  so 0.0 is expected). float64 recomputation is REPORTED and carries no
  1e-12 gate.

## Status

- Preflight pack: `temp/Aurora_TimeMMD_zeroshot_preflight_<ts>.tar.gz`
- Final pack: `temp/Aurora_TimeMMD_zeroshot_final_<ts>.tar.gz`
- See `results/summary.json` after the full run.
