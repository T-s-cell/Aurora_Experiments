# Aurora x CGTSF single-modality zero-shot (repro_cgtsf_zeroshot_v1)

Aurora (aurora-model==0.2.0, weights frozen) receives ONLY the Hist series
of the CGTSF subsets and is scored against the archived VisionTS EXP-016
run (commit `0932a5ac841ea399271ce88bde5b607ee0bcb831`, protocol token
`a4c01683793a`), frozen locally under `visionts_ref/`. Local commits only;
nothing is pushed without explicit approval.

- Subsets / windows (reused verbatim from EXP-016, same CSVs/sha256, CSV
  row order = window order, all-zero windows kept):
  PTF 11,520 (120->24) / LEU 11,616 (240->48) / MSPG 10,557 (480->96);
  3 seeds (2021/2022/2023) x 33,693 = 101,079 forwards, batch=1, FP32,
  eval()+inference_mode, TF32/AMP off, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Frozen inference config: itl=48, num_samples=100, revin=True,
  max_output_length = 24/48/96 (the old 96->12 hard-coded wrapper is NOT
  reused; history is never truncated). Internal behavior of the pinned
  package (verified in preflight): left zero-pad (48 - L%48)%48 =>
  {PTF:24, LEU:0, MSPG:0}; predict_token_num = ceil(P/48) => {1,1,2};
  revin stats on the UNPADDED input.
- Per-window seed: `int.from_bytes(sha256("aurora-cgtsf-v1|{base}|{ds}|
  {Idx}|{Date}")[:8], "big") % (2**31-1)`, reset immediately before each
  `generate()`; point forecast = float64 sample-mean over the 100 samples.
- Scoring: float64 ladder point->window->sequence-equal-weight, original
  units, per seed, then mean+-std(ddof=1) across seeds (predictions of
  different seeds are never averaged, no best-seed selection). VisionTS
  reference scored with the same ladder; delta = Aurora - VisionTS at
  subset and sequence level. No cross-subset total; no ChatTime table
  comparison.
- Sharding/resume: 512-window shards per (subset, seed); tier-1 fingerprint
  mismatch = hard stop; tier-2 corrupt/unfinished shard = quarantine +
  recompute; any NaN/Inf forward = hard stop.

## Run order (theta GPU1, aurora env, via wrapper_theta.sh)

1. `python freeze_ref.py` (local; builds visionts_ref/ + MANIFEST.json)
2. `python protocol.py freeze` (local; approval token = sha256[:12])
3. `python verify_visionts_ref.py` (local; rtol=atol=1e-12 vs frozen EXP-016
   metrics — MUST pass before any GPU work)
4. `./wrapper_theta.sh preflight.py --i-approve-frozen-protocol <token>`
   (230 windows x 3 seeds, all hard gates; on PASS the evidence pack is
   stored and the full run starts automatically — no separate go-ahead)
5. `./wrapper_theta.sh run_infer.py --i-approve-frozen-protocol <token>`
   (nohup; PTF -> LEU -> MSPG x 3 seeds)
6. verify_cache.py -> evaluate.py -> recompute_check.py (1e-12)
7. `python audit_pack.py --stage final` -> copy to /home/wlt/MMTS/temp/

No model/package source is modified; the old-root loader
(`../load_aurora.py`) and old `predict.py` are read-only and md5-pinned by
the protocol, as are the 8 aurora package source files.

## Archival notes (2026-10-11, user-approved freeze)

- **Primary result = a COMPLETE RE-RUN on RTX 5060 Ti / WSL2 (host "dt")**
  after a fresh on-host preflight (token `9c164222d4b9`, PASS). The
  frozen code commit is `3526c68673d55c8bcd4ae47aed4b950743d3b70f`; no
  theta shards are mixed in. Environment: `torch_50_series` env
  (py3.10 + torch 2.13.0+cu129, sm_120), aurora-model==0.2.0 (package
  md5s pinned and verified), weights snapshot sha `df2fb968…` revision
  `c495b02c`. Per the user's ruling this is a VERIFICATION experiment
  with too many co-tenants on the other hosts, so machine-induced
  numeric differences (different torch/GPU) are intentionally not
  chased; all on-host hard gates passed self-consistently.
- Full run: 201/201 shards, 101,079 forwards, 0 quarantined,
  **124.8 minutes** wall (~74 ms/window). Verification chain on the full
  set: verify_cache 101,079/101,079 zero anomalies; recompute_check and
  verify_visionts_ref both max abs diff = 0.0 (rtol=atol=1e-12).
- Results (Aurora mean±std(ddof=1) vs VisionTS, original units):
  PTF MSE 11653.213497±8.760588 vs 13602.87 / MAE 66.911664±0.013326 vs
  63.215; LEU MSE 0.064015±0.000029 vs 0.074473 / MAE 0.128443±0.000025
  vs 0.127683; MSPG MSE 20.470400±0.009472 vs 23.8101 / MAE
  1.650005±0.000315 vs 1.492204. **MSE lower by ~14% on all three
  subsets; MAE higher on all three.**
- Known reporting bug: `preflight_report.json`'s `full_run_eta_hours`
  uses a weighted-averaging mistake and is NOT trustworthy (e.g. dt
  reports 174.97 h); actual measured wall time above is authoritative.
- theta/iota partial parallel runs (stopped per user ruling at 78/201
  and 170/201) are archived locally only (predictions/, predictions_iota/,
  logs_theta_archive/, logs_iota_archive/); they are not part of the
  primary result and are not committed.
- Final audit pack:
  temp/Aurora_CGTSF_zeroshot_final_20261011_1603.tar.gz (681 files,
  sha256 7722ee4ec8f33ae3fca2010912dbc375666e97b7065d1fe346ef9fb6b15cd729).

