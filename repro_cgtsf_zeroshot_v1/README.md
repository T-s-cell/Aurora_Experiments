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
