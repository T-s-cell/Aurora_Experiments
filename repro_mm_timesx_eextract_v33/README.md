# repro_mm_timesx_eextract_v33 — EXP-013: E-Extract v3.3 three-seed test prediction

Controlled prediction round on the frozen TimesX test split (2,474 windows,
19 domains), controlled against **EXP-012 M48T512_D2** (`repro_mm_timesx_d2`,
commit `f30f56b`). ONLY the Events token ids/mask change: the frozen E-Extract
v3.3 whole-sentence scheme from the diagnostic `analysis-event-compression-v1`
@ `8fe2320` replaces the D2 Events input. Model, weights, windows, seeds
(2021/2022/2023), itl=48, batch 1, 100-sample mean, ReVIN, per-window seed
derivation and the scoring ladder are EXP-012-identical. Method tag
`M48T512_E33`.

Question: does v3.3's higher event coverage (val 27/57 windows compressed,
event entry rate 58.8%, model screening did not flag distortion) reduce
prediction error vs D2? The diagnostic measured coverage; this round measures
prediction. Neither implies the other.

## Pipeline (execute on theta; commits stay local, never pushed)

    bash wrapper_theta.sh budgets        # 2474-window budgets; bitwise gate vs trace_D2
    bash wrapper_theta.sh compress       # LLM build (hours, resumable; terminal ledger)
    bash wrapper_theta.sh assemble       # offline re-assembly + npz + build_report
    bash wrapper_theta.sh freeze         # protocol_e33.json (prints the approval token)
    bash wrapper_theta.sh preflight-cpu  # PF-DATA' + PF-D
    bash wrapper_theta.sh preflight      # + PF-A/PF-E/PF-C on GPU (train/val only)
    bash wrapper_theta.sh eval-dry       # no-inference plan
    bash wrapper_theta.sh eval <12hex>   # official 2474 x 3 run + aggregate

## Terminal-event policy (frozen)

Per unique (event text, budget) cache key: ONE initial request + at most 2
correction requests (model turns), counted across restarts. Terminal success
AND terminal failure are persisted in `outputs/terminal_ledger_test.jsonl`
(first-wins, immutable) and always reused; nothing is regenerated or
overwritten by the diagnostic's last-wins cache behavior. Transport failures
(tunnel blips) are not terminal: the window is left unrecorded and retried.

## Layout

- `build_budgets_e33.py` — test-window budgets; bitwise D2 gate
- `build_cache_e33.py` — compress / assemble / freeze phases
- `predict_e33.py` — TextTokenStore (sha-gated npz) + line-identical
  `predict_window_mm`
- `run_eval_e33.py` — dual gate (approval token rejects EXP-010/011/012
  tokens; runtime re-verifies all fingerprints; GPU >= 20 GB)
- `run_preflight_e33.py` — PF-DATA' (independent full cache re-verification),
  PF-D (gate matrix), PF-A (D0 path bitwise-equals EXP-011 entry), **PF-E**
  (REAL compressed + fallback E-Extract train/val windows: final ids/mask
  reach the model exactly, shapes/finiteness, same-seed repeatability,
  fallback predictions bitwise-equal D2), PF-C (weights unchanged, cross-process)
- `aggregate_e33.py` — ladder metrics, paired E33−D2 deltas (negative =
  improvement), fallback-window bitwise validation, input-change ratio
- `protocol_e33.json` — frozen at freeze step; its sha prefix is the approval token
- `outputs/` (gitignored), `cache/` (npz + records), `predictions/`, `results/`, `preflight/`

## Staged audit gate before official inference

Before any test-set predictions, build a review package in the repository
`temp/` containing the frozen protocol and its full source hashes, E33 test
text cache (ids/mask + composite keys), window/event mapping and terminal
success/failure ledger (sanitized; no raw auth/service secrets), budget and
assembly gate reports, all PF-DATA'/PF-D/PF-A/PF-E/PF-C evidence, and the
EXP-012 D2 reference predictions needed for independent comparison. Include
checksums/manifest. Return this package for user audit and **do not run official
inference until the user explicitly approves after review**. The approval token
only unlocks the driver; it is not a substitute for audit approval. Once
approved, run the official 2,474 x 3 evaluation and aggregate.

- v3.3 diagnostic: branch `analysis-event-compression-v1` @ `8fe2320`;
  prompt `prompts/extract_v3.md` sha256 `fd102d9a…`; code reused read-only via
  sys.path (`compress.py`, `ec_common.py`, `cache.py`, `llm_client.py`).
- Baseline: `repro_mm_timesx_d2/protocol_d2.json` (EXP-012, sha-verified at
  aggregate time).
- LLM service: zeta via theta SSH tunnel `localhost:8020`; key via
  `TIMESX_LLM_API_KEY` env only — never in code, CLI, git, logs or packs.
