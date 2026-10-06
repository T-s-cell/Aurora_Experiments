# EXP-D2 — Frozen D2 Text Scheme vs EXP-011 (Aurora × TimesX multimodal zero-shot, paired prediction comparison)

Branch: `repro-mm-timesx-d2` · Protocol: `aurora-timesx-mm-d2-v1` · Frozen
`protocol_d2.json` (approval token = first 12 hex of its sha256: `7220a9aff32b`)

Controlled prediction comparison of the **frozen D2 text scheme** (deterministic
rule cleanup R1–R6 + idle-budget recovery, both frozen at
`analysis-text-budget-v1` commit `3e09d4b`) against the **EXP-011 M48T512
baseline** (branch `repro-mm-timesx-v1`, commit `2285a24`). Aurora
(`aurora-model==0.2.0`, weights sha256 `df2fb968…`) zero-shot on the same frozen
TimesX test split (2,474 windows, 19 domains), 3 seeds (2021/2022/2023).
**Only the text token ids/mask change (D0 → D2)**; model, weights, windows,
seeds, itl, denominators and scoring are EXP-011-identical. **No training, no
fine-tuning, no budget search; the D2 rules/budgets were frozen before this
round and never re-tuned on its results.**

## Result (19-domain equal weight, mean ± std ddof=1 over seeds 2021/2022/2023)

| Method    | MSE                 | MAE                 |
|-----------|---------------------|---------------------|
| Z0        | 3.654700 (single)   | 1.049081 (single)   |
| A48       | 4.293598 ± 0.001592 | 1.200945 ± 0.000305 |
| M48T512   | 4.306481 ± 0.001689 | 1.202691 ± 0.000269 |
| **M48T512_D2** | **4.305541 ± 0.001669** | **1.202446 ± 0.000269** |

M48T512 is recomputed at full precision from the frozen EXP-011 shards and
reconciles exactly with the displayed value in `repro_mm_timesx_v1/README_MM.md`.

**Paired vs EXP-011 M48T512** (per-window Δ = D2 − M48T512, same seed, same
composite key → variable mean → domain equal weight → 19 domains): **dMSE
−0.000940 ± 0.000021, dMAE −0.000244 ± 0.000001 (relative ≈ +0.02% both)** —
negative delta = D2 improves. The direction is consistent across all three
seeds (seed-level std is ~2% of the mean delta), but the magnitude is
marginal: this is **not** a material accuracy gain.

- Domain level (MSE): **14/19 improved, 5/19 degraded** (MAE: 13/19, 6/19).
  Variable level: 110/190 improved (MSE).
- Largest improvements: public_health −0.0035, pets −0.0029, public_policy
  −0.0026, traffic −0.0016. Largest regressions: science +0.0009, finance
  +0.0006, EnergyAndFuels +0.0002, shopping +0.0002.
- **D2 vs A48 remains clearly negative** (+0.011943 MSE, −0.28% relative):
  like EXP-011, the text pathway still does not beat the unimodal no-text
  baseline; D2 only slightly narrows the gap.

**Reading**: the frozen D2 input scheme, as a whole, produces a small but
directionally consistent improvement over the D0 baseline — far short of
closing the gap to A48. See scope statement below for why the gain cannot be
attributed to rule cleanup vs budget recovery separately.

## Method tag

`M48T512_D2` = itl 48 + Text 512 under the **D2 scheme**:

- **Rule cleanup R1–R6** (frozen `d2_rules.json`) applied to the four native
  field strings: scenario-prefix rewrite (period dates kept), bare numeric
  citation brackets deleted, background template compression, covariate/holiday
  prefix compression, whitespace repair. **Preserved, never deleted**: `<N>`
  event tags, event bodies, all numbers/dates/units/negation/forecast wording,
  covariate statistics.
- **Budgets 48/270/64/128** per block (including block names) after cleanup,
  then **idle-budget recovery**: leftover content budget (510 − Σ alloc)
  redistributed to still-truncated blocks in the fixed order Events →
  Covariates → Calendar → Background; alloc ≥ base per block; Σ ≤ 510.
- Assembly order, skip rule (empty / bare `Unknown`), tokenizer
  (package-bundled `BertTokenizer`, `bert_config/`, `add_special_tokens=False`),
  `[CLS] + content + [SEP]`, right-pad to 512 — all identical to EXP-011.
- Text tensors via `generate(..., text_input_ids/text_attention_mask/
  text_token_type_ids)` explicit Long tensors, `token_type_ids` = zeros;
  the string path is forbidden (as in EXP-011).
- The D0 scheme itself is defined by `repro_mm_timesx_v1/protocol_mm.json`
  (normalize → `<Name>: <field>` → first-N truncation, **no** budget transfer)
  and is not restated here.

## Reproduction

```bash
# full run (theta, idle GPU, >=20GB free gate; ~6 min at ~45 ms/window)
CUDA_VISIBLE_DEVICES=1 python -u run_eval_d2.py --itl 48 --device cuda \
  --seeds 2021 2022 2023 --i-approve-frozen-protocol 7220a9aff32b

# D2 cache extraction from the frozen diagnostic trace (S1; local, CPU)
python build_cache_d2.py

# preflight (PF-DATA/PF-D CPU; PF-A/B/C GPU; train/val windows only,
# zero test-window inference)
python run_preflight_d2.py

# aggregation (re-verifies D2 + EXP-011 shard fingerprints; recomputes
# M48T512, A48, Z0 at full precision)
python aggregate_d2.py
```

Gate details: `run_eval_d2.py` requires sha256(protocol_d2.json)[:12] on the
command line (the EXP-010 token `580eb3c3f51d` **and** the EXP-011 token
`74b08ecac1ed` are explicitly rejected); `--dry-run` is mutually exclusive and
never infers; itl must equal the protocol's 48; the D2 text-cache sha and its
key set are checked against the frozen test rows; GPU ≥20 GB. Shard
fingerprints (13 fields incl. `protocol_d2_sha256`, D2 cache sha,
`code_md5_d2`) are re-verified on resume; mismatch → `predictions/quarantine/`
(never deleted). Per-seed coverage: unique keys == Z0 key set (2,474), target/d
bitwise equal to the frozen data cache, finite predictions.

## Freeze & provenance

- Data / model / seeds / determinism: **identical to EXP-011** — zip
  `9adb9540…`, `split_manifest.json` `b47feebf…`, `data_cache.npz`
  `f83924d9…`, `Z0__test.npz` `49d456f5…`; weights `df2fb968…`
  (hf_revision `c495b02c…`); seeds 2021/2022/2023 with the shared
  `aurora-timesx-v1|` per-window derivation (EXP-011 pairable).
- D2 source: `analysis-text-budget-v1` @ `3e09d4b715c2ecdd…` —
  `clean_rules.py` / `allocate.py` / `build_diag.py` / `d2_rules.json` /
  `budget_config.json` (md5s in `protocol_d2.json: d2_provenance`).
- D2 test cache: `cache/text_tokens_M48T512_D2.npz` `45d929e68dfc…` +
  `text_meta_M48T512_D2.jsonl` `0de508bad2e1…` — extracted **bitwise** from the
  frozen diagnostic trace `trace_D2.npz` `e601f22d4c22…` restricted to
  `manifest.native.test`; key set == EXP-011 frozen cache == `test_rows()`;
  D0 trace rows verified bitwise against the EXP-011 cache. Decoded text in the
  meta is for inspection only and is never re-tokenized for prediction.
- Per-window D2 traces (block token counts, allocations, coverage, decoded
  input) ship in the meta JSONL — reviewers can inspect exactly what the model
  consumed.

## Preflight (all green, theta 3090 GPU1; test windows never predicted)

- **PF-DATA** (CPU): D2 cache sha + schema + 2,474 unique keys == EXP-011 cache
  == test rows; **bitwise** equality vs source trace for all 2,474 rows; meta
  row-wise re-check (scope==test, content ≤510).
- **PF-A** (GPU, 8 train/val windows): the D2 driver fed **D0 tensors**
  produces predictions **bitwise identical** to `predict_mm.predict_window_mm`
  (EXP-011 entry); `text=None` path bitwise identical to root
  `predict_window`. 8/8 windows, both directions.
- **PF-B** (GPU): hook-captured `generate` kwargs **hard-asserted** bitwise
  equal to the D2 cache rows (int64, [1,512], device, ttids all-zero,
  text_inputs=None), 8/8. D0-vs-D2 prediction differences recorded only
  (8/8 changed; max |Δ| 0.404) — no must-differ assertion by design.
- **PF-C** (GPU): raw samples [1,100,12] finite; same-seed repeat bitwise;
  model state digest (896 tensors, 29 BN counters) unchanged; cross-process
  child bitwise; 43.4 ms/window, peak 0.84 GB.
- **PF-D** (CPU): dry-run zero inference; token matrix (correct token passes;
  EXP-010/EXP-011 tokens, wrong itl, wrong flag all rejected); tampered cache
  sha refused; wrong-fingerprint shard quarantined.
- Preflight text inputs come exclusively from
  `cache/preflight_text_trainval.npz` (8 train/val windows from the full
  traces, disjointness from test/excluded key sets enforced by assertion).
  Evidence: `preflight/pf_{data,d,a,b,c}.json`.

## Statement boundaries

- This round evaluates the **D2 input scheme as a whole**: rule cleanup and
  budget redistribution change simultaneously, so the (marginal) gain **cannot
  be attributed to either alone**.
- Higher text coverage does **not** imply better prediction; the coverage-only
  diagnostic lives in `analysis/text_budget_v1/results/report.md` and is not
  evidence of accuracy.
- The D2 rules/budgets were **frozen before** this round (commit `3e09d4b`);
  nothing was selected or re-tuned on these test results; best seed is never
  selected.
- Aurora is **not trained or fine-tuned** on TimesX; pre-training data overlap
  **unaudited**; text source, event publication timing and truncation
  limitations **carry over unchanged** from EXP-011 (`protocol_mm.json`).
