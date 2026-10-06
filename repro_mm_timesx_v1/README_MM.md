# EXP-011 — Fixed-Text-Budget M48T512 Baseline (Aurora × TimesX multimodal zero-shot)

Branch: `repro-mm-timesx-v1` · Protocol: `aurora-timesx-mm-v1` · Frozen
`protocol_mm.json` (approval token = first 12 hex of its sha256: `74b08ecac1ed`)

Zero-shot evaluation of **aurora-model 0.2.0** (revision `c495b02c…`, weights
sha256 `df2fb968…`) on the frozen TimesX test split (2,474 windows, 19 domains),
consuming the dataset's own English text fields under a **fixed 512-token
budget**. Same windows, denominators, seeds, and scoring as EXP-010 (A48).
**No training, no fine-tuning, no hyperparameter search.**

## Result (19-domain equal weight, mean ± std ddof=1 over seeds 2021/2022/2023)

| Method   | MSE               | MAE               |
|----------|-------------------|-------------------|
| Z0       | 3.654700 (single) | 1.049081 (single) |
| A48      | 4.293598 ± 0.001592 | 1.200945 ± 0.000305 |
| M48T512  | 4.306481 ± 0.001689 | 1.202691 ± 0.000269 |

**Paired vs A48** (per-window Δ → variable mean → domain equal weight → 19
domains): **dMSE +0.012884 ± 0.000384 (+0.30%), dMAE +0.001746** — the fixed
text budget slightly **hurts**. 7/19 domains better (largest:
electronic_technology −0.032, shopping −0.015, Currency −0.012), 12/19 worse
(largest: StrategicAndHighValueMaterials +0.090, society +0.053, finance
+0.040, climate +0.039).

**Valid negative result**: Aurora's native text pathway, under this fixed
budget, does not improve zero-shot accuracy on TimesX and slightly degrades
it relative to the unimodal A48 baseline.

## Method tag

`M48T512` = itl 48 (timeseries token length, EXP-010-identical) + Text 512
(fixed text token budget):

- Blocks in order, budget **includes the block name**, no budget transfer:
  `Background: `48 / `Events: `270 / `Calendar: `64 / `Covariates: `128.
- Whitespace normalization only; empty or bare `Unknown` → block skipped
  (618 daily windows skip Calendar). Each block encoded independently
  (`BertTokenizer`, package-bundled `bert_config/`, `add_special_tokens=False`),
  first N tokens kept; content ≤510; `[CLS] + content + [SEP]`, right-pad to 512.
- Text tensors passed via `generate(..., text_input_ids/text_attention_mask/
  text_token_type_ids)` (raw tensor passthrough); `token_type_ids` = zeros.
- 125-retention mechanism (NOT modified): BERT contextualizes all valid
  positions; the first 125 contextualized features (input content positions
  1..125) are retained and distilled by 10 learnable queries. Rear text
  (indices ≥126) still influences output through attention; the preflight
  quantified this (fix 0..125 + mask, replace from 126: retained-feature max
  diff 8.50, prediction diff 0.199).

## Reproduction

```bash
# full run (theta, idle GPU, >=20GB free gate; ~6 min at 44 ms/window)
CUDA_VISIBLE_DEVICES=1 python -u run_eval_mm.py --itl 48 --device cuda \
  --seeds 2021 2022 2023 --i-approve-frozen-protocol 74b08ecac1ed

# text cache rebuild (S3) + data verification (S2)
python text_cache.py build
python text_cache.py verify

# preflight (MM-DATA / MM-A..E; zero test-window inference without GPU section)
python run_preflight_mm.py            # add --skip-gpu for CPU-only sections

# aggregation (recomputes A48 full-precision from root predictions/)
python aggregate_mm.py
```

Gate details: `run_eval_mm.py` recomputes sha256(protocol_mm.json) and requires
it on the command line (EXP-010's token is explicitly rejected); dry-run mode
is mutually exclusive; itl must equal the protocol's 48; text-cache sha and its
key set are checked against the frozen test rows; GPU ≥20 GB. Per-seed shard
coverage is verified (unique keys == EXP-010/Z0 key set, target/d bitwise equal
to the frozen data cache, finite predictions) before aggregation.

## Freeze & provenance

- Data: `data/TimesX_Datasets.zip` `9adb9540…`, `split_manifest.json`
  `b47feebf…`, `data_cache.npz` `f83924d9…`, `Z0__test.npz` `49d456f5…`
  (identical to EXP-010; hashes in `SOURCE_MM_HASHES.json`).
- Text cache: `cache/text_tokens_M48T512.npz` `0a3cc424b095…` +
  `text_meta_M48T512.jsonl` `bca9f960a707…` (per-window raw/kept lengths,
  block token boundaries, decoded full text + sha256 — reviewers can inspect
  exactly what the model consumed).
- Full code/protocol/log/provenance hashes: `SOURCE_MM_HASHES.json`,
  `protocol_mm.json`.

## Preflight (all green, theta 3090 GPU1, zero test-window inference)

MM-DATA mapping 2474/2474 unique via the original sample-id logic, covariate
spans 0 violations; MM-A text=None bitwise 0.0 vs root `predict_window`;
MM-B hooks + rear perturbation (above); MM-C cross-process bitwise, 896
tensors + 29 BN counters unchanged; MM-D budget/traceability sampled-exact;
MM-E tamper/coverage/flag matrix all rejected as designed. Evidence in
`preflight/`, sign-off in `preflight/REVIEW_SIGNOFF.md`.

Events date review: all **162 flagged windows / 201 year-explicit date marks**
(input scope) were reviewed individually — **no clear after-the-fact result
leakage found**; wording is schedules, effective dates, or forward-looking
statements. **Actual publication times remain independently unverified.**

## Statement boundaries

- This measures only the **fixed-text-budget baseline** relative to EXP-010;
  it does **not** represent Aurora's capability under full text conditions
  (Events truncated to ≤270 tokens in every window here).
- Aurora is **not trained or fine-tuned** on TimesX; **pre-training overlap
  unaudited**; event publication timing **not independently verified**.
- The Covariates block contains additional numeric statistics; changes cannot
  be attributed solely to event semantics.
