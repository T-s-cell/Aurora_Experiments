# repro_timemmd_economy_chrono_v1 (EXP-020)

Aurora TimeMMD multimodal zero-shot on the **Economy** domain with rows
**sorted chronologically** (stable ascending by the parsed `date`
column), everything else identical to the frozen EXP-019 official
reproduction (`../repro_timemmd_zeroshot_v1`, protocol token
`32b843660cc3`, add commit `36a9294…`, freeze commit `8878412…`).

This is a **single-protocol-change control experiment**: only the row
order of `Economy.csv` changes. The original CSV is year-descending
blocks (first row 2024-01, last row 1989-12); after sorting it covers
1989-01 .. 2024-03 ascending. Everything downstream follows the official
pipeline unchanged: split (70/10/20 = 296/43/84 rows), StandardScaler
fit on the new train rows only, history windows (seq_len 192), text
selection (first fact of the window, BERT max_length 500), inference
settings (inference_token_len 48, label_len 0, features S, target OT,
seed 2021, num_samples 100, batch 256, workers 4).

## Interpretation boundary

- Original-row-order test target rows cover **1989-01..1995-12**;
  chronological-order test target rows cover **2017-04..2024-03** —
  different periods.
- The sort simultaneously changes history-window composition, text
  selection, train/test periods and scaler statistics. The chrono-vs-orig
  diffs describe the two data protocols; they do not by themselves prove
  any paper error.
- Paper Economy numbers are reference columns only (protocol changed).

## Layout

```
official/TimeMMD/     vendored official tree (42 files, md5-pinned,
                      byte-identical to EXP-019's; same single
                      prediction-saving patch)
patches/              exp_main_pred_save.diff
prepare_data.py       sort + sort_mapping.csv + inverse-permutation proof
protocol.py           EXP-020 frozen protocol (token = sha256[:12])
run_chrono_dt.sh      one-shot driver (dt): sort -> gates -> freeze ->
                      Economy:12 (first official group) -> 6/8/10 ->
                      evaluate -> final audit pack
evaluate.py           float32 primary metrics + read-only EXP-019
                      baseline recompute + comparison CSV
audit_pack.py         final audit bundle (<50MB, incl. base evidence)
```

## Reproduce

```bash
bash run_chrono_dt.sh        # everything, on dt
# or step by step:
./wrapper_dt.sh prepare_data.py
./wrapper_dt.sh verify_refs.py
./wrapper_dt.sh preflight.py          # prints the frozen token
./wrapper_dt.sh run_all.py --i-approve-frozen-protocol <TOKEN> --only Economy:12
./wrapper_dt.sh run_all.py --i-approve-frozen-protocol <TOKEN>
./wrapper_dt.sh evaluate.py
./wrapper_dt.sh audit_pack.py --stage final
```

Requires the conda env `aurora_tmmd_v1` and `AURORA_HF_SNAPSHOT` pointing
at the pinned HF snapshot (see `wrapper_dt.sh`).

## Gates

- data: input CSV sha pinned to the EXP-019 frozen CSV; 423 rows; dates
  unique, strictly increasing after sort; OT finite; written CSV re-read
  and inverse-permutation verified cell-for-cell; three sha256s recorded
  in `results/data_prep.json` and frozen into the protocol.
- sources: official/ md5s == EXP-019 pins.
- weights: snapshot sha256/revision/config pins + `output_loading_info`
  four lists empty (`results/weights_load_check.json`).
- run: window counts 79/77/75/73 (total 304); window dates match the
  sorted CSV row-for-row; `trues` bitwise equal to
  `float32((OT - mean)/scale)`; official float32 replica vs printed
  metrics <= 1e-12; done markers bound to protocol sha256 + batch +
  npz sha256, re-validated (shapes included) before any resume skip.
