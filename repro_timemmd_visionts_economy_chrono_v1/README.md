# repro_timemmd_visionts_economy_chrono_v1 (EXP-022)

VisionTS zero-shot on the **Economy chronological (ascending)** TimeMMD
protocol: the **EXP-020 data protocol** (whole rows stably sorted
ascending by date; official split/scaler/windows unchanged) applied to
the **EXP-021 method** (vendored Keytoyze VisionTS, ImageNet MAE-Base
weights, periodicity P selected on the validation set by the VisionTS
paper's own method, re-selected here on the ASCENDING validation split
2013-09..2017-03).

- Logical predecessor: EXP-021 (`repro_timemmd_visionts_zeroshot_v1`,
  freeze `c905d43db23caad732afad91dbbffc48ae693b62`).
- Data protocol basis: EXP-020 (`repro_timemmd_economy_chrono_v1`,
  freeze `dea73ba4886512b5686cc4ca57cb23b9e8d4877a`).
- Both base directories are used READ-ONLY.

## Protocol facts

| item | value |
|---|---|
| rows / split | 423 rows 1989-01..2024-03; train 296 / val 43 / test 84 |
| seq_len L | 192 |
| horizons / windows | H=6/8/10/12 -> 79/77/75/73 (304 total) |
| test starts | s = 339..423-H stride 1; first history rows 147..338; last pred starts 417/415/413/411 |
| val starts | s in [296, 339-H] closed -> 38/36/34/32 windows; first history rows 104..295 |
| scaler | sklearn StandardScaler fit on ascending train OT rows [0,296); fitted mean_/scale_ gated bitwise vs EXP-020 split_report |
| candidates P | {1, 3, 6, 12}; EXP-021's raw-order choice P=12 recorded as reference only |
| inference | mae_base, finetune none, norm=align=0.4, bilinear, float32, fp64=False, TF32 off, seed 2021, batch 256, inference_mode |

## Reproduce (dt)

```bash
cd /home/wlt/MMTS/Aurora_Experiments/repro_timemmd_visionts_economy_chrono_v1
nohup bash run_visionts_dt.sh all > watcher.out 2>&1 &
```

Single stage: data/refs gates -> preflight -> select P (16 combos) ->
freeze -> 4 test groups (identity gates vs EXP-020 frozen npz: true
bitwise / rows / dates) -> evaluate (three-way comparison + anchor
4H-mean gates, atol 1e-5) -> final audit packs under `pack_out/`.

## Comparison

Per-H and 4H-mean tables compare:
1. **EXP-020 Aurora ascending** (main comparison; recomputed from frozen
   npz; gate 0.276172/0.413902)
2. **EXP-022 VisionTS ascending** (this run)
3. **EXP-021 VisionTS raw-order** (P=12, test period 1989-01..1995-12;
   recomputed from frozen npz; gate 0.017808/0.112694)

Main question: under the SAME ascending test windows and
standardization, which has lower error — VisionTS or the measured
Aurora?

## Interpretation boundary

- vs EXP-020: spans model AND input modality (univariate numeric vs
  text+numeric) — cannot be attributed to the text modality alone.
- vs EXP-021: different test period (2017-04..2024-03 vs
  1989-01..1995-12), history window, standardization statistics and
  selection basis — describes the two protocols, not row order alone.
