# repro_timemmd_visionts_zeroshot_v1 (EXP-021)

Zero-shot **VisionTS** (original Keytoyze implementation, ImageNet
MAE-Base weights, all parameters frozen) on the **TimeMMD raw-row-order
protocol**, exactly as frozen in EXP-019
(`../repro_timemmd_zeroshot_v1`, protocol token `32b843660cc3`, add
commit `36a9294…`, freeze commit `8878412…`): same 9 CSVs (byte-pinned,
Economy kept year-descending), same splits (70/10/20), same per-dataset
seq_len, same windows (36 groups / 15,892 test windows), same
sklearn-StandardScaler-on-train-OT standardization, same official
float32 metric replica.

## What differs from EXP-019

Only the model: instead of Aurora, the vendored VisionTS
(`visionts/`, 5 files, md5-pinned to the EXP-018 frozen copies,
repo commit `a169bb39…`) runs univariate zero-shot forecasting with
`arch=mae_base`, `finetune_type='none'`, `norm_const=align_const=0.4`,
bilinear alignment, float32 inputs, `fp64=False`, TF32 off,
`torch.inference_mode()`, seed 2021.

## Periodicity selection (the experiment's point)

The Aurora paper does not disclose the periodicity P (or history L) it
used for its VisionTS baseline column. We therefore select P **by the
VisionTS paper's own method**: for every candidate P (monthly
{1,3,6,12}, weekly {1,4,52}, daily {1,7,30,365}), evaluate standardized
validation MSE at all 4 test horizons, take the equal-weight mean over
H, argmin with ties to the smaller P. Validation windows
`s ∈ [max(n_train, L), val_end − H]` never touch test rows. All
candidate evidence is kept (`results/val_evidence/{ds}__P{p}.npz`,
float32). The selected P per domain is embedded in the frozen
`protocol.json` (with an independent argmin recheck) before any test
window is predicted.

## Inference-configuration single path

`common.build_visionts()` is the only constructor used by preflight,
select_period and the 36 test groups: deterministic env seeds, TF32 off
(matmul + cudnn), cudnn deterministic, checkpoint isfile assert (the
constructor would otherwise auto-download), strict constructor load,
**external strict weight check** (fresh `mae_vit_base_patch16()`
`strict=True` load + per-tensor bitwise compare against the live model
— the constructor's own try/except is never trusted), `eval()`, all
parameters `requires_grad=False`. AMP is never engaged.

## Two-stage driver (dt)

```
bash run_visionts_dt.sh stageA   # gates -> select P -> freeze -> preflight audit packs -> STOP
bash run_visionts_dt.sh stageB   # (after user approval) 36 groups -> evaluate -> final packs -> STOP
```

Stage A stops for user review of the preflight pack; stage B stops for
the freeze review. No git actions inside the driver.

## Interpretation boundary

Numbers are a **same-data-protocol re-scoring** of VisionTS on TimeMMD;
the paper's VisionTS P/L are unknown, so per-H / per-domain differences
vs `paper_ref_visionts.json` (Table 12 / Table 1) do not by themselves
prove any paper error. Known dataset deviations carried over from
EXP-019: paper Table 7 Environment 11,102 vs actual 15,248 rows;
SocialGood 900 vs 892; Economy raw row order.
