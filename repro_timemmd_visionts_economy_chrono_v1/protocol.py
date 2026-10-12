#!/usr/bin/env python3
"""EXP-022 frozen protocol (repro_timemmd_visionts_economy_chrono_v1).

Applies the EXP-021 VisionTS method (vendored Keytoyze VisionTS, ImageNet
MAE-Base weights, periodicity selected on the validation set by the
VisionTS paper's own method) to the EXP-020 Economy chronological
(ascending) data protocol: whole rows stably sorted ascending by parsed
date, official split/scaler/window construction unchanged.

Two frozen anchors:
  BASE_DATA   = EXP-020 repro_timemmd_economy_chrono_v1   (data protocol)
  BASE_METHOD = EXP-021 repro_timemmd_visionts_zeroshot_v1 (model+selection)

Usage:
  python protocol.py freeze   # requires results/data_prep.json and
                              # results/selected_p.json; embeds selected P
                              # (argmin independently recomputed), writes
                              # protocol.json, prints token
  python protocol.py verify <token>

Token = sha256(protocol.json bytes)[:12].
"""
import hashlib
import json
import os
import sys

import common as C

EXP_ID = "EXP-022"
PROTOCOL_NAME = "repro_timemmd_visionts_economy_chrono_v1"
LABEL = ("Economy 升序协议（EXP-020 数据口径）× 原版 VisionTS zero-shot "
         "复评 v1（升序验证集按 VisionTS 论文方法选 P）")

BASE_DATA = {
    "exp_id": "EXP-020",
    "dir": "repro_timemmd_economy_chrono_v1",
    "add_commit": "244e953",
    "freeze_commit": "dea73ba4886512b5686cc4ca57cb23b9e8d4877a",
    "protocol_sha256": ("712d06fb9900cb2d634920d189dd1e6d6a8d6d78fa896166"
                        "ebf260cf1fba7997"),
    "protocol_token": "712d06fb9900",
    "role": "data protocol: Economy whole-row stable ascending sort, "
            "official split/scaler/windows",
}
BASE_METHOD = {
    "exp_id": "EXP-021",
    "dir": "repro_timemmd_visionts_zeroshot_v1",
    "add_commit": "d95004d4edb70595a54ae13dd188cc3b28b5cc88",
    "freeze_commit": "c905d43db23caad732afad91dbbffc48ae693b62",
    "protocol_sha256": ("06c003f25618665914939bf641d41f217edb6320db9baebd"
                        "a8c0a3dd63706eda"),
    "protocol_token": "06c003f25618",
    "role": "method: vendored VisionTS + validation-set periodicity "
            "selection (VisionTS paper method)",
}

VISIONTS_CODE_MD5 = {
    "visionts/__init__.py": "60249b5b1f0bd22cc6b0bd76e5eee077",
    "visionts/model.py": "621f3f87b299c4bfeb5a64c87eefaa62",
    "visionts/models_mae.py": "20edd0d6da5b21455d6c2ffb35de3d43",
    "visionts/pos_embed.py": "d31375b8ea4a1f1a9803a68b3c292399",
    "visionts/util.py": "d14f84b9bd2650c5bfdab3128457b540",
}
VISIONTS_REPO_COMMIT = "a169bb39b23c6708048f206e6df9951c7e439e73"
CGTSF_TOKEN = "a4c01683793a"  # EXP-018 provenance token for this code

DOMAIN_ORDER = ["Economy"]

DATASETS = {
    "Economy": {
        "seq_len": 192, "horizons": [6, 8, 10, 12],
        "windows": {"6": 79, "8": 77, "10": 75, "12": 73},
        "candidates": [1, 3, 6, 12], "frequency": "monthly",
        "split_rows": [296, 43, 84],
        "exp021_selected_P_raw_order": 12,
        "csv_sha256": ("023292b969b915a73185e047debfb18f5572544f9463b23e64"
                       "e9f496a0b52b26"),
        "original_csv_sha256": ("c35325f0181a426a78c35b9172d8908f64639796a"
                                "535c5fdc8b082b538030e5d"),
        "sort_mapping_sha256": ("43c3fd29f6275eb346066c2d3d6e2182f4f174bfd"
                                "9d8bd6c6ab75d0492cc74b2"),
        "order_assert": {"first_date": "1989-01-01",
                         "last_date": "2024-03-01"},
        "val_first_window_history_rows": [104, 295],
        "test_first_window_history_rows": [147, 338],
        "test_last_pred_start_rows": {"6": 417, "8": 415, "10": 413,
                                      "12": 411},
    },
}

TOTAL_TEST_WINDOWS = sum(sum(d["windows"].values()) for d in
                        DATASETS.values())
assert TOTAL_TEST_WINDOWS == 304, TOTAL_TEST_WINDOWS

SCALER = {
    "fit_rows": "[0, n_train) = [0, 296) ascending OT, sklearn "
                "StandardScaler (official-loader replica)",
    "mean_ot": -35405.91486486487,
    "scale_ot": 23084.49105192197,
    "gate": "fitted mean_/scale_ must equal the EXP-020 split_report "
            "values exactly (bitwise float64), else HARD STOP",
}

RUN_SETTINGS = {
    "arch": "mae_base",
    "finetune_type": "none",
    "norm_const": 0.4,
    "align_const": 0.4,
    "interpolation": "bilinear",
    "fp64": False,
    "input_dtype": "float32",
    "autocast": "disabled",
    "tf32_matmul": False,
    "tf32_cudnn": False,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "seed": 2021,
    "cublas_workspace_config": ":4096:8",
    "inference_mode": True,
    "features": "S",
    "target": "OT",
    "preferred_batch": 256,
    "batch_ladder": [256, 128, 64, 32, 16],
}

SELECTION_RULE = {
    "source": "VisionTS paper (Keytoyze) A.2 / Table 5 monthly candidate "
              "set, re-selected on the ASCENDING validation split "
              "(2013-09..2017-03) — EXP-021's raw-order choice P=12 is "
              "recorded only as a reference (exp021_selected_P_raw_order)",
    "candidates": [1, 3, 6, 12],
    "metric": "standardized-space validation MSE, official float32 "
              "replica (np.mean default dtype); float64 kept as appendix",
    "aggregate": "equal-weight mean of the 4 horizon MSEs per candidate "
                 "P, then argmin; ties resolved to the SMALLER P",
    "val_windows": "s in [max(n_train, seq_len), val_end - H] closed "
                   "(= [296, 339-H], 38/36/34/32 windows for H=6/8/10/12); "
                   "hist [s-L, s), label [s, s+H); stride 1; no test row "
                   "is ever touched",
    "weight_updates": "none — model stays in eval() with all parameters "
                      "frozen during selection",
    "shared_by_test": "the single selected P is used for all 4 test "
                      "horizons",
}


def selected_p_path():
    return os.path.join(C.RESULTS_DIR, "selected_p.json")


def data_prep_path():
    return os.path.join(C.RESULTS_DIR, "data_prep.json")


def recompute_argmin(sel):
    """Independent argmin over the embedded per-candidate scores
    (equal-weight 4-H mean, ties to smaller P). Returns {ds: P}."""
    out = {}
    for ds, blk in sel["datasets"].items():
        scored = []
        for cs, cand in blk["candidates"].items():
            per_h = cand["per_H_mean_mse_float32"]
            scored.append((float(per_h), int(cs)))
        scored.sort(key=lambda t: (t[0], t[1]))
        out[ds] = scored[0][1]
    return out


def cmd_freeze():
    dp_path = data_prep_path()
    sp_path = selected_p_path()
    for p in (dp_path, sp_path):
        if not os.path.isfile(p):
            sys.exit(f"[protocol] HARD STOP: freeze requires {p}")
    dp = C.load_json(dp_path)
    sel = C.load_json(sp_path)

    for ds in DOMAIN_ORDER:
        if ds not in dp["datasets"] or ds not in sel["datasets"]:
            sys.exit(f"[protocol] HARD STOP: {ds} missing from data_prep/"
                     f"selected_p")

    chosen = recompute_argmin(sel)
    declared = {ds: int(sel["datasets"][ds]["selected"])
                for ds in DOMAIN_ORDER}
    if chosen != declared:
        sys.exit(f"[protocol] HARD STOP: argmin recheck mismatch "
                 f"recomputed={chosen} declared={declared}")

    datasets_out = {}
    for ds in DOMAIN_ORDER:
        spec = DATASETS[ds]
        d = dict(spec)
        d["horizons"] = list(spec["horizons"])
        d["windows"] = dict(spec["windows"])
        d["candidates"] = list(spec["candidates"])
        d["selected_periodicity"] = chosen[ds]
        d["row_order"] = ("whole rows stably sorted ascending by parsed "
                          "date (EXP-020 protocol; sort mapping "
                          "provenance in results/base_evidence/)")
        d["dates"] = dp["datasets"][ds]["dates"]
        datasets_out[ds] = d

    proto = {
        "exp_id": EXP_ID,
        "protocol_name": PROTOCOL_NAME,
        "label": LABEL,
        "base_data": dict(BASE_DATA),
        "base_method": dict(BASE_METHOD),
        "provenance": {
            "visionts_code_md5": dict(VISIONTS_CODE_MD5),
            "visionts_repo_commit": VISIONTS_REPO_COMMIT,
            "exp018_cgtsf_token": CGTSF_TOKEN,
            "weights_file": C.CKPT_FILENAME,
            "weights_sha256": C.WEIGHTS_SHA256,
            "weights_bytes": C.WEIGHTS_BYTES,
            "weights_dir_on_dt": C.CKPT_DIR,
            "scaler": dict(SCALER),
        },
        "run_settings": dict(RUN_SETTINGS),
        "selection_rule": dict(SELECTION_RULE),
        "domain_order": list(DOMAIN_ORDER),
        "datasets": datasets_out,
        "total_test_windows": TOTAL_TEST_WINDOWS,
        "selected_periodicities": {ds: chosen[ds] for ds in DOMAIN_ORDER},
        "selected_p_sha256": C.sha256_file(sp_path),
        "argmin_recheck": "PASS",
        "data_prep_sha256": C.sha256_file(dp_path),
        "runtime": C.env_info(),
        "interpretation_boundary_notes": {
            "vs_exp020_aurora": "same ascending test windows and same "
                                "standardization; the comparison still "
                                "spans model AND input modality "
                                "(univariate numeric vs text+numeric), so "
                                "it cannot be attributed to the text "
                                "modality alone",
            "vs_exp021_raw_order": "different test period "
                                   "(2017-04..2024-03 vs 1989-01..1995-12), "
                                   "different history window, different "
                                   "standardization statistics and "
                                   "different selection basis — the "
                                   "difference describes the two protocols, "
                                   "not row order alone",
            "periodicity_unknown_upstream": "Aurora paper does not disclose "
                                            "the periodicity P / history L "
                                            "used for its VisionTS column; "
                                            "P here is re-selected on the "
                                            "ascending validation split",
        },
    }
    C.dump_json(proto, C.PROTO_PATH)
    token = hashlib.sha256(open(C.PROTO_PATH, "rb").read()).hexdigest()[:12]
    print(f"[protocol] frozen token: {token}")
    return token


def cmd_verify(token):
    if not os.path.isfile(C.PROTO_PATH):
        sys.exit("[protocol] HARD STOP: protocol.json missing")
    want = hashlib.sha256(open(C.PROTO_PATH, "rb").read()).hexdigest()[:12]
    if want != token:
        sys.exit(f"[protocol] token mismatch: {token} != {want}")
    print(f"[protocol] token OK: {token}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "freeze":
        cmd_freeze()
    elif a and a[0] == "verify":
        if len(a) < 2:
            sys.exit("usage: protocol.py verify <token>")
        cmd_verify(a[1])
    else:
        sys.exit("usage: python protocol.py freeze | verify <token>")
