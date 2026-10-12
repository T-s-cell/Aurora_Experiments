#!/usr/bin/env python3
"""EXP-021 frozen protocol (repro_timemmd_visionts_zeroshot_v1).

Data protocol is byte-identical to the frozen EXP-019 base
(repro_timemmd_zeroshot_v1, token 32b843660cc3); only the model changes:
vendored Keytoyze VisionTS (mae_base, finetune 'none', ImageNet MAE-Base
weights) with periodicity P selected per domain on the validation set by
the VisionTS paper's own method.

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

EXP_ID = "EXP-021"
PROTOCOL_NAME = "repro_timemmd_visionts_zeroshot_v1"
LABEL = ("Aurora 官方数据协议（TimeMMD 原始行序）、原版 VisionTS zero-shot "
         "复评 v1（按 VisionTS 论文方法在验证集选 P）")

BASE = {
    "exp_id": "EXP-019",
    "dir": "repro_timemmd_zeroshot_v1",
    "add_commit": "36a92942366e9185bc1356e6a0d2e3ba1b459a0a",
    "freeze_commit": "8878412fd5ab68ff97143bca71c9c2de4e7bcdfb",
    "protocol_sha256": ("32b843660cc3434cb32e350f24e39985e6d855eebc0edf"
                        "ec4aa4ba7f442e2b3d"),
    "protocol_token": "32b843660cc3",
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

DOMAIN_ORDER = ["Agriculture", "Climate", "Economy", "Energy",
                "Environment", "Health", "Security", "Traffic",
                "SocialGood"]

DATASETS = {
    "Agriculture": {
        "seq_len": 192, "horizons": [6, 8, 10, 12],
        "windows": {"6": 94, "8": 92, "10": 90, "12": 88},
        "candidates": [1, 3, 6, 12], "frequency": "monthly",
        "split_rows": [347, 50, 99],
        "csv_sha256": ("41203f1af2c4fd9f3207ec286f0a1462df146b5ae8b6d9cd"
                       "cec6cb569c57a0be")},
    "Climate": {
        "seq_len": 192, "horizons": [6, 8, 10, 12],
        "windows": {"6": 94, "8": 92, "10": 90, "12": 88},
        "candidates": [1, 3, 6, 12], "frequency": "monthly",
        "split_rows": [347, 50, 99],
        "csv_sha256": ("4096994d5e8691de404b15d16ab9e2a98d69b03eb1d4eaf"
                       "a4283ea44ead0e51b")},
    "Economy": {
        "seq_len": 192, "horizons": [6, 8, 10, 12],
        "windows": {"6": 79, "8": 77, "10": 75, "12": 73},
        "candidates": [1, 3, 6, 12], "frequency": "monthly",
        "split_rows": [296, 43, 84],
        "csv_sha256": ("c35325f0181a426a78c35b9172d8908f64639796a535c5fd"
                       "c8b082b538030e5d")},
    "Energy": {
        "seq_len": 1056, "horizons": [12, 24, 36, 48],
        "windows": {"12": 284, "24": 272, "36": 260, "48": 248},
        "candidates": [1, 4, 52], "frequency": "weekly",
        "split_rows": [1035, 149, 295],
        "csv_sha256": ("fdef4b4f58ab319656b60e49c96d1dc4158d2c5b17467456"
                       "d4d12d746ff55cd5")},
    "Environment": {
        "seq_len": 528, "horizons": [48, 96, 192, 336],
        "windows": {"48": 3002, "96": 2954, "192": 2858, "336": 2714},
        "candidates": [1, 7, 30, 365], "frequency": "daily",
        "split_rows": [10673, 1526, 3049],
        "csv_sha256": ("3bcda88e0831294678d412807fa2eac2022c5665a9f1f044"
                       "cd23cff65657a0d6")},
    "Health": {
        "seq_len": 96, "horizons": [12, 24, 36, 48],
        "windows": {"12": 266, "24": 254, "36": 242, "48": 230},
        "candidates": [1, 4, 52], "frequency": "weekly",
        "split_rows": [972, 140, 277],
        "csv_sha256": ("88a81d443533991b8e2b55da8db0a6496d68fc510077b1c0"
                       "f568d6f186394cf8")},
    "Security": {
        "seq_len": 220, "horizons": [6, 8, 10, 12],
        "windows": {"6": 54, "8": 52, "10": 50, "12": 48},
        "candidates": [1, 3, 6, 12], "frequency": "monthly",
        "split_rows": [207, 31, 59],
        "csv_sha256": ("adea376a0e6858979b2d42cde6f3901d7d60639bb78cd64e"
                       "ec6f58e2e85ed23c")},
    "Traffic": {
        "seq_len": 96, "horizons": [6, 8, 10, 12],
        "windows": {"6": 101, "8": 99, "10": 97, "12": 95},
        "candidates": [1, 3, 6, 12], "frequency": "monthly",
        "split_rows": [371, 54, 106],
        "csv_sha256": ("82439ff69196789968f48245d8bce699ac09a566f24f027d"
                       "abd6b99333417311")},
    "SocialGood": {
        "seq_len": 192, "horizons": [6, 8, 10, 12],
        "windows": {"6": 173, "8": 171, "10": 169, "12": 167},
        "candidates": [1, 3, 6, 12], "frequency": "monthly",
        "split_rows": [624, 90, 178],
        "csv_sha256": ("a6f25ea421d1e74b9c7aa29b0119c2dcd49bcd5b6c266ab7"
                       "cd4ba498a5f814cf")},
}

TOTAL_TEST_WINDOWS = sum(sum(d["windows"].values()) for d in
                        DATASETS.values())
assert TOTAL_TEST_WINDOWS == 15892, TOTAL_TEST_WINDOWS

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
    "source": "VisionTS paper (Keytoyze) A.2 / Table 5 candidate sets by "
              "dataset frequency",
    "candidates_by_frequency": {"monthly": [1, 3, 6, 12],
                                "weekly": [1, 4, 52],
                                "daily": [1, 7, 30, 365]},
    "metric": "standardized-space validation MSE, official float32 "
              "replica (np.mean default dtype); float64 kept as appendix",
    "aggregate": "equal-weight mean of the 4 horizon MSEs per candidate "
                 "P, then argmin; ties resolved to the SMALLER P",
    "val_windows": "s in [max(n_train, seq_len), val_end - H] closed; "
                   "hist [s-L, s), label [s, s+H); stride 1; no test row "
                   "is ever touched",
    "weight_updates": "none — model stays in eval() with all parameters "
                      "frozen during selection",
    "shared_by_test": "the single selected P per domain is used for all "
                      "4 test horizons",
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
        d["row_order"] = "original CSV order (Economy kept year-descending)"
        d["dates"] = dp["datasets"][ds]["dates"]
        datasets_out[ds] = d

    proto = {
        "exp_id": EXP_ID,
        "protocol_name": PROTOCOL_NAME,
        "label": LABEL,
        "base": dict(BASE),
        "provenance": {
            "visionts_code_md5": dict(VISIONTS_CODE_MD5),
            "visionts_repo_commit": VISIONTS_REPO_COMMIT,
            "exp018_cgtsf_token": CGTSF_TOKEN,
            "weights_file": C.CKPT_FILENAME,
            "weights_sha256": C.WEIGHTS_SHA256,
            "weights_bytes": C.WEIGHTS_BYTES,
            "weights_dir_on_dt": C.CKPT_DIR,
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
        "paper_deviation_notes": {
            "past_tokens": "paper A.4: past tokens maintained at a "
                           "constant value of 11; official script instead "
                           "pins per-dataset seq_len (this run follows the "
                           "script, same as EXP-019)",
            "periodicity_unknown": "Aurora paper does not disclose the "
                                   "periodicity P / history length L it "
                                   "used for its VisionTS column; this run "
                                   "selects P by the VisionTS paper's own "
                                   "validation-set method — a re-scoring "
                                   "under the same data protocol, NOT a "
                                   "bit-identical replication of the "
                                   "paper's VisionTS numbers",
            "environment_rows": "paper Table 7 says 11,102; actual CSV has "
                                "15,248",
            "socialgood_rows": "paper Table 7 says 900; actual CSV has 892",
            "economy_order": "CSV row order is year-descending blocks "
                             "(2024 first, 1989 last); kept as-is per the "
                             "frozen EXP-019 base protocol",
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
