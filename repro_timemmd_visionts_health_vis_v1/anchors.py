#!/usr/bin/env python3
"""Hard pins for EXP-023 (repro_timemmd_visionts_health_vis_v1).

Everything is read-only from the frozen EXP-021 experiment
(repro_timemmd_visionts_zeroshot_v1, token 06c003f25618, freeze c905d43):
its protocol.json, selected_p.json, Health predictions npz + done.json,
and the Health dataset CSV (sha-pinned). No model inference happens in
EXP-023 — frozen predictions are reused as-is.

All values below were extracted from the verified EXP-021 final pack
(temp/Aurora_TimeMMD_visionts_zeroshot_final_20261012_1352.tar.gz,
sha256 d9c6f47b…) and cross-checked against the frozen git branch
repro-timemmd-visionts-zeroshot-v1.
"""
import os

EXP_ID = "EXP-023"
PROTOCOL_NAME = "repro_timemmd_visionts_health_vis_v1"

# EXP-021 anchor (read-only base experiment)
BASE = {
    "exp_id": "EXP-021",
    "dir_name": "repro_timemmd_visionts_zeroshot_v1",
    "add_commit": "d95004d4edb70595a54ae13dd188cc3b28b5cc88",
    "freeze_commit": "c905d43db23caad732afad91dbbffc48ae693b62",
    "protocol_token": "06c003f25618",
    "protocol_sha256": "06c003f25618665914939bf641d41f217edb6320db9"
                       "baebda8c0a3dd63706eda",
    "selected_p_sha256": "91b0d5887938a32cf89272d148b5fde2a94a94065b"
                         "8552913914c25c78e1e05a",
}

BASE_DIR = os.environ.get(
    "EXP023_BASE_DIR",
    "/home/wlt/MMTS/Aurora_Experiments/repro_timemmd_visionts_zeroshot_v1")
PROTO_PATH = os.path.join(BASE_DIR, "protocol.json")
SEL_PATH = os.path.join(BASE_DIR, "results", "selected_p.json")
PER_H_CSV = os.path.join(BASE_DIR, "results", "comparison_per_h.csv")
PRED_DIR = os.path.join(BASE_DIR, "predictions")
CSV_PATH = os.path.join(BASE_DIR, "dataset", "Health.csv")

# Health dataset CSV (byte-identical to the EXP-019 base; sha-pinned in
# the EXP-021 protocol)
CSV_SHA256 = "88a81d443533991b8e2b55da8db0a6496d68fc510077b1c0f568d6f186394cf8"

# Health frozen config (== EXP-021 protocol datasets.Health block)
DS = "Health"
SEQ_LEN = 96
HORIZONS = [12, 24, 36, 48]
WINDOWS = {12: 266, 24: 254, 36: 242, 48: 230}
TOTAL_TEST_WINDOWS = sum(WINDOWS.values())  # 992
SPLIT_ROWS = [972, 140, 277]
N_ROWS = 1389
N_TRAIN = 972
VAL_END = 1112
SELECTED_P = 52
CANDIDATES = [1, 4, 52]
BATCH = 256
SEED = 2021
FREQ = "weekly"
DATES = {"first": "1997-09-29", "last": "2024-05-06",
         "val_end": "2019-01-21", "test_first": "2019-01-21",
         "test_last": "2024-05-06", "val_first": "2016-05-09"}

# Frozen prediction npz sha256 (== EXP-021 done.json npz_sha256 pins)
NPZ_SHA256 = {
    12: "54d2cad190d9b7302a56567f3c3a9ca01221f1103ea6169a68d7058c5d8a0b31",
    24: "83a1922f46abf41d3f5ee6bf74295047b3a8b1419f4c8c681bdae8aa7f5f38ba",
    36: "3da5a41f67351ce8d17ef975b7ad32e904e49df0c8d8678e81283b656063f458",
    48: "146eb15e39bfd4466bfefa84c4ed7faff88259a268cb7a2eb5d33153b833e899",
}

# Frozen group metrics, official float32 replica
# (== EXP-021 done.json metrics_float32 == comparison_per_h.csv ours f32)
FROZEN_METRICS_F32 = {
    12: {"mse": 1.9547765254974365, "mae": 0.9514692425727844},
    24: {"mse": 1.9851282835006714, "mae": 0.9490209221839905},
    36: {"mse": 2.0321850776672363, "mae": 0.9560883045196533},
    48: {"mse": 2.1119909286499023, "mae": 0.9795336127281189},
}
# 4H equal-weight mean of the per-H f32 values
# (== EXP-021 comparison_per_domain.csv Health row)
FROZEN_4H_MEAN_F32 = {"mse": 2.0210202038288116, "mae": 0.9590280205011368}

# Gates
GROUP_ATOL = 1e-6      # group recompute vs frozen f32 (expect bitwise)
MEAN_ATOL = 1e-5       # 4H equal-weight mean vs frozen
LOCAL_SMOKE = os.environ.get("EXP023_LOCAL_SMOKE") == "1"
SCALER_SMOKE_ATOL = 1e-5  # only used when LOCAL_SMOKE (fake sklearn)

# Window selection: fixed horizon, k best + k worst
SEL_H = 12
K = 5

# Plot styling (user-fixed palette)
COLOR_HIST = "#1f77b4"      # history truth — blue
COLOR_TRUTH = "#ff7f0e"     # prediction-span truth — orange
COLOR_PRED = "#d62728"      # model prediction — red dashed
COLOR_POST = "#7f7f7f"      # post-horizon truth — gray
COLOR_TEST_BG = "#b0b0b0"
DPI = 150

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIG_DIR = os.path.join(HERE, "figures")
PACK_OUT = os.path.join(HERE, "pack_out")


def npz_path(h):
    return os.path.join(PRED_DIR, f"{DS}__h{h}.npz")


def done_path(h):
    return os.path.join(PRED_DIR, f"{DS}__h{h}.done.json")
