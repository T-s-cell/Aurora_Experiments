#!/usr/bin/env python3
"""Frozen protocol: Aurora-timesx-all-native-A48-v1.

Aurora A48 unimodal zero-shot on ALL 8,106 TimesX native windows (190 vars /
19 domains, past96 -> target12), three base seeds 2021/2022/2023, window-for-
window aligned with the frozen VisionTS EXP-014 predictions. Old EXP-010 A48
cache (LN test 2,474 windows x 3 seeds) is reused only where fully verified;
everything else is re-inferred. Old Aurora repo and VisionTS EXP-014 are
READ-ONLY (path import + md5 records).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)  # /home/wlt/MMTS/Aurora_Experiments (old EXP-010 project, read-only)
OLD_DIR = REPO_ROOT
VTS_DIR = "/home/wlt/MMTS/VisionTS_Experiments/repro_zeroshot_all_timesx_v1"
LN_MANIFEST_SOURCE = ("/home/wlt/MMTS/VisionTS_Experiments/repro_ln_timesx_v1"
                      "/manifests/split_manifest.json")  # frozen into visionts_ref/ at S1a
LN_MANIFEST = os.path.join(HERE, "visionts_ref", "split_manifest.json")

PROTOCOL_VERSION = "Aurora-timesx-all-native-A48-v1"
METHOD = "A48"

# ---- frozen VisionTS EXP-014 references (sha256 pinned; commit 7b434dd) ----
VTS_INVENTORY_SHA256 = "0cb6e73971b0f8b54077ff3df5a332f7dbe34493b337ab0a2b9744aa88c217de"
VTS_ARRAYS_SHA256 = "fb0c0c454afd8f7b9f276b99704450bdc6c99a454c8d774ea8846f729fa9423d"
VTS_VARS_SHA256 = "22018d215b0dfe8ccc842846cf5e04bb2050828d993548728c72215681267068"
VTS_PREDS_SHA256 = "0ceeb654c3dc78e86aca9c56f7a7ee1f5c8db590785cd23a7e37eb938de041b0"

# ---- counts ----
N_VARS = 190
N_DOMAINS = 19
TOTAL_WINDOWS = 8106
N_LN_TEST = 2474          # fixed LN-test subset size (identity), NOT the reuse count
FILL_EXPECT_FULL_REUSE = 5632   # per seed when all 2,474 cached windows are reusable
FILL_TOTAL_EXPECT_FULL_REUSE = 16896
PRED_TOTAL = 24318
SEEDS = (2021, 2022, 2023)
CONTEXT_LEN = 96
PRED_LEN = 12
EPS_STD = 1e-8
ZERO_VAR_EXPECTED = 4     # constant-history windows; none in LN test -> preflight new_only

EXPECTED_DOMAINS = sorted([
    'CropsAndStaples', 'Currency', 'EnergyAndFuels', 'LivestockAndFoodProducts',
    'RawMaterialsAndConstruction', 'SpecialtyAndAdvancedMaterials',
    'StrategicAndHighValueMaterials', 'arts', 'climate', 'economy',
    'electronic_technology', 'finance', 'pets', 'public_health',
    'public_policy', 'science', 'shopping', 'society', 'traffic',
])
assert len(EXPECTED_DOMAINS) == N_DOMAINS

# ---- old EXP-010 pins (provenance + reuse verification; never rewritten) ----
OLD_ZIP_SHA256 = "9adb9540733c128e05e774e81cf0e238a9cc80b37e99da0c883bee9f57988bcc"
OLD_SPLIT_MANIFEST_SHA256 = "b47feebfe5e7fbf6c650ce2d23735794d853b515774c43c658724cf8328363a9"
OLD_DATA_CACHE_SHA256 = "f83924d9f459da096c45ccd880f204efc095fbed5513794a5d706d71371881aa"
OLD_HEADER_CODE_FILES = ["load_aurora.py", "data_loader.py", "predict.py", "aggregate.py"]

# ---- model / inference (frozen A48 config, identical to EXP-010 forward) ----
WEIGHTS_SHA256 = "df2fb96852a59515a14552d5bddc35c03588b6a8bea69355984b3dd926a72b58"
WEIGHTS_BYTES = 843564328
HF_REPO = "DecisionIntelligence/Aurora"
HF_REVISION = "c495b02c1b151be52a3c174237ed240aa66e6384"
ITL = 48
NUM_SAMPLES = 100
BATCH_SIZE = 1
REVIN = True
MAX_OUTPUT_LENGTH = 12
AURORA_PACKAGE_VERSION = "0.2.0"

SEED_PREFIX = "aurora-timesx-v1"
SEED_RULE = ("w = int.from_bytes(sha256(f'aurora-timesx-v1|{base_seed}|{var_key}|"
             "{sample_id}'.encode()).digest()[:8], 'big') % (2**31 - 1); "
             "torch.manual_seed(w) immediately before each window's generate(); "
             "no other RNG consumption in between")

# determinism (asserted at runtime on the GPU host)
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

# ---- preflight dual criteria (cache-compatibility engineering tolerances) ----
TOL_PRED_REL = 1e-5       # c1: max|new-old| / max(1e-8, max|old|)
TOL_PRED_OVER_D = 1e-4    # c2: max|new-old| / d   (d = frozen scoring denominator)
PF_VARS_PER_DOMAIN = 2
PF_WINDOWS_PER_VAR = 5

# ---- references / tolerances ----
REF_VTS_STD_MSE = 4.2122984787894
REF_VTS_STD_MAE = 0.856010666064058
REF_VTS_TOL = 1e-12       # relative, denominator max(1e-12, |ref|)
REF_A48_2474_STD_MSE = 4.293598
REF_A48_2474_STD_MSE_STD = 0.001592
REF_A48_2474_STD_MAE = 1.200945
REF_A48_2474_STD_MAE_STD = 0.000305
REF_A48_TOL = 5e-7        # six-decimal reproduction (mean AND seed-std)
RECOMPUTE_TOL_D = 1e-12
RECOMPUTE_TOL_VAR = 1e-9
RECOMPUTE_TOL_AGG = 1e-12

# ---- code fingerprints ----
NEW_CODE_FORWARD = ["protocol.py", "common.py", "preflight.py", "infer_fill.py"]
NEW_CODE_ALL = NEW_CODE_FORWARD + [
    "stage_guard.py", "freeze_ref.py", "verify_cache.py", "assemble.py",
    "evaluate.py", "recompute_check.py", "verify_acceptance.py", "make_report.py",
]
OLD_CODE_REUSED = ["predict.py", "load_aurora.py", "data_loader.py"]   # forward path (read-only import)
OLD_CODE_INFO = ["aggregate.py"]                                       # provenance only

OLD_PRED_DIR = os.path.join(REPO_ROOT, "predictions")
OLD_DATA_DIR = os.path.join(REPO_ROOT, "data")


def old_shard_name(domain, seed):
    return f"{METHOD}__{domain}__s{seed}__test.npz"
