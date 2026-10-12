#!/usr/bin/env python3
"""Frozen protocol for repro_timemmd_economy_chrono_v1 (EXP-020).

Aurora TimeMMD multimodal zero-shot on the Economy domain with rows
sorted chronologically (stable ascending by parsed `date`), everything
else identical to the frozen EXP-019 official reproduction
(repro_timemmd_zeroshot_v1, token 32b843660cc3).

Usage:
  python protocol.py check_official   # verify vendored official/ md5s
  python protocol.py freeze           # requires results/batch_decision.json
                                      # and results/data_prep.json;
                                      # writes protocol.json, prints token
  python protocol.py verify <token>   # recompute token from protocol.json

The base experiment's frozen protocol is hashed (and its token
recomputed) at freeze time; the sort mapping is hashed from
results/sort_mapping.csv written by prepare_data.py BEFORE freeze.
Token = sha256(protocol.json bytes)[:12].
"""
import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C

PROTOCOL_NAME = "repro_timemmd_economy_chrono_v1"
LABEL = "Aurora 官方脚本、TimeMMD Economy 时间升序对照、zero-shot v1"
EXP_ID = "EXP-020"

# ---- base experiment (EXP-019) anchors --------------------------------
BASE_DIR_NAME = "repro_timemmd_zeroshot_v1"
BASE_EXP_ID = "EXP-019"
BASE_ADD_COMMIT = "36a92942366e9185bc1356e6a0d2e3ba1b459a0a"
BASE_FREEZE_COMMIT = "8878412fd5ab68ff97143bca71c9c2de4e7bcdfb"
BASE_PROTOCOL_TOKEN = "32b843660cc3"
BASE_DIR = os.path.join(os.path.dirname(HERE), BASE_DIR_NAME)

# ---- upstream anchors (identical to EXP-019) --------------------------
OFFICIAL_REPO_COMMIT = "a247760abbc9d17a861bc365c032368d317815f2"
HF_REPO = "DecisionIntelligence/Aurora"
HF_REVISION = "c495b02c1b151be52a3c174237ed240aa66e6384"
WEIGHTS_SHA256 = ("df2fb96852a59515a14552d5bddc35c03588b6a8bea69355984b3d"
                  "d926a72b58")
WEIGHTS_BYTES = 843564328
HF_CONFIG_SHA256 = ("37f1f42f3b7cb856691985aa44dd3c59b86c481f31fec874a06a"
                    "3a0800bf8985")
HF_GENERATION_CONFIG_SHA256 = ("d90f7f1d9ef012f9ec0bd76fdf42e6979d086f15"
                               "7d65910b3b273edfb100e748")

# ---- original Economy CSV (EXP-019 frozen pin) ------------------------
ORIGINAL_CSV_SHA256 = ("c35325f0181a426a78c35b9172d8908f64639796a535c5fdc"
                       "8b082b538030e5d")

# ---- frozen run settings ----------------------------------------------
SEED = 2021
NUM_SAMPLES = 100
MAX_TEXT_TOKEN = 500
LABEL_LEN = 0
FEATURES = "S"
TARGET = "OT"
PREFERRED_BATCH = 256
NUM_WORKERS = 4
BATCH_LADDER = (256, 128, 64, 32, 16)

# ---- official source pin (same vendored tree as EXP-019) ---------------
OFFICIAL_MD5 = {
    "aurora/__init__.py": "1e96fdddcd7b86964c6bebaa99217ce9",
    "aurora/bert_config/config.json": "0939b914fc32135f6c12d8ef281dbd7a",
    "aurora/bert_config/tokenizer.json": "eac14313ee1816d9e8ef962f7615fd93",
    "aurora/bert_config/tokenizer_config.json": "aee76cec0170859d909d411dc7cdaa76",
    "aurora/bert_config/vocab.txt": "64800d5d8528ce344256daf115d4965e",
    "aurora/config.json": "505a1b93de20bd9bc9e304a692d7bb0b",
    "aurora/configuration_aurora.py": "452cbcf5a954e5a83ef78df5b778df9d",
    "aurora/flow_loss.py": "4bc657919426405fa0ce999dc275da53",
    "aurora/generation_config.json": "64e8d0b0b84cc46cb1e7c00740e54dd1",
    "aurora/modality_connector.py": "49fbbbc03575a554c83bc1a107c3bdc8",
    "aurora/modeling_aurora.py": "16cc50855050e1d60dc2ae4ce15e3dd0",
    "aurora/prototype_retriever.py": "77e0438345684dc26f5b290c6a4a34f0",
    "aurora/ts_generation_mixin.py": "faad71b0796b01ea69a3177c32520cd0",
    "aurora/util_functions.py": "546a7518b1d36bd57092962adf0fe040",
    "aurora/vit_config/config.json": "1efb30ac5009825d92ad2e9aead6b5c0",
    "aurora/vit_config/preprocessor_config.json": "9f21ed69c720eee2a5181343f526d5c5",
    "data_provider/data_factory.py": "59b7e4011fb3425ba1b5ab8bec712072",
    "data_provider/data_loader.py": "b5790b00f4057c16b6399aa5ce36f703",
    "exp/exp_basic.py": "54d5d7e26de3fc4940088429f365e6c4",
    "exp/exp_main.py": "54873e0347367b45cbf49ef0dc428e11",
    "layers/AutoCorrelation.py": "522c914ac43665a64ec2cf63cbed5bde",
    "layers/Autoformer_EncDec.py": "aeb888302d1e84dde9595f279aa841e0",
    "layers/Embed.py": "dd4981476d66920403ceb99fe82a5777",
    "layers/PatchTST_backbone.py": "7c75f07285bbee5b42fa5104c55071d1",
    "layers/PatchTST_layers.py": "0cab2689ef2005b16254acbeff1c3ee5",
    "layers/RevIN.py": "2ad3ca99ea95f624031241a9cdf890d0",
    "layers/SelfAttention_Family.py": "018c948b328153947c9e6afa89bf858b",
    "layers/Transformer_EncDec.py": "d348ca1e53a95b50d18d4957b5ca589c",
    "models/Autoformer.py": "99c5c32046672f55b0e94fd7012d2008",
    "models/DLinear.py": "d2a260d9629be3e6a2a87b1b2939d7b2",
    "models/Informer.py": "95ff987a845c701c8090124324ae8f56",
    "models/Linear.py": "d234d59b76529ba43b9c11e6195ed4de",
    "models/NLinear.py": "2efde899865ad83959161128285dd6e9",
    "models/PatchTST.py": "8830566e1934e9f44c88e04bd6062b56",
    "models/Stat_models.py": "9901e685295237bf398b0d78c4fc37ed",
    "models/Transformer.py": "ac9810eab36f4d617ed3225d6a545fb7",
    "run_longExp.py": "46b3e492f498986cb074dbc4ce016d41",
    "scripts/run_aurora_timemmd_zero_shot.sh": "45d0d8d3a6e2705b1ff6f1752c78cc32",
    "utils/masking.py": "973dad6ad8c5e312cc7d518c9a105867",
    "utils/metrics.py": "82a75f67eb1bdeca5bf1a2279b898dcc",
    "utils/timefeatures.py": "ce8cc920c8700fd240e52c8cc67ceb99",
    "utils/tools.py": "891d5456598dec067bd127c89768b65d",
}
ORIGINAL_EXP_MAIN_MD5 = "bcc775a2494fcfb98f39e60717daad56"

# ---- dataset pin (Economy only, chronological ascending) ---------------
# horizons list = RUN order (H=12 first as the validation group);
# presentation order is numeric (6,8,10,12).
# windows = num_test(84) - H + 1 on the sorted 423-row CSV.
DATASETS = {
    "Economy": {
        "seq_len": 192, "token_len": 48, "horizons": (12, 6, 8, 10),
        "windows": {6: 79, 8: 77, 10: 75, 12: 73},
        "order_assert": {"Month": "January 1989", "date": "1989-01-01",
                         "last_Month": "March 2024",
                         "last_date": "2024-03-01"},
    },
}
DOMAIN_ORDER = ("Economy",)
TOTAL_WINDOWS = 304

# ---- official split on the sorted CSV (verified in split_report) -------
SPLIT_CLAIMS = {
    "n_rows": 423,
    "num_train": 296, "train_dates": ["1989-01-01", "2013-08-01"],
    "num_val": 43, "val_dates": ["2013-09-01", "2017-03-01"],
    "num_test": 84, "test_dates": ["2017-04-01", "2024-03-01"],
    "test_border1": 147,
    "first_test_pred_row": 339,
}

SORT_RULE = ("stable ascending sort of whole rows by parsed `date` "
             "column (ISO yyyy-mm-dd, unique month-first days); all "
             "columns and cell contents move with their row; no field "
             "regenerated")

PAPER_DEVIATION_NOTES = {
    "past_tokens": ("paper A.4: past tokens maintained at a constant value "
                    "of 11; official script instead pins per-dataset seq_len "
                    "(this run follows the script)"),
    "single_protocol_change": ("this experiment differs from frozen EXP-019 "
                               "ONLY in Economy row order (chronological "
                               "ascending vs original year-descending "
                               "blocks); history windows, text selection, "
                               "train/test periods and scaler statistics "
                               "all change as a consequence"),
    "period_shift": ("EXP-019 original-row-order test target rows cover "
                     "1989-01..1995-12; EXP-020 ascending test target rows "
                     "cover 2017-04..2024-03"),
}


def csv_path(ds):
    return os.path.join(C.DATASET_DIR, f"{ds}.csv")


def base_protocol_path():
    return os.path.join(BASE_DIR, "protocol.json")


def check_official():
    bad = []
    found = set()
    for root, dirs, names in os.walk(C.OFFICIAL_TMM):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for n in sorted(names):
            if n.endswith(".pyc"):
                continue
            rel = os.path.relpath(os.path.join(root, n), C.OFFICIAL_TMM)
            found.add(rel)
    expect = set(OFFICIAL_MD5)
    if found != expect:
        bad.append(f"file set mismatch: missing={sorted(expect-found)} "
                   f"extra={sorted(found-expect)}")
    for rel, want in sorted(OFFICIAL_MD5.items()):
        p = os.path.join(C.OFFICIAL_TMM, rel)
        got = C.md5_file(p)
        if got != want:
            bad.append(f"{rel}: md5 {got} != pinned {want}")
    if bad:
        for b in bad:
            print(f"[protocol] OFFICIAL MISMATCH: {b}")
        sys.exit(2)
    print(f"[protocol] official/ {len(expect)} files md5 all match "
          f"(commit {OFFICIAL_REPO_COMMIT[:12]}, exp_main patched "
          f"{OFFICIAL_MD5['exp/exp_main.py'][:8]}, pristine "
          f"{ORIGINAL_EXP_MAIN_MD5[:8]})")


def check_batch_decision():
    bd = C.load_json(C.BATCH_DECISION_PATH)
    if not bd.get("ok"):
        sys.exit("[protocol] batch_decision.json not ok; run preflight first")
    plan = bd["batch_per_domain"]
    for ds in DOMAIN_ORDER:
        if ds not in plan:
            sys.exit(f"[protocol] batch_decision missing domain {ds}")
        if plan[ds] not in BATCH_LADDER:
            sys.exit(f"[protocol] {ds} batch {plan[ds]} not in ladder")
    return plan


def check_data_prep():
    dp = C.load_json(os.path.join(C.RESULTS_DIR, "data_prep.json"))
    if not dp.get("ok"):
        sys.exit("[protocol] data_prep.json not ok; run prepare_data first")
    if dp["original_csv_sha256"] != ORIGINAL_CSV_SHA256:
        sys.exit("[protocol] data_prep original csv sha != EXP-019 pin")
    if C.sha256_file(csv_path("Economy")) != dp["sorted_csv_sha256"]:
        sys.exit("[protocol] sorted Economy.csv sha drift vs data_prep.json")
    if C.sha256_file(os.path.join(C.RESULTS_DIR, "sort_mapping.csv")) != \
            dp["sort_mapping_sha256"]:
        sys.exit("[protocol] sort_mapping.csv sha drift vs data_prep.json")
    return dp


def check_base():
    bp = base_protocol_path()
    if not os.path.isfile(bp):
        sys.exit(f"[protocol] base protocol missing: {bp}")
    sha = C.sha256_file(bp)
    token = hashlib.sha256(open(bp, "rb").read()).hexdigest()[:12]
    if token != BASE_PROTOCOL_TOKEN:
        sys.exit(f"[protocol] base protocol token {token} != "
                 f"{BASE_PROTOCOL_TOKEN} — base dir drifted")
    return {"protocol_sha256": sha, "token": token}


def freeze():
    check_official()
    batch_plan = check_batch_decision()
    dp = check_data_prep()
    base = check_base()
    cfg = DATASETS["Economy"]
    proto = {
        "protocol_name": PROTOCOL_NAME,
        "label": LABEL,
        "exp_id": EXP_ID,
        "designation": "Aurora 官方脚本、TimeMMD Economy 时间升序对照、zero-shot v1",
        "official_repo_commit": OFFICIAL_REPO_COMMIT,
        "official_file_count": len(OFFICIAL_MD5),
        "original_exp_main_md5": ORIGINAL_EXP_MAIN_MD5,
        "hf_repo": HF_REPO,
        "hf_revision": HF_REVISION,
        "weights_sha256": WEIGHTS_SHA256,
        "weights_bytes": WEIGHTS_BYTES,
        "hf_config_sha256": HF_CONFIG_SHA256,
        "hf_generation_config_sha256": HF_GENERATION_CONFIG_SHA256,
        "seed": SEED,
        "num_samples": NUM_SAMPLES,
        "max_text_token": MAX_TEXT_TOKEN,
        "label_len": LABEL_LEN,
        "features": FEATURES,
        "target": TARGET,
        "preferred_batch": PREFERRED_BATCH,
        "batch_ladder": list(BATCH_LADDER),
        "num_workers": NUM_WORKERS,
        "batch_plan": batch_plan,
        "batch_decision_source": C.BATCH_DECISION_PATH,
        "datasets": {"Economy": {
            "seq_len": cfg["seq_len"], "token_len": cfg["token_len"],
            "horizons": list(cfg["horizons"]),
            "windows": cfg["windows"],
            "order_assert": cfg["order_assert"],
            "csv_sha256": dp["sorted_csv_sha256"],
            "original_csv_sha256": dp["original_csv_sha256"],
            "sort_mapping_sha256": dp["sort_mapping_sha256"],
            "sort_rule": SORT_RULE,
            "n_rows": dp["n_rows"],
        }},
        "domain_order": list(DOMAIN_ORDER),
        "total_windows": TOTAL_WINDOWS,
        "split_claims": SPLIT_CLAIMS,
        "base_experiment": {
            "dir": BASE_DIR_NAME,
            "exp_id": BASE_EXP_ID,
            "add_commit": BASE_ADD_COMMIT,
            "freeze_commit": BASE_FREEZE_COMMIT,
            "protocol_sha256": base["protocol_sha256"],
            "protocol_token": base["token"],
        },
        "paper_deviation_notes": PAPER_DEVIATION_NOTES,
    }
    C.dump_json(proto, C.PROTO_PATH)
    token = hashlib.sha256(open(C.PROTO_PATH, "rb").read()).hexdigest()[:12]
    print(f"[protocol] frozen -> {C.PROTO_PATH}")
    print(f"[protocol] approval token: {token}")
    return token


def verify_frozen(token):
    if not os.path.isfile(C.PROTO_PATH):
        sys.exit("[protocol] protocol.json missing")
    want = hashlib.sha256(open(C.PROTO_PATH, "rb").read()).hexdigest()[:12]
    if want != token:
        sys.exit(f"[protocol] token mismatch: {token} != {want}")
    print(f"[protocol] token OK: {token}")
    return C.load_json(C.PROTO_PATH)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check_official", "freeze", "verify"])
    ap.add_argument("token", nargs="?", default=None)
    a = ap.parse_args()
    if a.cmd == "check_official":
        check_official()
    elif a.cmd == "freeze":
        freeze()
    else:
        if not a.token:
            sys.exit("usage: protocol.py verify <token>")
        verify_frozen(a.token)
