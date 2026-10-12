#!/usr/bin/env python3
"""Frozen protocol for repro_timemmd_zeroshot_v1
(Aurora TimeMMD multimodal zero-shot reproduction, original row order).

Usage:
  python protocol.py check_official   # verify vendored official/ md5s
  python protocol.py freeze           # requires results/batch_decision.json;
                                      # writes protocol.json, prints token
  python protocol.py verify <token>   # recompute token from protocol.json

Official entry semantics (TimeMMD @ a247760) are referenced, never re-defined:
batch plan comes from the preflight probe (written BEFORE freeze), everything
else is pinned below. Token = sha256(protocol.json bytes)[:12].
"""
import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C

PROTOCOL_NAME = "repro_timemmd_zeroshot_v1"
LABEL = "Aurora TimeMMD 官方公开实现零样本复现（原始行序）"

# ---- upstream anchors -------------------------------------------------
OFFICIAL_REPO_COMMIT = "a247760abbc9d17a861bc365c032368d317815f2"
HF_REPO = "DecisionIntelligence/Aurora"
HF_REVISION = "c495b02c1b151be52a3c174237ed240aa66e6384"
WEIGHTS_SHA256 = ("df2fb96852a59515a14552d5bddc35c03588b6a8bea69355984b3d"
                  "d926a72b58")
WEIGHTS_BYTES = 843564328
# fetched from the same pinned revision via hf-mirror; byte-identical to
# the vendored aurora/ defaults (md5 505a1b93… / 64e8d0b0…)
HF_CONFIG_SHA256 = ("37f1f42f3b7cb856691985aa44dd3c59b86c481f31fec874a06a"
                    "3a0800bf8985")
HF_GENERATION_CONFIG_SHA256 = ("d90f7f1d9ef012f9ec0bd76fdf42e6979d086f15"
                               "7d65910b3b273edfb100e748")
EXP017_ARCHIVE_COMMIT = "3d20862a35fce5854ab2bd2bc6b0e849bd67da57"

# ---- frozen run settings ---------------------------------------------
SEED = 2021                 # run_longExp.py --random_seed default
NUM_SAMPLES = 100           # official test(): generate(num_samples=100)
MAX_TEXT_TOKEN = 500        # official max_text_token_length / BERT 500
LABEL_LEN = 0
FEATURES = "S"
TARGET = "OT"
PREFERRED_BATCH = 256       # official script batch; probe may lower it
NUM_WORKERS = 4             # dt host RAM is 15G; official default 10
BATCH_LADDER = (256, 128, 64, 32, 16)

# ---- official source pin ---------------------------------------------
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
# exp/exp_main.py above is the PATCHED md5; the pristine upstream value is
# recorded here and must be reproducible via patches/exp_main_pred_save.diff.
ORIGINAL_EXP_MAIN_MD5 = "bcc775a2494fcfb98f39e60717daad56"

# ---- dataset pin (9 domains, original row order) ----------------------
# seq_len / inference_token_len from the official zero-shot script assoc
# array; expected windows = int(0.2*N) - H + 1 (official border logic).
DATASETS = {
    "Agriculture": {
        "seq_len": 192, "token_len": 48, "horizons": (6, 8, 10, 12),
        "windows": {6: 94, 8: 92, 10: 90, 12: 88},
        "csv_sha256": "41203f1af2c4fd9f3207ec286f0a1462df146b5ae8b6d9cdcec6cb569c57a0be",
    },
    "Climate": {
        "seq_len": 192, "token_len": 48, "horizons": (6, 8, 10, 12),
        "windows": {6: 94, 8: 92, 10: 90, 12: 88},
        "csv_sha256": "4096994d5e8691de404b15d16ab9e2a98d69b03eb1d4eafa4283ea44ead0e51b",
    },
    "Economy": {
        "seq_len": 192, "token_len": 48, "horizons": (6, 8, 10, 12),
        "windows": {6: 79, 8: 77, 10: 75, 12: 73},
        "csv_sha256": "c35325f0181a426a78c35b9172d8908f64639796a535c5fdc8b082b538030e5d",
        "order_assert": {"Month": "January 2024", "date": "2024-01-01",
                         "last_Month": "December 1989"},
    },
    "Energy": {
        "seq_len": 1056, "token_len": 48, "horizons": (12, 24, 36, 48),
        "windows": {12: 284, 24: 272, 36: 260, 48: 248},
        "csv_sha256": "fdef4b4f58ab319656b60e49c96d1dc4158d2c5b17467456d4d12d746ff55cd5",
    },
    "Environment": {
        "seq_len": 528, "token_len": 48, "horizons": (48, 96, 192, 336),
        "windows": {48: 3002, 96: 2954, 192: 2858, 336: 2714},
        "csv_sha256": "3bcda88e0831294678d412807fa2eac2022c5665a9f1f044cd23cff65657a0d6",
    },
    "Health": {
        "seq_len": 96, "token_len": 48, "horizons": (12, 24, 36, 48),
        "windows": {12: 266, 24: 254, 36: 242, 48: 230},
        "csv_sha256": "88a81d443533991b8e2b55da8db0a6496d68fc510077b1c0f568d6f186394cf8",
    },
    "Security": {
        "seq_len": 220, "token_len": 24, "horizons": (6, 8, 10, 12),
        "windows": {6: 54, 8: 52, 10: 50, 12: 48},
        "csv_sha256": "adea376a0e6858979b2d42cde6f3901d7d60639bb78cd64eec6f58e2e85ed23c",
    },
    "SocialGood": {
        "seq_len": 192, "token_len": 48, "horizons": (6, 8, 10, 12),
        "windows": {6: 173, 8: 171, 10: 169, 12: 167},
        "csv_sha256": "a6f25ea421d1e74b9c7aa29b0119c2dcd49bcd5b6c266ab7cd4ba498a5f814cf",
    },
    "Traffic": {
        "seq_len": 96, "token_len": 48, "horizons": (6, 8, 10, 12),
        "windows": {6: 101, 8: 99, 10: 97, 12: 95},
        "csv_sha256": "82439ff69196789968f48245d8bce699ac09a566f24f027dabd6b99333417311",
    },
}
DOMAIN_ORDER = ("Agriculture", "Climate", "Economy", "Energy", "Environment",
                "Health", "Security", "Traffic", "SocialGood")  # official script order
TOTAL_WINDOWS = 15892

# paper-side claims recorded for the deviation report (NOT run settings)
PAPER_DEVIATION_NOTES = {
    "past_tokens": ("paper A.4: past tokens maintained at a constant value "
                    "of 11; official script instead pins per-dataset seq_len "
                    "(this run follows the script)"),
    "environment_rows": "paper Table 7 says 11,102; actual CSV has 15,248",
    "socialgood_rows": "paper Table 7 says 900; actual CSV has 892",
    "economy_order": ("CSV row order is year-descending blocks (2024 first, "
                      "1989 last), ascending within year; kept as-is and "
                      "results labeled 'original row order'"),
}


def csv_path(ds):
    return os.path.join(C.DATASET_DIR, f"{ds}.csv")


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


def freeze():
    check_official()
    batch_plan = check_batch_decision()
    csvs = {}
    for ds, cfg in DATASETS.items():
        p = csv_path(ds)
        if not os.path.isfile(p):
            sys.exit(f"[protocol] missing CSV {p}")
        got = C.sha256_file(p)
        if got != cfg["csv_sha256"]:
            sys.exit(f"[protocol] {ds}.csv sha256 {got} != pinned")
        csvs[ds] = got
    proto = {
        "protocol_name": PROTOCOL_NAME,
        "label": LABEL,
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
        "datasets": {ds: {"seq_len": c["seq_len"], "token_len": c["token_len"],
                          "horizons": list(c["horizons"]),
                          "windows": c["windows"], "csv_sha256": c["csv_sha256"],
                          **({"order_assert": c["order_assert"]}
                             if "order_assert" in c else {})}
                     for ds, c in DATASETS.items()},
        "domain_order": list(DOMAIN_ORDER),
        "total_windows": TOTAL_WINDOWS,
        "exp017_archive_commit": EXP017_ARCHIVE_COMMIT,
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
