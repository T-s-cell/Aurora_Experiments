#!/usr/bin/env python3
"""Pre-GPU reference gate for repro_timemmd_economy_chrono_v1.

Checks, CPU only, BEFORE any GPU work (run on dt):
  1. results/data_prep.json exists; the sorted Economy CSV sha256
     matches the data_prep pin and the original CSV sha recorded there
     equals the frozen EXP-019 pin;
  2. vendored official/ md5s match the protocol pin (incl. the single
     recorded prediction-saving patch);
  3. HF snapshot exists with pinned revision dir; model.safetensors
     sha256 == pin; config.json + generation_config.json sha256 == pin
     (gate SKIPPED when AURORA_HF_SNAPSHOT is unset, e.g. local box);
  4. vendored BERT tokenizer loads offline (local_files_only).

Exit 0 only if all gates PASS.
"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C
import protocol as P


def main():
    rep = {"gates": {}}
    ok = True

    # 1. CSVs — the sorted Economy CSV is pinned via data_prep.json
    #    (prepare_data.py must have run first); the original CSV sha
    #    recorded there must equal the frozen EXP-019 pin.
    csv_bad = []
    try:
        dp = C.load_json(os.path.join(C.RESULTS_DIR, "data_prep.json"))
    except FileNotFoundError:
        dp = None
        csv_bad.append("results/data_prep.json missing — run prepare_data.py")
    if dp is not None:
        if not dp.get("ok"):
            csv_bad.append("data_prep.json not ok")
        elif dp["original_csv_sha256"] != P.ORIGINAL_CSV_SHA256:
            csv_bad.append("data_prep original csv sha != EXP-019 pin")
        for ds in P.DATASETS:
            p = P.csv_path(ds)
            if not os.path.isfile(p):
                csv_bad.append(f"{ds}: missing {p}")
                continue
            got = C.sha256_file(p)
            if got != dp["sorted_csv_sha256"]:
                csv_bad.append(f"{ds}: sha256 {got[:12]} != data_prep pin")
    extra = sorted(set(os.path.basename(x) for x in glob.glob(
        os.path.join(C.DATASET_DIR, "*.csv")))
        - {f"{d}.csv" for d in P.DATASETS})
    if extra:
        csv_bad.append(f"unexpected CSVs: {extra}")
    rep["gates"]["csv_sha256"] = {"ok": not csv_bad, "bad": csv_bad}
    ok &= not csv_bad
    print(f"[verify-refs] csv gate: {'PASS' if not csv_bad else csv_bad}")

    # 2. official source md5 (raises SystemExit on mismatch)
    try:
        P.check_official()
        rep["gates"]["official_md5"] = {"ok": True}
    except SystemExit:
        rep["gates"]["official_md5"] = {"ok": False}
        ok = False
        print("[verify-refs] official md5 gate: FAIL")

    # 3. weights — SKIPPED when AURORA_HF_SNAPSHOT is unset (local box has
    # no HF cache); the dt wrapper always sets it, so there it must PASS.
    snap = os.environ.get("AURORA_HF_SNAPSHOT", "")
    w_bad = []
    if not snap:
        w_status = "SKIPPED"
    else:
        w_status = "PASS"
        if not os.path.isdir(snap):
            w_bad.append(f"AURORA_HF_SNAPSHOT not a dir: {snap!r}")
        else:
            if os.path.basename(os.path.normpath(snap)) != P.HF_REVISION:
                w_bad.append(f"snapshot dir is not revision {P.HF_REVISION}")
            st = os.path.join(snap, "model.safetensors")
            if not os.path.isfile(st):
                w_bad.append("model.safetensors missing")
            else:
                sz = os.path.getsize(st)
                if sz != P.WEIGHTS_BYTES:
                    w_bad.append(f"size {sz} != {P.WEIGHTS_BYTES}")
                else:
                    got = C.sha256_file(st)
                    if got != P.WEIGHTS_SHA256:
                        w_bad.append(f"sha256 {got[:12]} != pinned")
            for fname, pin in (("config.json", P.HF_CONFIG_SHA256),
                               ("generation_config.json",
                                P.HF_GENERATION_CONFIG_SHA256)):
                fp = os.path.join(snap, fname)
                if not os.path.isfile(fp):
                    w_bad.append(f"{fname} missing")
                else:
                    got = C.sha256_file(fp)
                    if got != pin:
                        w_bad.append(f"{fname} sha256 {got[:12]} != pinned")
    rep["gates"]["weights"] = {"ok": not w_bad, "status": w_status,
                               "bad": w_bad, "snapshot": snap}
    ok &= not w_bad
    print(f"[verify-refs] weights gate: {w_status if not w_bad else w_bad}")

    # 4. tokenizer offline smoke (no torch needed)
    t_bad = []
    try:
        from transformers import BertTokenizer
        tok = BertTokenizer.from_pretrained(
            os.path.join(C.OFFICIAL_TMM, "aurora", "bert_config"),
            local_files_only=True)
        enc = tok("probe", padding="max_length", truncation=True,
                  max_length=8, return_tensors="pt")
        if tuple(enc["input_ids"].shape) != (1, 8):
            t_bad.append(f"tokenizer shape {tuple(enc['input_ids'].shape)}")
    except Exception as e:
        t_bad.append(repr(e))
    rep["gates"]["tokenizer"] = {"ok": not t_bad, "bad": t_bad}
    ok &= not t_bad
    print(f"[verify-refs] tokenizer gate: {'PASS' if not t_bad else t_bad}")

    rep["env"] = C.env_info()
    rep["ok"] = bool(ok)
    C.dump_json(rep, os.path.join(C.RESULTS_DIR, "verify_refs.json"))
    print(f"[verify-refs] RESULT: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
