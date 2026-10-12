#!/usr/bin/env python3
"""Stage 2 (EXP-021): provenance gates BEFORE anything touches the GPU.

- base EXP-019 protocol.json: file sha256 == pinned full value, token
  recompute from bytes == pinned token
- 9 CSVs in dataset/: sha256 == protocol pins
- vendored visionts/ 5 files: md5 == EXP-018 frozen pins
- checkpoint at the exact path passed to the constructor: isfile +
  sha256 + byte count (missing file = HARD STOP: constructor would
  auto-download)
- EXP-019 base predictions: all 36 npz present (read-only identity
  anchors), sha256s recorded
- environment versions recorded
"""
import os
import sys

import common as C
import protocol as P


def main():
    rep = {"exp_id": P.EXP_ID}

    base_path = os.path.join("/home/wlt/MMTS/Aurora_Experiments",
                             P.BASE["dir"], "protocol.json")
    if not os.path.isfile(base_path):
        sys.exit(f"[verify] HARD STOP: base protocol missing {base_path}")
    sha = C.sha256_file(base_path)
    if sha != P.BASE["protocol_sha256"]:
        sys.exit(f"[verify] HARD STOP: base protocol sha256 {sha} != pin")
    import hashlib
    tok = hashlib.sha256(open(base_path, "rb").read()).hexdigest()[:12]
    if tok != P.BASE["protocol_token"]:
        sys.exit(f"[verify] HARD STOP: base token {tok} != "
                 f"{P.BASE['protocol_token']}")
    rep["base_protocol"] = {"sha256": sha, "token": tok, "ok": True}

    csvs = {}
    for ds in P.DOMAIN_ORDER:
        p = os.path.join(C.HERE, "dataset", f"{ds}.csv")
        if not os.path.isfile(p):
            sys.exit(f"[verify] HARD STOP: dataset/{ds}.csv missing — run "
                     f"prepare_data first")
        got = C.sha256_file(p)
        if got != P.DATASETS[ds]["csv_sha256"]:
            sys.exit(f"[verify] HARD STOP: {ds}.csv sha256 {got} != pin")
        csvs[ds] = got
    rep["csv_sha256"] = csvs

    md5s = {}
    for rel, want in P.VISIONTS_CODE_MD5.items():
        p = os.path.join(C.HERE, rel)
        if not os.path.isfile(p):
            sys.exit(f"[verify] HARD STOP: vendored {rel} missing")
        got = C.md5_file(p)
        if got != want:
            sys.exit(f"[verify] HARD STOP: {rel} md5 {got} != EXP-018 pin "
                     f"{want}")
        md5s[rel] = got
    rep["visionts_code_md5"] = md5s
    rep["visionts_repo_commit"] = P.VISIONTS_REPO_COMMIT

    ck = C.assert_ckpt_present()
    wsha = C.sha256_file(ck)
    nbytes = os.path.getsize(ck)
    if wsha != C.WEIGHTS_SHA256 or nbytes != C.WEIGHTS_BYTES:
        sys.exit(f"[verify] HARD STOP: weights mismatch at {ck}: "
                 f"sha256={wsha} bytes={nbytes}")
    rep["weights"] = {"path": ck, "sha256": wsha, "bytes": nbytes,
                      "ok": True}

    base_pred = {}
    for ds in P.DOMAIN_ORDER:
        for h in P.DATASETS[ds]["horizons"]:
            p = os.path.join("/home/wlt/MMTS/Aurora_Experiments",
                             P.BASE["dir"], "predictions",
                             f"{ds}__h{h}.npz")
            if not os.path.isfile(p):
                sys.exit(f"[verify] HARD STOP: base npz missing: {p}")
            base_pred[f"{ds}__h{h}"] = C.sha256_file(p)
    rep["base_npz_sha256"] = base_pred
    rep["base_npz_count"] = len(base_pred)

    rep["runtime"] = C.env_info()
    C.dump_json(rep, os.path.join(C.RESULTS_DIR, "verify_refs.json"))
    print(f"[verify] ALL OK: base token {tok}, 9 CSV, 5 md5, weights "
          f"{wsha[:12]}…, {len(base_pred)} base npz")


if __name__ == "__main__":
    main()
