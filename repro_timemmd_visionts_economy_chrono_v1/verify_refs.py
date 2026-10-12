#!/usr/bin/env python3
"""Stage 2 (EXP-022): provenance gates BEFORE anything touches the GPU.

- EXP-020 (BASE_DATA) and EXP-021 (BASE_METHOD) protocol.json: file
  sha256 == pinned full value, token recomputed from bytes == pinned token
- dataset/Economy.csv (ascending copy): sha256 == EXP-020 pin;
  sort_mapping evidence sha == pin; original EXP-019 CSV sha == pin
- vendored visionts/ 5 files: md5 == EXP-018 frozen pins
- checkpoint at the exact path passed to the constructor: isfile +
  sha256 + byte count (missing file = HARD STOP: constructor would
  auto-download)
- OLD prediction files used later as identity/comparison anchors
  (EXP-020 Economy x4 and EXP-021 Economy x4): each npz's actual sha256
  must equal the npz_sha256 recorded in its frozen done.json — the
  anchors are byte-identical to what the frozen experiments recorded
- environment versions recorded
"""
import os
import sys

import common as C
import protocol as P

AURORA_ROOT = "/home/wlt/MMTS/Aurora_Experiments"


def check_base(tag, base):
    path = os.path.join(AURORA_ROOT, base["dir"], "protocol.json")
    if not os.path.isfile(path):
        sys.exit(f"[verify] HARD STOP: {tag} protocol missing {path}")
    sha = C.sha256_file(path)
    if sha != base["protocol_sha256"]:
        sys.exit(f"[verify] HARD STOP: {tag} protocol sha256 {sha} != pin")
    import hashlib
    tok = hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]
    if tok != base["protocol_token"]:
        sys.exit(f"[verify] HARD STOP: {tag} token {tok} != "
                 f"{base['protocol_token']}")
    return {"sha256": sha, "token": tok, "ok": True}


def check_old_predictions(tag, base_dir, npz_map):
    """npz_map: {h: npz_sha256_from_this_protocol_pin} — actual file sha
    must equal BOTH the frozen done.json record and the pin."""
    out = {}
    for h, pin in sorted(npz_map.items()):
        npz_path = os.path.join(base_dir, "predictions",
                                f"Economy__h{h}.npz")
        done_path = os.path.join(base_dir, "predictions",
                                 f"Economy__h{h}.done.json")
        for p in (npz_path, done_path):
            if not os.path.isfile(p):
                sys.exit(f"[verify] HARD STOP: {tag} anchor missing: {p}")
        actual = C.sha256_file(npz_path)
        if actual != pin:
            sys.exit(f"[verify] HARD STOP: {tag} h{h} npz sha256 {actual} "
                     f"!= pin {pin}")
        done = C.load_json(done_path)
        if not done.get("ok"):
            sys.exit(f"[verify] HARD STOP: {tag} h{h} done.json not ok")
        rec = done.get("npz_sha256")
        if rec != actual:
            sys.exit(f"[verify] HARD STOP: {tag} h{h} npz sha256 {actual} "
                     f"!= frozen done.json record {rec}")
        out[f"h{h}"] = {"npz_sha256": actual,
                        "done_npz_sha256_record": rec,
                        "matches_frozen_record": True}
    return out


def main():
    rep = {"exp_id": P.EXP_ID}

    rep["base_data_protocol"] = check_base("EXP-020", P.BASE_DATA)
    rep["base_method_protocol"] = check_base("EXP-021", P.BASE_METHOD)

    p = os.path.join(C.HERE, "dataset", "Economy.csv")
    if not os.path.isfile(p):
        sys.exit("[verify] HARD STOP: dataset/Economy.csv missing — run "
                 "prepare_data first")
    got = C.sha256_file(p)
    if got != P.DATASETS["Economy"]["csv_sha256"]:
        sys.exit(f"[verify] HARD STOP: Economy.csv sha256 {got} != pin")
    rep["ascending_csv_sha256"] = got

    sm = os.path.join(C.RESULTS_DIR, "base_evidence", "sort_mapping.csv")
    if C.sha256_file(sm) != P.DATASETS["Economy"]["sort_mapping_sha256"]:
        sys.exit("[verify] HARD STOP: base_evidence/sort_mapping.csv "
                 "sha256 != pin")
    rep["sort_mapping_sha256"] = P.DATASETS["Economy"]["sort_mapping_sha256"]

    orig = os.path.join(AURORA_ROOT, "repro_timemmd_zeroshot_v1",
                        "dataset", "Economy.csv")
    got = C.sha256_file(orig)
    if got != P.DATASETS["Economy"]["original_csv_sha256"]:
        sys.exit(f"[verify] HARD STOP: original Economy.csv sha256 {got} "
                 f"!= pin")
    rep["original_csv_sha256"] = got

    md5s = {}
    for rel, want in P.VISIONTS_CODE_MD5.items():
        fp = os.path.join(C.HERE, rel)
        if not os.path.isfile(fp):
            sys.exit(f"[verify] HARD STOP: vendored {rel} missing")
        g = C.md5_file(fp)
        if g != want:
            sys.exit(f"[verify] HARD STOP: {rel} md5 {g} != EXP-018 pin "
                     f"{want}")
        md5s[rel] = g
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

    spec = P.DATASETS["Economy"]
    for tag, base in (("EXP-020", P.BASE_DATA), ("EXP-021", P.BASE_METHOD)):
        bdir = os.path.join(AURORA_ROOT, base["dir"])
        rep[f"{'exp020' if tag == 'EXP-020' else 'exp021'}"
            f"_anchor_predictions"] = check_old_predictions(
            tag, bdir, {str(h): _npz_pin(bdir, h)
                        for h in spec["horizons"]})

    rep["runtime"] = C.env_info()
    C.dump_json(rep, os.path.join(C.RESULTS_DIR, "verify_refs.json"))
    print(f"[verify] ALL OK: EXP-020 token "
          f"{rep['base_data_protocol']['token']}, EXP-021 token "
          f"{rep['base_method_protocol']['token']}, CSV+mapping+original, "
          f"5 md5, weights {wsha[:12]}…, 4+4 anchor npz "
          f"(sha == frozen done.json records)")


def _npz_pin(base_dir, h):
    """Pin = the npz_sha256 recorded in the frozen done.json; verify_refs
    cross-checks file bytes against this record (the frozen record IS the
    pin for the anchor files)."""
    done = C.load_json(os.path.join(base_dir, "predictions",
                                    f"Economy__h{h}.done.json"))
    return done["npz_sha256"]


if __name__ == "__main__":
    main()
