#!/usr/bin/env python3
"""M48T512_D2 preflight (sections PF-DATA, PF-A..D) — train/val windows ONLY;
zero test-window inference.

Text comes exclusively from cache/preflight_text_trainval.npz (D0+D2 pairs,
8 train/val windows, keys disjoint from test/excluded — asserted here and at
build time). PF-A verifies the comparison path is unchanged (new driver + D0
tensors bitwise-equals the EXP-011 entry point, same window/seed/input);
PF-B hard-asserts the D2 tensors handed to the model are exactly the cache
rows and RECORDS (never asserts) D0-vs-D2 prediction differences; PF-C checks
shapes/finiteness/repeat/cross-process/model-state; PF-D checks the gate
matrix, cache-sha refusal, and fingerprint-mismatch quarantine. PF-DATA
re-verifies the D2 test cache against the frozen diagnostic trace.

Set CUBLAS_WORKSPACE_CONFIG before any CUDA init; determinism pack comes from
load_aurora before first inference. Evidence: repro_mm_timesx_d2/preflight/*.json
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SUBDIR = Path(__file__).resolve().parent
PROJECT = SUBDIR.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(SUBDIR))
sys.path.insert(0, str(PROJECT / "repro_mm_timesx_v1"))

PREFLIGHT = SUBDIR / "preflight"
PREFLIGHT.mkdir(exist_ok=True)
RESULTS = {"sections": {}}
PROTO = None
PF_SEED = 424242  # preflight-only seed (NOT an official run seed)
PF_NPZ = SUBDIR / "cache" / "preflight_text_trainval.npz"


def save(section, report):
    RESULTS["sections"][section] = report
    name = f"{section.lower().replace('-', '_')}.json"
    (PREFLIGHT / name).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    ok = report.get("ok", False)
    print(f"[preflight] {section}: {'OK' if ok else 'FAILED'} -> {name}")


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_pf():
    """Preflight text npz + train/val windows from the frozen data."""
    import data_loader
    z = np.load(PF_NPZ, allow_pickle=False)
    keys = list(zip([str(x) for x in z["var_keys"]],
                    [str(x) for x in z["sample_ids"]]))
    assert len(keys) == 8 and len(set(keys)) == 8
    test_keys = {(vk, sid) for vk, sid, *_ in
                 data_loader.TimesXData(PROJECT / "data").test_rows()}
    assert not (set(keys) & test_keys), "preflight key overlaps the test split"
    td = data_loader.TimesXData(PROJECT / "data")
    rows = []
    for vk, sid in keys:
        past, tgt = td.split_window_by_id(vk, sid)
        rows.append((vk, sid, vk.split("__", 1)[0], past, tgt, td.d_of(vk, sid)))
    return z, keys, rows


# ------------------------------------------------------------------ PF-DATA

def pf_data():
    import data_loader
    protocol = json.loads((SUBDIR / "protocol_d2.json").read_text())
    cache_npz = SUBDIR / "cache" / "text_tokens_M48T512_D2.npz"
    cache_meta = SUBDIR / "cache" / "text_meta_M48T512_D2.jsonl"

    got = sha256_file(cache_npz)
    sha_ok = got == protocol["text"]["cache"]["npz_sha256"]
    meta_sha_ok = sha256_file(cache_meta) == protocol["text"]["cache"]["meta_jsonl_sha256"]

    z = np.load(cache_npz, allow_pickle=False)
    keys = list(zip([str(x) for x in z["var_keys"]],
                    [str(x) for x in z["sample_ids"]]))
    n_unique = len(set(keys))
    schema_ok = (z["ids"].shape == (2474, 512) and z["mask"].shape == (2474, 512)
                 and z["ids"].dtype == np.int32 and z["mask"].dtype == np.int32
                 and n_unique == 2474)

    rows_test = data_loader.TimesXData(PROJECT / "data").test_rows()
    test_keys = {(vk, sid) for vk, sid, *_ in rows_test}
    keys_ok = set(keys) == test_keys

    z11 = np.load(PROJECT / "repro_mm_timesx_v1" / "cache" /
                  "text_tokens_M48T512.npz", allow_pickle=False)
    keys11 = set(zip([str(x) for x in z11["var_keys"]],
                     [str(x) for x in z11["sample_ids"]]))
    keys_eq_exp011 = keys11 == set(keys)

    # bitwise vs frozen diagnostic trace (source of the extraction)
    trace_path = PROJECT / "analysis" / "text_budget_v1" / "outputs" / "trace_D2.npz"
    trace_bitwise = None
    n_mismatch = None
    if trace_path.exists():
        zt = np.load(trace_path, allow_pickle=False)
        idx = {k: i for i, k in enumerate(zip(
            [str(x) for x in zt["var_keys"]], [str(x) for x in zt["sample_ids"]]))}
        sel = np.array([idx[k] for k in keys])
        eq = bool(np.array_equal(zt["ids"][sel], z["ids"])
                  and np.array_equal(zt["mask"][sel], z["mask"]))
        n_mismatch = 0 if eq else 2474
        trace_bitwise = eq

    # meta sanity: line i describes npz row i
    meta_ok, meta_n = True, 0
    with open(cache_meta, encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            meta_n += 1
            if (r["var_key"], r["sample_id"]) != keys[i] or r["scope"] != "test" \
                    or r["content_tokens"] > 510 or "decoded_input" not in r:
                meta_ok = False
                break

    # preflight npz hygiene
    pfz, pf_keys, _ = load_pf()
    pf_scopes = sorted({str(x) for x in pfz["scope"]})
    pf_ok = (pfz["ids_d0"].shape == (8, 512) and pfz["ids_d2"].shape == (8, 512)
             and set(pf_scopes) <= {"train", "val"}
             and pfz["ids_d0"].dtype == np.int32)

    ok = all([sha_ok, meta_sha_ok, schema_ok, keys_ok, keys_eq_exp011,
              bool(trace_bitwise), meta_ok, meta_n == 2474, pf_ok])
    save("PF-DATA", {
        "ok": ok, "cache_npz_sha256_ok": sha_ok, "meta_sha256_ok": meta_sha_ok,
        "schema_ok": schema_ok, "n_rows": len(keys), "n_unique_keys": n_unique,
        "keys_equal_test_rows": keys_ok, "keys_equal_exp011_cache": keys_eq_exp011,
        "bitwise_vs_trace_d2": trace_bitwise, "n_mismatch_vs_trace": n_mismatch,
        "meta_lines": meta_n, "meta_rowwise_ok": meta_ok,
        "preflight_npz_ok": pf_ok, "preflight_scopes": pf_scopes,
    })
    return ok


# --------------------------------------------------------------------- PF-A

def pf_a(model):
    """Comparison path unchanged: new driver + D0 tensors bitwise-equals the
    EXP-011 entry point (same window/seed/input); text=None path bitwise-equals
    the root predict_window."""
    import torch
    from predict import predict_window
    from predict_mm import predict_window_mm
    from predict_d2 import predict_window_mm as predict_window_d2
    itl, ns = PROTO["inference"]["itl_main"], PROTO["inference"]["num_samples"]

    pfz, keys, rows = load_pf()
    device = next(model.parameters()).device
    checks, all_ok = [], True
    for i, (vk, sid, dom, past, tgt, dv) in enumerate(rows):
        ids = torch.from_numpy(pfz["ids_d0"][i].astype(np.int64)).unsqueeze(0).to(device)
        mask = torch.from_numpy(pfz["mask_d0"][i].astype(np.int64)).unsqueeze(0).to(device)
        tt = torch.zeros_like(ids)
        p_d2drv = predict_window_d2(model, past, itl, ns, PF_SEED,
                                    text_ids=ids, text_mask=mask, text_typeids=tt)
        p_mm = predict_window_mm(model, past, itl, ns, PF_SEED,
                                 text_ids=ids, text_mask=mask, text_typeids=tt)
        p_root = predict_window(model, past, itl, ns, PF_SEED)
        p_d2drv_none = predict_window_d2(model, past, itl, ns, PF_SEED)
        b1 = bool(np.array_equal(p_d2drv, p_mm))
        b2 = bool(np.array_equal(p_d2drv_none, p_root))
        checks.append({"var_key": vk, "sample_id": sid,
                       "d0_bitwise_vs_exp011_entry": b1,
                       "text_none_bitwise_vs_root": b2,
                       "max_abs_diff": float(np.max(np.abs(p_d2drv - p_mm)))})
        all_ok &= b1 and b2
    save("PF-A", {"ok": all_ok, "n_windows": len(checks), "seed": PF_SEED,
                  "checks": checks})
    return all_ok


# --------------------------------------------------------------------- PF-B

def pf_b(model):
    """D2 tensors reach the model EXACTLY as cached (hard assert); D0-vs-D2
    prediction differences are RECORDED, never asserted."""
    import torch
    from predict_d2 import predict_window_mm as predict_window_d2
    itl, ns = PROTO["inference"]["itl_main"], PROTO["inference"]["num_samples"]

    pfz, keys, rows = load_pf()
    device = next(model.parameters()).device
    orig_generate = model.generate
    captured = {}

    def spy_generate(*args, **kwargs):
        captured["kwargs"] = {k: (v if not torch.is_tensor(v) else v.clone())
                              for k, v in kwargs.items()}
        return orig_generate(*args, **kwargs)

    tensors_ok, checks, all_ok = True, [], True
    for i, (vk, sid, dom, past, tgt, dv) in enumerate(rows):
        ids0 = torch.from_numpy(pfz["ids_d0"][i].astype(np.int64)).unsqueeze(0).to(device)
        mask0 = torch.from_numpy(pfz["mask_d0"][i].astype(np.int64)).unsqueeze(0).to(device)
        ids2 = torch.from_numpy(pfz["ids_d2"][i].astype(np.int64)).unsqueeze(0).to(device)
        mask2 = torch.from_numpy(pfz["mask_d2"][i].astype(np.int64)).unsqueeze(0).to(device)
        tt = torch.zeros_like(ids2)

        model.generate = spy_generate
        try:
            p2 = predict_window_d2(model, past, itl, ns, PF_SEED,
                                   text_ids=ids2, text_mask=mask2, text_typeids=tt)
        finally:
            model.generate = orig_generate
        kw = captured["kwargs"]
        t_ids_ok = (torch.equal(kw["text_input_ids"], ids2)
                    and kw["text_input_ids"].dtype == torch.int64
                    and tuple(kw["text_input_ids"].shape) == (1, 512)
                    and kw["text_input_ids"].device == ids2.device)
        t_mask_ok = torch.equal(kw["text_attention_mask"], mask2)
        t_tt_ok = bool((kw["text_token_type_ids"] == 0).all())
        str_path_none = kw.get("text_inputs") is None
        tensors_ok &= t_ids_ok and t_mask_ok and t_tt_ok and str_path_none

        p1 = predict_window_d2(model, past, itl, ns, PF_SEED,
                               text_ids=ids0, text_mask=mask0,
                               text_typeids=torch.zeros_like(ids0))
        diff = np.abs(p2 - p1)
        changed = bool(np.any(p2 != p1))
        checks.append({"var_key": vk, "sample_id": sid,
                       "text_tensors_exact": bool(t_ids_ok and t_mask_ok and t_tt_ok
                                                  and str_path_none),
                       "inputs_differ": bool(not np.array_equal(pfz["ids_d0"][i],
                                                                pfz["ids_d2"][i])),
                       "pred_changed_vs_d0": changed,
                       "max_abs_diff_vs_d0": float(diff.max()),
                       "mean_abs_diff_vs_d0": float(diff.mean())})
        all_ok &= t_ids_ok and t_mask_ok and t_tt_ok and str_path_none
    save("PF-B", {"ok": all_ok, "tensors_exact_all_windows": tensors_ok,
                  "note": "pred_changed_vs_d0 is RECORDED ONLY: different text "
                          "inputs do not guarantee different predictions; equal "
                          "outputs are NOT an implementation failure.",
                  "n_windows": len(checks),
                  "n_changed_vs_d0": sum(c["pred_changed_vs_d0"] for c in checks),
                  "checks": checks})
    return all_ok


# --------------------------------------------------------------------- PF-C

def _model_state_digest(model):
    import torch
    tensors, bn = {}, {}
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                            torch.nn.BatchNorm3d)):
            bn[name] = [float(mod.running_mean.sum()), float(mod.running_var.sum()),
                        int(mod.num_batches_tracked)]
    for name, p in model.state_dict().items():
        tensors[name] = hashlib.sha256(
            p.detach().cpu().numpy().tobytes()).hexdigest()
    return {"n_tensors": len(tensors), "tensor_sha": hashlib.sha256(
                "".join(tensors[k] for k in sorted(tensors)).encode()).hexdigest(),
            "n_bn_modules": len(bn), "bn": bn}


def pf_c(model):
    import torch
    from predict_d2 import predict_window_mm as predict_window_d2
    itl, ns = PROTO["inference"]["itl_main"], PROTO["inference"]["num_samples"]

    pfz, keys, rows = load_pf()
    vk, sid, dom, past, tgt, dv = rows[0]
    i = 0
    device = next(model.parameters()).device
    ids2 = torch.from_numpy(pfz["ids_d2"][i].astype(np.int64)).unsqueeze(0).to(device)
    mask2 = torch.from_numpy(pfz["mask_d2"][i].astype(np.int64)).unsqueeze(0).to(device)
    tt = torch.zeros_like(ids2)

    before = _model_state_digest(model)
    torch.cuda.reset_peak_memory_stats()
    x = torch.from_numpy(np.asarray(past, dtype=np.float32)).unsqueeze(0).to(device)
    torch.manual_seed(PF_SEED)
    with torch.inference_mode():
        raw = model.generate(inputs=x, text_inputs=None, text_input_ids=ids2,
                             text_attention_mask=mask2, text_token_type_ids=tt,
                             vision_inputs=None, revin=True, num_samples=ns,
                             max_output_length=12, inference_token_len=itl)
    peak_gb = torch.cuda.max_memory_allocated() / 2 ** 30
    shape_ok = list(raw.shape) == [1, ns, 12]
    finite_ok = bool(torch.isfinite(raw).all())
    p1 = raw.to(torch.float64).mean(dim=1).squeeze(0).cpu().numpy()
    t0 = time.time()
    p2 = predict_window_d2(model, past, itl, ns, PF_SEED, ids2, mask2, tt)
    ms = (time.time() - t0) * 1000
    repeat_bitwise = bool(np.array_equal(p1, p2))
    after = _model_state_digest(model)
    state_ok = before == after

    child = subprocess.run(
        [sys.executable, str(SUBDIR / "run_preflight_d2.py"), "--pf-c-child",
         "--vk", vk, "--sid", sid, "--seed", str(PF_SEED)],
        capture_output=True, text=True)
    cross_ok = False
    cross_detail = {"rc": child.returncode}
    if child.returncode == 0:
        child_pred = np.load(PREFLIGHT / "pf_c_child_pred.npy")
        cross_ok = bool(np.array_equal(child_pred, p1))
        cross_detail["bitwise"] = cross_ok

    ok = shape_ok and finite_ok and repeat_bitwise and state_ok and cross_ok
    save("PF-C", {
        "ok": ok, "window": {"var_key": vk, "sample_id": sid},
        "raw_shape": list(raw.shape), "raw_finite": finite_ok,
        "same_seed_repeat_bitwise": repeat_bitwise,
        "weights_unchanged": state_ok,
        "n_tensors": after["n_tensors"], "n_bn_modules": after["n_bn_modules"],
        "cross_process_bitwise": cross_ok, "cross_process_detail": cross_detail,
        "ms_per_window_with_text": ms, "peak_vram_gb": peak_gb,
    })
    return ok


def pf_c_child(vk, sid, seed):
    """Hidden child mode: recompute ONE train/val window with its D2 tensors
    from the preflight npz, dump npy."""
    import torch
    from load_aurora import load_aurora
    from predict_d2 import predict_window_mm as predict_window_d2
    model, _ = load_aurora(device="cuda")
    pfz, keys, _rows = load_pf()
    i = keys.index((vk, sid))
    device = next(model.parameters()).device
    past, _tgt = __import__("data_loader").TimesXData(
        PROJECT / "data").split_window_by_id(vk, sid)
    pred = predict_window_d2(
        model, past, PROTO["inference"]["itl_main"],
        PROTO["inference"]["num_samples"], seed,
        torch.from_numpy(pfz["ids_d2"][i].astype(np.int64)).unsqueeze(0).to(device),
        torch.from_numpy(pfz["mask_d2"][i].astype(np.int64)).unsqueeze(0).to(device),
        torch.zeros(1, 512, dtype=torch.long, device=device))
    np.save(PREFLIGHT / "pf_c_child_pred.npy", pred)
    print(f"[pf-c-child] wrote pred for {vk} {sid}")
    return 0


# --------------------------------------------------------------------- PF-D

def pf_d():
    """Gate matrix + cache-sha refusal + fingerprint-mismatch quarantine."""
    from predict import ShardStore
    from predict_d2 import TextTokenStore

    driver = SUBDIR / "run_eval_d2.py"
    py = sys.executable

    def run(args):
        return subprocess.run([py, str(driver)] + args, capture_output=True,
                              text=True, cwd=str(SUBDIR))

    # dry-run: no inference, rc 0, plan JSON
    r = run(["--dry-run"])
    dry_ok = r.returncode == 0 and '"mode": "dry-run (no inference performed)"' in r.stdout

    # token matrix: all of these MUST be rejected with rc 2
    matrix = {
        "exp010_token": ["--itl", "48", "--i-approve-frozen-protocol", "580eb3c3f51d"],
        "exp011_token": ["--itl", "48", "--i-approve-frozen-protocol", "74b08ecac1ed"],
        "wrong_token": ["--itl", "48", "--i-approve-frozen-protocol", "deadbeef0000"],
        "wrong_itl": ["--itl", "9", "--dry-run"],
        "token_plus_dryrun": ["--dry-run", "--i-approve-frozen-protocol",
                              sha256_file(SUBDIR / "protocol_d2.json")[:12]],
        "no_token": ["--itl", "48"],
    }
    rejects = {k: run(v).returncode for k, v in matrix.items()}
    # bad tokens/flags must be REJECTED (rc 2); a missing token must simply not
    # unlock the run (rc 1 = locked, guidance printed, no inference)
    rejects_ok = all(rc == 2 for k, rc in rejects.items() if k != "no_token") \
        and rejects["no_token"] == 1

    # tampered protocol cache sha -> TextTokenStore must raise
    protocol = json.loads((SUBDIR / "protocol_d2.json").read_text())
    bad = json.loads(json.dumps(protocol))
    bad["text"]["cache"]["npz_sha256"] = "0" * 64
    raised = False
    try:
        TextTokenStore(bad)
    except Exception as e:
        raised = "sha256 mismatch" in str(e)

    # fingerprint-mismatch shard -> NOT reused, moved to quarantine
    scratch = PREFLIGHT / "scratch_shards"
    if scratch.exists():
        shutil.rmtree(scratch)
    store = ShardStore(out_dir=scratch)
    rows = [{"var_key": "d__v", "sample_id": "d__v__s", "domain": "d",
             "pred": np.zeros(12), "target": np.zeros(12), "d": 1.0}]
    fp_bad = {"protocol_d2_sha256": "f" * 64}
    store.save("M48T512_D2", "d", 2021, fp_bad, rows)
    reused = store.reuse_or_init("M48T512_D2", "d", 2021,
                                 {"protocol_d2_sha256": "e" * 64})
    quarantined = (not reused
                   and not (scratch / "M48T512_D2__d__s2021__test.npz").exists()
                   and any(p.name.endswith("M48T512_D2__d__s2021__test.npz")
                           for p in (scratch / "quarantine").glob("*")))

    ok = dry_ok and rejects_ok and raised and quarantined
    save("PF-D", {"ok": ok, "dry_run_ok": dry_ok,
                  "reject_returncodes": rejects, "rejects_all_rc2": rejects_ok,
                  "tampered_cache_sha_refused": raised,
                  "fingerprint_mismatch_quarantined": quarantined})
    return ok


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gpu", action="store_true",
                    help="run only CPU sections (PF-DATA, PF-D)")
    ap.add_argument("--pf-c-child", default=None, nargs=2, metavar=("VK", "SID"),
                    help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    global PROTO
    PROTO = json.loads((SUBDIR / "protocol_d2.json").read_text())

    if args.pf_c_child:
        return pf_c_child(args.pf_c_child[0], args.pf_c_child[1], args.seed)

    ok = pf_data() and pf_d()
    if not args.skip_gpu:
        from load_aurora import load_aurora
        model, _ = load_aurora(device="cuda")
        ok = pf_a(model) and ok
        ok = pf_b(model) and ok
        ok = pf_c(model) and ok
    print("[preflight] RESULT:", "ALL OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
