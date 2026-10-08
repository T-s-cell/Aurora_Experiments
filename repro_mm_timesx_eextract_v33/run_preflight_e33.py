#!/usr/bin/env python3
"""M48T512_E33 preflight — CPU sections + train/val-only GPU sections; zero
test-window inference.

Sections:
  PF-DATA'  Independent full re-verification of the frozen E33 test cache:
            rebuild D2 per window (bitwise vs trace_D2.npz AND vs the D2 test
            cache), recompute per-event budgets vs budgets_test.jsonl,
            re-assemble every compressed window from its persisted record and
            compare bitwise to the npz row; fallback/kept rows must equal the
            D2 cache rows bitwise; key sets equal across
            npz/test_rows/EXP-011/D2 caches; record file line i <-> npz row i.
  PF-D      Gate matrix: dry-run OK; EXP-010/011/012 tokens, wrong token,
            wrong itl, token+dry-run all REJECTED; no token stays locked;
            tampered cache sha refused; fingerprint-mismatch shard quarantined.
  PF-A      Comparison path unchanged: D0 tensors via this driver
            bitwise-equal the EXP-011 entry point; additionally D2 tensors
            via this driver bitwise-equal the EXP-012 (predict_d2) entry point.
            Both checks use the train/val-only preflight npz; no test windows.
  PF-E      REAL E-Extract inputs on train/val: compressed windows (inputs
            actually changed vs D2) and fallback windows (inputs bitwise ==
            D2) from the frozen v3.3 diagnostic val outputs. Hard-asserts the
            final ids/mask reach the model exactly (spy on generate), output
            shape/finiteness, same-seed repeatability; fallback windows must
            produce bitwise-identical predictions from the E33 and D2 rows;
            prediction CHANGES on compressed windows are recorded, not
            asserted.
  PF-C      Weights unchanged across inference, same-seed repeat and
            cross-process bitwise repeatability on a REAL compressed
            E-Extract input.

Set CUBLAS_WORKSPACE_CONFIG before any CUDA init. Evidence:
repro_mm_timesx_eextract_v33/preflight/*.json
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
for _p in (str(PROJECT), str(PROJECT / "repro_mm_timesx_v1"),
           str(PROJECT / "repro_mm_timesx_d2"),
           str(PROJECT / "analysis" / "event_compression_v1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PREFLIGHT = SUBDIR / "preflight"
PREFLIGHT.mkdir(exist_ok=True)
RESULTS = {"sections": {}}
PROTO = None
PF_SEED = 424242  # preflight-only seed (NOT an official run seed)

D2_CACHE_NPZ = PROJECT / "repro_mm_timesx_d2" / "cache" / "text_tokens_M48T512_D2.npz"
EXP011_CACHE_NPZ = PROJECT / "repro_mm_timesx_v1" / "cache" / "text_tokens_M48T512.npz"
D2_PF_NPZ = PROJECT / "repro_mm_timesx_d2" / "cache" / "preflight_text_trainval.npz"
DIAG_OUT = PROJECT / "analysis" / "event_compression_v1" / "outputs"
E33_NPZ = SUBDIR / "cache" / "text_tokens_M48T512_E33.npz"
E33_RECORDS = SUBDIR / "cache" / "e33_windows_test.jsonl"


def save(section, report):
    RESULTS["sections"][section] = report
    name = f"{section.lower().replace('-', '_')}.json"
    (PREFLIGHT / name).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    ok = report.get("ok", False)
    print(f"[preflight] {section}: {'OK' if ok else 'FAILED'} -> {name}")


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def row_sha(ids, mask):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(ids, dtype=np.int32).tobytes())
    h.update(np.ascontiguousarray(mask, dtype=np.int32).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------ PF-DATA'

def pf_data_prime():
    import data_loader
    from ec_common import (compute_event_budgets, event_pieces, rebuild_d2,
                           split_events_clean, trace_npz_index,
                           verify_bitwise_vs_trace)
    from allocate import assemble
    from text_cache import build_index

    protocol = json.loads((SUBDIR / "protocol_e33.json").read_text())
    from ec_common import load_tokenizer
    tok = load_tokenizer()

    sha_ok = sha256_file(E33_NPZ) == protocol["text"]["cache"]["npz_sha256"]
    z = np.load(E33_NPZ, allow_pickle=False)
    keys = list(zip([str(x) for x in z["var_keys"]],
                    [str(x) for x in z["sample_ids"]]))
    schema_ok = (z["ids"].shape == (2474, 512) and z["mask"].shape == (2474, 512)
                 and z["ids"].dtype == np.int32 and z["mask"].dtype == np.int32
                 and len(set(keys)) == 2474)

    rows_test = data_loader.TimesXData(PROJECT / "data").test_rows()
    test_keys = {(vk, sid) for vk, sid, *_ in rows_test}
    keys_ok = set(keys) == test_keys
    zd = np.load(D2_CACHE_NPZ, allow_pickle=False)
    d2_keys = set(zip([str(x) for x in zd["var_keys"]],
                      [str(x) for x in zd["sample_ids"]]))
    z11 = np.load(EXP011_CACHE_NPZ, allow_pickle=False)
    k11 = set(zip([str(x) for x in z11["var_keys"]],
                  [str(x) for x in z11["sample_ids"]]))
    keys_eq = d2_keys == k11 == set(keys)

    # records + budgets
    records = {}
    with open(E33_RECORDS, encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            records[(r["var_key"], r["sample_id"])] = (i, r)
    records_ok = len(records) == 2474
    fps = {r.get("build_fp") for _i, r in records.values()}
    svc = {r.get("service_version") for _i, r in records.values()}
    fp_constant = len(fps) == 1
    svc_constant = len(svc) == 1

    budgets = {}
    with open(SUBDIR / "outputs" / "budgets_test.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            budgets[(r["var_key"], r["sample_id"])] = r

    zt, tidx = trace_npz_index()
    index = build_index()
    outcomes = {"compressed": 0, "fallback_d2": 0, "kept_d2": 0}
    changed = 0
    n_mismatch = 0
    for i, key in enumerate(keys):
        rec = records[key][1]
        b = budgets[key]
        rb = rebuild_d2(index[key]["fields"], tok)
        if not verify_bitwise_vs_trace(rb["ids"], rb["mask"], key[0], key[1], zt, tidx):
            n_mismatch += 1
            continue
        if not (np.array_equal(rb["ids"], zd["ids"][i])
                and np.array_equal(rb["mask"], zd["mask"][i])):
            n_mismatch += 1
            continue
        # budget recompute
        if b["status"] == "ok":
            split = split_events_clean(rb["cleaned"]["scenario"], tok)
            head_ids, tag_ids, prose_ids = event_pieces(split, tok)
            bb = compute_event_budgets(split, head_ids, tag_ids, prose_ids, b["E"])
            if not (bb["status"] == "ok" and bb["n"] == b["n"]
                    and bb["need"] == b.get("need")
                    and bb["budget"] == b.get("budget")):
                n_mismatch += 1
                continue
        # row re-assembly
        if rec["outcome"] in ("kept_d2", "fallback_d2"):
            ids, mask = rb["ids"], rb["mask"]
        else:
            split = split_events_clean(rb["cleaned"]["scenario"], tok)
            pieces = {}
            for ev in rec["events"]:
                j = ev["k"] - 1
                pieces[j] = (split["events"][j][1]
                             if ev["source"] == "fit_verbatim"
                             else " " + ev["piece"])
            events_str = ("Events: " + split["head"]
                          + "".join(split["events"][j][0] + pieces[j]
                                    for j in range(b["n"])))
            ev_ids = tok.encode(events_str, add_special_tokens=False)
            if len(ev_ids) > b["E"] or len(ev_ids) != rec["events_tokens"]:
                n_mismatch += 1
                continue
            counts_new = dict(rb["counts"])
            counts_new["Events"] = {"skipped": False, "raw_ids": ev_ids,
                                    "raw_tokens": len(ev_ids)}
            ids, mask, meta_new = assemble(counts_new, rb["alloc"], tok)
            gates_ok = (sum(mask) - 2 <= 510
                        and all(meta_new[n]["kept_ids"] == rb["meta"][n]["kept_ids"]
                                for n in ("Background", "Calendar", "Covariates"))
                        and ids[0] == tok.cls_token_id and mask[0] == 1
                        and ids[sum(mask) - 1] == tok.sep_token_id)
            if not gates_ok:
                n_mismatch += 1
                continue
        if row_sha(ids, mask) != rec["ids_sha256"] or \
                not np.array_equal(ids, z["ids"][i]) or \
                not np.array_equal(mask, z["mask"][i]):
            n_mismatch += 1
            continue
        outcomes[rec["outcome"]] += 1
        if not (np.array_equal(ids, zd["ids"][i]) and np.array_equal(mask, zd["mask"][i])):
            changed += 1
        if (i + 1) % 200 == 0:
            print(f"[PF-DATA'] {i + 1}/2474", flush=True)

    br = json.loads((SUBDIR / "outputs" / "build_report.json").read_text())
    outcomes_ok = outcomes == br["outcomes"]
    changed_ok = changed == br["windows_input_changed_vs_d2"]

    ok = all([sha_ok, schema_ok, keys_ok, keys_eq, records_ok,
              fp_constant, svc_constant, n_mismatch == 0, outcomes_ok,
              changed_ok])
    save("PF-DATA-prime", {
        "ok": ok, "cache_npz_sha256_ok": sha_ok, "schema_ok": schema_ok,
        "keys_equal_test_rows": keys_ok,
        "keys_equal_d2_exp011_caches": keys_eq,
        "records_2474": records_ok, "build_fp_constant": fp_constant,
        "service_version_constant": svc_constant,
        "n_row_mismatches": n_mismatch, "outcomes": outcomes,
        "outcomes_match_build_report": outcomes_ok,
        "windows_input_changed": changed,
        "changed_match_build_report": changed_ok,
    })
    return ok


# --------------------------------------------------------------------- PF-D

def pf_d():
    from predict import ShardStore
    from predict_e33 import TextTokenStore

    driver = SUBDIR / "run_eval_e33.py"
    py = sys.executable
    exp012_token = sha256_file(PROJECT / "repro_mm_timesx_d2" / "protocol_d2.json")[:12]

    def run(args):
        return subprocess.run([py, str(driver)] + args, capture_output=True,
                              text=True, cwd=str(SUBDIR))

    r = run(["--dry-run"])
    dry_ok = r.returncode == 0 and '"mode": "dry-run (no inference performed)"' in r.stdout

    matrix = {
        "exp010_token": ["--itl", "48", "--i-approve-frozen-protocol", "580eb3c3f51d"],
        "exp011_token": ["--itl", "48", "--i-approve-frozen-protocol", "74b08ecac1ed"],
        "exp012_token": ["--itl", "48", "--i-approve-frozen-protocol", exp012_token],
        "wrong_token": ["--itl", "48", "--i-approve-frozen-protocol", "deadbeef0000"],
        "wrong_itl": ["--itl", "9", "--dry-run"],
        "token_plus_dryrun": ["--dry-run", "--i-approve-frozen-protocol",
                              sha256_file(SUBDIR / "protocol_e33.json")[:12]],
        "no_token": ["--itl", "48"],
    }
    rejects = {k: run(v).returncode for k, v in matrix.items()}
    rejects_ok = all(rc == 2 for k, rc in rejects.items() if k != "no_token") \
        and rejects["no_token"] == 1

    protocol = json.loads((SUBDIR / "protocol_e33.json").read_text())
    bad = json.loads(json.dumps(protocol))
    bad["text"]["cache"]["npz_sha256"] = "0" * 64
    raised = False
    try:
        TextTokenStore(bad)
    except Exception as e:
        raised = "sha256 mismatch" in str(e)

    scratch = PREFLIGHT / f"scratch_shards_{time.strftime('%Y%m%d_%H%M%S')}"
    store = ShardStore(out_dir=scratch)
    rows = [{"var_key": "d__v", "sample_id": "d__v__s", "domain": "d",
             "pred": np.zeros(12), "target": np.zeros(12), "d": 1.0}]
    store.save("M48T512_E33", "d", 2021, {"protocol_e33_sha256": "f" * 64}, rows)
    reused = store.reuse_or_init("M48T512_E33", "d", 2021,
                                 {"protocol_e33_sha256": "e" * 64})
    quarantined = (not reused
                   and not (scratch / "M48T512_E33__d__s2021__test.npz").exists()
                   and any(p.name.endswith("M48T512_E33__d__s2021__test.npz")
                           for p in (scratch / "quarantine").glob("*")))

    ok = dry_ok and rejects_ok and raised and quarantined
    save("PF-D", {"ok": ok, "dry_run_ok": dry_ok,
                  "reject_returncodes": rejects, "rejects_all_rc2": rejects_ok,
                  "exp012_token_rejected": rejects["exp012_token"] == 2,
                  "tampered_cache_sha_refused": raised,
                  "fingerprint_mismatch_quarantined": quarantined})
    return ok


# --------------------------------------------------------------------- PF-A

def pf_a(model):
    import torch
    from predict import predict_window
    from predict_mm import predict_window_mm
    from predict_e33 import predict_window_mm as predict_window_e33
    from predict_d2 import predict_window_mm as predict_window_d2
    itl, ns = PROTO["inference"]["itl_main"], PROTO["inference"]["num_samples"]

    z = np.load(D2_PF_NPZ, allow_pickle=False)
    keys = list(zip([str(x) for x in z["var_keys"]],
                    [str(x) for x in z["sample_ids"]]))
    import data_loader
    test_keys = {(vk, sid) for vk, sid, *_ in
                 data_loader.TimesXData(PROJECT / "data").test_rows()}
    assert not (set(keys) & test_keys), "PF-A key overlaps the test split"
    td = data_loader.TimesXData(PROJECT / "data")
    device = next(model.parameters()).device
    checks, all_ok = [], True
    for i, (vk, sid) in enumerate(keys):
        past, _tgt = td.split_window_by_id(vk, sid)
        ids = torch.from_numpy(z["ids_d0"][i].astype(np.int64)).unsqueeze(0).to(device)
        mask = torch.from_numpy(z["mask_d0"][i].astype(np.int64)).unsqueeze(0).to(device)
        tt = torch.zeros_like(ids)
        p_e33 = predict_window_e33(model, past, itl, ns, PF_SEED,
                                   text_ids=ids, text_mask=mask, text_typeids=tt)
        p_mm = predict_window_mm(model, past, itl, ns, PF_SEED,
                                 text_ids=ids, text_mask=mask, text_typeids=tt)
        p_d2 = predict_window_d2(model, past, itl, ns, PF_SEED,
                                 text_ids=ids, text_mask=mask, text_typeids=tt)
        p_root = predict_window(model, past, itl, ns, PF_SEED)
        p_e33_none = predict_window_e33(model, past, itl, ns, PF_SEED)
        b1 = bool(np.array_equal(p_e33, p_mm))
        b2 = bool(np.array_equal(p_e33_none, p_root))
        b3 = bool(np.array_equal(p_e33, p_d2))
        checks.append({"var_key": vk, "sample_id": sid,
                       "d0_bitwise_vs_exp011_entry": b1,
                       "text_none_bitwise_vs_root": b2,
                       "d0_bitwise_vs_exp012_entry": b3})
        all_ok &= b1 and b2 and b3
    save("PF-A", {"ok": all_ok, "n_windows": len(checks), "seed": PF_SEED,
                  "checks": checks})
    return all_ok


# --------------------------------------------------------------------- PF-E

def select_pf_e_windows():
    """First 2 compressed + first 2 fallback E-Extract val windows (frozen
    diagnostic v3.3 outputs), file order."""
    want = {"compressed": 2, "fallback_d2": 2}
    got = {"compressed": [], "fallback_d2": []}
    with open(DIAG_OUT / "attempts_val.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("scheme") != "E-Extract":
                continue
            o = r.get("outcome")
            if o in want and len(got[o]) < want[o]:
                got[o].append((r["var_key"], r["sample_id"]))
            if all(len(got[k]) == want[k] for k in want):
                break
    assert all(len(got[k]) == want[k] for k in want), \
        f"PF-E selection incomplete: { {k: len(v) for k, v in got.items()} }"
    return got


def load_diag_rows(keys):
    """(vk,sid) -> {'D2': (ids,mask), 'E-Extract': (ids,mask)} from the frozen
    diagnostic final_inputs_val.npz."""
    z = np.load(DIAG_OUT / "final_inputs_val.npz", allow_pickle=False)
    out = {}
    for i in range(len(z["var_keys"])):
        k = (str(z["var_keys"][i]), str(z["sample_ids"][i]))
        m = str(z["method"][i])
        if k in keys and m in ("D2", "E-Extract"):
            out.setdefault(k, {})[m] = (z["ids"][i], z["mask"][i])
    for k in keys:
        assert k in out and set(out[k]) == {"D2", "E-Extract"}, \
            f"diagnostic rows missing for {k}"
    return out


def pf_e(model):
    import torch
    from predict_e33 import predict_window_mm as predict_window_e33
    itl, ns = PROTO["inference"]["itl_main"], PROTO["inference"]["num_samples"]

    sel = select_pf_e_windows()
    flat = sel["compressed"] + sel["fallback_d2"]
    rows = load_diag_rows(set(flat))
    import data_loader
    test_keys = {(vk, sid) for vk, sid, *_ in
                 data_loader.TimesXData(PROJECT / "data").test_rows()}
    assert not (set(flat) & test_keys), "PF-E key overlaps the test split"
    td = data_loader.TimesXData(PROJECT / "data")
    device = next(model.parameters()).device
    orig_generate = model.generate
    captured = {}

    def spy_generate(*args, **kwargs):
        captured["kwargs"] = {k: (v if not torch.is_tensor(v) else v.clone())
                              for k, v in kwargs.items()}
        return orig_generate(*args, **kwargs)

    checks, all_ok = [], True
    for vk, sid in flat:
        cls = "compressed" if (vk, sid) in sel["compressed"] else "fallback"
        past, _tgt = td.split_window_by_id(vk, sid)
        ids2, mask2 = rows[(vk, sid)]["E-Extract"]
        ids1, mask1 = rows[(vk, sid)]["D2"]
        t2 = (torch.from_numpy(ids2.astype(np.int64)).unsqueeze(0).to(device),
              torch.from_numpy(mask2.astype(np.int64)).unsqueeze(0).to(device))
        t1 = (torch.from_numpy(ids1.astype(np.int64)).unsqueeze(0).to(device),
              torch.from_numpy(mask1.astype(np.int64)).unsqueeze(0).to(device))
        tt2 = torch.zeros_like(t2[0])

        if cls == "compressed":
            inputs_differ = bool(not np.array_equal(ids2, ids1)
                                 or not np.array_equal(mask2, mask1))
        else:
            inputs_differ = not (bool(np.array_equal(ids2, ids1)
                                      and np.array_equal(mask2, mask1)))
            assert not inputs_differ, \
                f"fallback window {vk}|{sid}: E33 input != D2 input"

        model.generate = spy_generate
        try:
            p2 = predict_window_e33(model, past, itl, ns, PF_SEED,
                                    text_ids=t2[0], text_mask=t2[1],
                                    text_typeids=tt2)
        finally:
            model.generate = orig_generate
        kw = captured["kwargs"]
        tensors_ok = (torch.equal(kw["text_input_ids"], t2[0])
                      and kw["text_input_ids"].dtype == torch.int64
                      and tuple(kw["text_input_ids"].shape) == (1, 512)
                      and kw["text_input_ids"].device == t2[0].device
                      and torch.equal(kw["text_attention_mask"], t2[1])
                      and bool((kw["text_token_type_ids"] == 0).all())
                      and kw.get("text_inputs") is None)
        shape_ok = p2.shape == (12,)
        finite_ok = bool(np.isfinite(p2).all())

        p1 = predict_window_e33(model, past, itl, ns, PF_SEED,
                                text_ids=t1[0], text_mask=t1[1],
                                text_typeids=torch.zeros_like(t1[0]))
        repeat_bitwise = bool(np.array_equal(
            p2, predict_window_e33(model, past, itl, ns, PF_SEED,
                                   text_ids=t2[0], text_mask=t2[1],
                                   text_typeids=tt2)))
        if cls == "fallback":
            equiv_ok = bool(np.array_equal(p1, p2))
        else:
            equiv_ok = True
        rec = {"var_key": vk, "sample_id": sid, "class": cls,
               "inputs_differ_vs_d2": inputs_differ,
               "text_tensors_exact": tensors_ok,
               "pred_shape_ok": shape_ok, "pred_finite": finite_ok,
               "same_seed_repeat_bitwise": repeat_bitwise,
               "fallback_pred_bitwise_vs_d2": equiv_ok}
        if cls == "compressed":
            rec["pred_changed_vs_d2"] = bool(np.any(p2 != p1))
            rec["max_abs_diff_vs_d2"] = float(np.max(np.abs(p2 - p1)))
        checks.append(rec)
        all_ok &= (inputs_differ if cls == "compressed" else not inputs_differ) \
            and tensors_ok and shape_ok and finite_ok and repeat_bitwise and equiv_ok
    save("PF-E", {"ok": all_ok, "seed": PF_SEED,
                  "n_windows": len(checks),
                  "n_compressed": len(sel["compressed"]),
                  "n_fallback": len(sel["fallback_d2"]),
                  "note": "pred_changed_vs_d2 RECORDED ONLY for compressed "
                          "windows; fallback windows MUST be bitwise-equal",
                  "checks": checks})
    return all_ok, flat


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


def pf_c(model, pf_window):
    import torch
    from predict_e33 import predict_window_mm as predict_window_e33
    itl, ns = PROTO["inference"]["itl_main"], PROTO["inference"]["num_samples"]

    vk, sid = pf_window
    rows = load_diag_rows({(vk, sid)})
    ids2, mask2 = rows[(vk, sid)]["E-Extract"]
    import data_loader
    td = data_loader.TimesXData(PROJECT / "data")
    past, _tgt = td.split_window_by_id(vk, sid)
    device = next(model.parameters()).device
    t_ids = torch.from_numpy(ids2.astype(np.int64)).unsqueeze(0).to(device)
    t_mask = torch.from_numpy(mask2.astype(np.int64)).unsqueeze(0).to(device)
    tt = torch.zeros_like(t_ids)

    before = _model_state_digest(model)
    torch.cuda.reset_peak_memory_stats()
    x = torch.from_numpy(np.asarray(past, dtype=np.float32)).unsqueeze(0).to(device)
    torch.manual_seed(PF_SEED)
    with torch.inference_mode():
        raw = model.generate(inputs=x, text_inputs=None, text_input_ids=t_ids,
                             text_attention_mask=t_mask, text_token_type_ids=tt,
                             vision_inputs=None, revin=True, num_samples=ns,
                             max_output_length=12, inference_token_len=itl)
    peak_gb = torch.cuda.max_memory_allocated() / 2 ** 30
    shape_ok = list(raw.shape) == [1, ns, 12]
    finite_ok = bool(torch.isfinite(raw).all())
    p1 = raw.to(torch.float64).mean(dim=1).squeeze(0).cpu().numpy()
    t0 = time.time()
    p2 = predict_window_e33(model, past, itl, ns, PF_SEED, t_ids, t_mask, tt)
    ms = (time.time() - t0) * 1000
    repeat_bitwise = bool(np.array_equal(p1, p2))
    after = _model_state_digest(model)
    state_ok = before == after

    child = subprocess.run(
        [sys.executable, str(SUBDIR / "run_preflight_e33.py"), "--pf-c-child",
         vk, sid, "--seed", str(PF_SEED)],
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
        "input": "real compressed E-Extract v3.3 val tensors",
        "raw_shape": list(raw.shape), "raw_finite": finite_ok,
        "same_seed_repeat_bitwise": repeat_bitwise,
        "weights_unchanged": state_ok,
        "cross_process_bitwise": cross_ok, "cross_process_detail": cross_detail,
        "ms_per_window_with_text": ms, "peak_vram_gb": peak_gb,
    })
    return ok


def pf_c_child(vk, sid, seed):
    import torch
    from load_aurora import load_aurora
    from predict_e33 import predict_window_mm as predict_window_e33
    model, _ = load_aurora(device="cuda")
    rows = load_diag_rows({(vk, sid)})
    ids2, mask2 = rows[(vk, sid)]["E-Extract"]
    device = next(model.parameters()).device
    past, _tgt = __import__("data_loader").TimesXData(
        PROJECT / "data").split_window_by_id(vk, sid)
    pred = predict_window_e33(
        model, past, PROTO["inference"]["itl_main"],
        PROTO["inference"]["num_samples"], seed,
        torch.from_numpy(ids2.astype(np.int64)).unsqueeze(0).to(device),
        torch.from_numpy(mask2.astype(np.int64)).unsqueeze(0).to(device),
        torch.zeros(1, 512, dtype=torch.long, device=device))
    np.save(PREFLIGHT / "pf_c_child_pred.npy", pred)
    print(f"[pf-c-child] wrote pred for {vk} {sid}")
    return 0


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gpu", action="store_true",
                    help="run only CPU sections (PF-DATA', PF-D)")
    ap.add_argument("--pf-c-child", default=None, nargs=2, metavar=("VK", "SID"),
                    help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    global PROTO
    PROTO = json.loads((SUBDIR / "protocol_e33.json").read_text())

    if args.pf_c_child:
        return pf_c_child(args.pf_c_child[0], args.pf_c_child[1], args.seed)

    ok = pf_data_prime() and pf_d()
    if not args.skip_gpu:
        from load_aurora import load_aurora
        model, _ = load_aurora(device="cuda")
        ok = pf_a(model) and ok
        pf_e_ok, first_compressed = pf_e(model)
        ok = pf_e_ok and ok
        ok = pf_c(model, first_compressed[0]) and ok
    print("[preflight] RESULT:", "ALL OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
