#!/usr/bin/env python3
"""M48T512 preflight (sections MM-DATA, MM-A..E) — synthetic + train/val windows
ONLY; zero test-window inference.

Set CUBLAS_WORKSPACE_CONFIG before any CUDA init; determinism pack comes from
load_aurora before first inference. Evidence: repro_mm_timesx_v1/preflight/*.json
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SUBDIR = Path(__file__).resolve().parent
PROJECT = SUBDIR.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(SUBDIR))

PREFLIGHT = SUBDIR / "preflight"
PREFLIGHT.mkdir(exist_ok=True)
RESULTS = {"sections": {}}
PROTO = None


def save(section, report):
    RESULTS["sections"][section] = report
    name = f"{section.lower().replace('-', '_')}.json"
    (PREFLIGHT / name).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    ok = report.get("ok", False)
    print(f"[preflight] {section}: {'OK' if ok else 'FAILED'} -> {name}")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------ MM-DATA

def mm_data():
    import text_cache as tc
    index = tc.build_index()
    by_var = tc.group_by_var(index)
    from data_loader import TimesXData
    rows = TimesXData(PROJECT / "data").test_rows()

    bad = tc.check_fields(index, rows, by_var)
    cov, _ = tc.check_covariate_span(index, rows, "test_preflight", by_var)
    ev, _ = tc.scan_events(index, rows, "test_preflight", by_var)
    ev_en, _ = tc.scan_events_english(index, rows, "test_preflight", by_var)
    ok = not bad and cov["n_violations"] == 0
    save("MM-DATA", {
        "ok": ok,
        "n_test_windows": len(rows),
        "mapping_problems": len(bad),
        "covariate_violations": cov["n_violations"],
        "covariate_tolerance_hits": cov["n_tolerance_hits"],
        "events_flagged": ev["n_flagged"],
        "events_note": ev["note"],
        "events_english_input_windows": ev_en["n_windows_flagged_input"],
        "events_english_full_windows": ev_en["n_windows_flagged_full"],
        "events_english_info_noyear_windows": ev_en["n_windows_info_noyear_input"],
        "events_english_review": "PRIMARY flags (year-explicit date >= pred start "
                                 "in the input Events decode) are listed in "
                                 "preflight/events_english_test_preflight_input.csv "
                                 "and require manual classification sign-off before "
                                 "the full run (gate = user verdict, not this scan; "
                                 "flagged != leakage — advance-published schedules "
                                 "expected). Year-less month-day mentions are "
                                 "informational only.",
    })
    return ok


# --------------------------------------------------------------------- MM-A

def _trainval_windows(n_domains=4):
    from data_loader import TimesXData, DOMAIN_NAMES
    td = TimesXData(PROJECT / "data")
    windows = []
    daily = [d for d in DOMAIN_NAMES if td.manifest["variables"][
        td.domain_vars[d][0]]["frequency"] == "daily"]
    weekly = [d for d in DOMAIN_NAMES if d not in daily]
    for d in (daily[:2] + weekly[:2]):
        rows = td.trainval_rows(d, max_per_var=1, splits=("train", "val"))
        if rows:
            windows.append(rows[0])
    return windows[:n_domains]


_ZIP_CACHE = {}


def _zip_index():
    import text_cache as tc
    if "by_var" not in _ZIP_CACHE:
        index = tc.build_index()
        _ZIP_CACHE["by_var"] = (index, tc.group_by_var(index))
    return _ZIP_CACHE["by_var"]


_TOK_CACHE = {}


def _tokenizer():
    from text_builder import find_bert_config
    from transformers import BertTokenizer
    if "tok" not in _TOK_CACHE:
        _TOK_CACHE["tok"] = BertTokenizer.from_pretrained(
            find_bert_config(), local_files_only=True)
    return _TOK_CACHE["tok"]


def _build_text_for(vk, sid):
    """Train/val windows are not in the frozen test cache; build text on the fly
    from the zip via the SAME frozen builder (MM-D validates builder==cache)."""
    import text_cache as tc
    from text_builder import build_tokens
    index, by_var = _zip_index()
    rec = tc.resolve_independently(index, vk, sid, by_var)
    ids, mask, meta = build_tokens(rec["fields"], _tokenizer())
    return np.array(ids, dtype=np.int32), np.array(mask, dtype=np.int32), meta, rec


def mm_a(model):
    """predict_window_mm(text=None) bitwise == root predict_window."""
    import torch
    from predict import predict_window
    from predict_mm import predict_window_mm
    itl, ns = PROTO["inference"]["itl_main"], PROTO["inference"]["num_samples"]

    rows = []
    for d in ("climate", "finance", "pets", "traffic"):
        td_rows = __import__("data_loader").TimesXData(
            PROJECT / "data").trainval_rows(d, max_per_var=1, splits=("train",))
        if td_rows:
            rows.append(td_rows[0])
    checks, all_ok = [], True
    for vk, sid, dom, past, tgt, dv in rows:
        w = 424242
        p_root = predict_window(model, past, itl, ns, w)
        p_mm = predict_window_mm(model, past, itl, ns, w)
        bitwise = bool(np.array_equal(p_root, p_mm))
        checks.append({"var_key": vk, "sample_id": sid, "bitwise": bitwise,
                       "max_abs_diff": float(np.max(np.abs(p_root - p_mm)))})
        all_ok &= bitwise
    save("MM-A", {"ok": all_ok, "n_windows": len(checks), "checks": checks})
    return all_ok


# --------------------------------------------------------------------- MM-B

def mm_b(model):
    """Text reaches BERT/fusion; rear-token perturbation with EXACT position
    control: fix input indices 0..125 (index 0 is [CLS]; the 125 retained
    feature rows correspond to content positions 1..125) AND the whole
    attention mask; equal-length replacement of rear valid tokens from 126."""
    import torch
    from predict_mm import predict_window_mm
    from text_builder import find_bert_config
    from transformers import BertTokenizer
    itl, ns = PROTO["inference"]["itl_main"], PROTO["inference"]["num_samples"]

    tok = BertTokenizer.from_pretrained(find_bert_config(), local_files_only=True)
    td = __import__("data_loader").TimesXData(PROJECT / "data")
    vk, sid, dom, past, tgt, dv = td.trainval_rows("climate", max_per_var=2,
                                                   splits=("train",))[0]
    ids_a, mask_a, meta_a, rec_a = _build_text_for(vk, sid)
    content_len = int(mask_a.sum())
    assert content_len > 150, f"window too short for rear test: {content_len}"

    device = next(model.parameters()).device
    caps = {}
    core = model.model if hasattr(model, "model") else model  # AuroraForPrediction.model = AuroraModel
    te = core.TextEncoder
    orig_extract = te.extract_bert_features

    def spy_extract(input_dict):
        caps["input_ids"] = input_dict["input_ids"].detach().cpu().numpy()
        caps["attention_mask"] = input_dict["attention_mask"].detach().cpu().numpy()
        out = orig_extract(input_dict)
        caps["fixed_features"] = out[2].detach().cpu().numpy()
        caps["valid_counts"] = list(out[3])
        return out
    te.extract_bert_features = spy_extract
    cross_caps = {}
    def _h1(m, i, o):
        cross_caps.setdefault("out", o.detach().cpu().numpy())  # no return: hook must not replace output
    h1 = te.cross_text.register_forward_hook(_h1)
    guider_caps = {}
    def _h2(m, i, o):
        guider_caps.setdefault("attn_text_not_none",
                               bool(o[1] is not None) if isinstance(o, tuple) else True)
    h2 = core.TextGuider.register_forward_hook(_h2)
    conn_caps = {}
    def _h3(m, i, o):
        conn_caps.setdefault("from_text_not_none", bool(o[0] is not None))
    h3 = core.ModalityConnector.register_forward_hook(_h3)

    torch.manual_seed(7)
    _ = model.generate(
        inputs=torch.from_numpy(np.asarray(past, dtype=np.float32)).unsqueeze(0).to(device),
        text_inputs=None,
        text_input_ids=torch.from_numpy(ids_a.astype(np.int64)).unsqueeze(0).to(device),
        text_attention_mask=torch.from_numpy(mask_a.astype(np.int64)).unsqueeze(0).to(device),
        text_token_type_ids=torch.zeros(1, len(ids_a), dtype=torch.long, device=device),
        vision_inputs=None, revin=True, num_samples=ns, max_output_length=12,
        inference_token_len=itl)

    h1.remove(); h2.remove(); h3.remove()
    del te.extract_bert_features  # un-shadow the class method


    hooks_ok = {
        "ids_match_passed": bool(np.array_equal(caps["input_ids"][0], ids_a)),
        "mask_match_passed": bool(np.array_equal(caps["attention_mask"][0], mask_a)),
        "fixed_features_shape": list(caps["fixed_features"].shape),  # expect [1,125,768]
        "cross_text_shape": list(cross_caps["out"].shape),           # expect [1,10,256]
        "textguider_attn_not_none": bool(guider_caps["attn_text_not_none"]),
        "from_text_not_none": bool(conn_caps["from_text_not_none"]),
    }
    hooks_ok["fixed_features_shape_ok"] = hooks_ok["fixed_features_shape"] == [1, 125, 768]
    hooks_ok["cross_text_shape_ok"] = hooks_ok["cross_text_shape"] == [1, 10, 256]

    # different text -> different prediction (non-degenerate check)
    ids_b, mask_b, meta_b, rec_b = _build_text_for(
        *td.trainval_rows("traffic", max_per_var=1, splits=("train",))[0][:2])
    seed = 424242
    pred_a = predict_window_mm(model, past, itl, ns, seed,
                               torch.from_numpy(ids_a.astype(np.int64)).unsqueeze(0).to(device),
                               torch.from_numpy(mask_a.astype(np.int64)).unsqueeze(0).to(device),
                               torch.zeros(1, len(ids_a), dtype=torch.long, device=device))
    pred_b = predict_window_mm(model, past, itl, ns, seed,
                               torch.from_numpy(ids_b.astype(np.int64)).unsqueeze(0).to(device),
                               torch.from_numpy(mask_b.astype(np.int64)).unsqueeze(0).to(device),
                               torch.zeros(1, len(ids_b), dtype=torch.long, device=device))
    diff_text = float(np.max(np.abs(pred_a - pred_b)))

    # REAR PERTURBATION: exact positions — fix indices [0,126) ([CLS] at 0 +
    # content positions 1..125, the sources of the 125 retained feature rows)
    # + whole mask; equal-length replacement of rear valid tokens [126, content_len-1)
    ids_c = ids_a.copy()
    alt = tok.encode("the quick brown fox jumps over the lazy dog and reports "
                     "an unusual market movement today", add_special_tokens=False)
    rear = content_len - 1 - 126  # exclude SEP at content_len-1
    reps = np.resize(np.array(alt, dtype=np.int32), rear)
    ids_c[126:content_len - 1] = reps
    exact = {
        "front_126_unchanged": bool(np.array_equal(ids_c[:126], ids_a[:126])),
        "mask_unchanged": bool(np.array_equal(mask_c := mask_a, mask_a)),
        "sep_unchanged": bool(ids_c[content_len - 1] == ids_a[content_len - 1]),
        "rear_changed": bool((ids_c[126:content_len - 1] != ids_a[126:content_len - 1]).any()),
        "same_length": bool(len(ids_c) == len(ids_a)),
    }
    caps2 = {}
    def spy2(input_dict):
        out = orig_extract(input_dict)
        caps2["fixed"] = out[2].detach().cpu().numpy()
        return out
    te.extract_bert_features = spy2
    pred_c = predict_window_mm(model, past, itl, ns, seed,
                               torch.from_numpy(ids_c.astype(np.int64)).unsqueeze(0).to(device),
                               torch.from_numpy(mask_a.astype(np.int64)).unsqueeze(0).to(device),
                               torch.zeros(1, len(ids_c), dtype=torch.long, device=device))
    del te.extract_bert_features

    ff_a, ff_c = caps["fixed_features"], caps2["fixed"]
    feat_diff = float(np.max(np.abs(ff_a - ff_c)))
    frac_changed = float(np.mean(np.any(ff_a != ff_c, axis=-1)))
    pred_diff = float(np.max(np.abs(pred_a - pred_c)))

    ok = (all(v for k, v in hooks_ok.items() if k.endswith(("_ok", "_not_none",
                                                             "_match_passed", "_unchanged",
                                                             "_changed", "_length")))
          and all(exact.values())
          and diff_text > 0.0)
    save("MM-B", {
        "ok": ok,
        "window": {"var_key": vk, "sample_id": sid, "content_len": content_len},
        "hooks": hooks_ok,
        "different_text_max_pred_diff": diff_text,
        "rear_perturbation": {
            "controls": exact,
            "retained_feature_max_abs_diff": feat_diff,
            "retained_features_changed_fraction": frac_changed,
            "pred_max_abs_diff": pred_diff,
            "interpretation": "rear tokens (indices >= 126) were replaced at equal "
                              "length with front input tokens while indices 0..125 "
                              "([CLS] + the content positions feeding the 125 "
                              "retained features) and the full "
                              "attention mask were fixed; any feature/prediction change "
                              "quantifies the contextualization path (BERT attends "
                              "over all valid positions before the first-125 "
                              "retention), NOT direct retention of rear features.",
        },
    })
    return ok


# --------------------------------------------------------------------- MM-C

def _model_state_digest(model):
    import torch
    tensors, bn = {}, {}
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                            torch.nn.BatchNorm3d)):
            bn[name] = [float(mod.running_mean.sum()), float(mod.running_var.sum()),
                        int(mod.num_batches_tracked)]
    for name, p in model.state_dict().items():
        tensors[name] = sha256_bytes(p.detach().cpu().numpy().tobytes())
    return {"n_tensors": len(tensors), "tensor_sha": sha256_bytes(
                "".join(tensors[k] for k in sorted(tensors)).encode()),
            "n_bn_modules": len(bn), "bn": bn}


def mm_c(model):
    import torch
    from predict_mm import predict_window_mm
    itl, ns = PROTO["inference"]["itl_main"], PROTO["inference"]["num_samples"]
    td = __import__("data_loader").TimesXData(PROJECT / "data")
    vk, sid, dom, past, tgt, dv = td.trainval_rows("finance", max_per_var=1,
                                                   splits=("train",))[0]
    ids, mask, meta, rec = _build_text_for(vk, sid)
    device = next(model.parameters()).device
    t_ids = torch.from_numpy(ids.astype(np.int64)).unsqueeze(0).to(device)
    t_mask = torch.from_numpy(mask.astype(np.int64)).unsqueeze(0).to(device)
    t_tt = torch.zeros(1, len(ids), dtype=torch.long, device=device)
    seed = 424242

    before = _model_state_digest(model)
    torch.cuda.reset_peak_memory_stats()
    x = torch.from_numpy(np.asarray(past, dtype=np.float32)).unsqueeze(0).to(device)
    torch.manual_seed(seed)
    with torch.inference_mode():
        raw = model.generate(inputs=x, text_inputs=None, text_input_ids=t_ids,
                             text_attention_mask=t_mask, text_token_type_ids=t_tt,
                             vision_inputs=None, revin=True, num_samples=ns,
                             max_output_length=12, inference_token_len=itl)
    peak_gb = torch.cuda.max_memory_allocated() / 2 ** 30
    shape_ok = list(raw.shape) == [1, ns, 12]
    finite_ok = bool(torch.isfinite(raw).all())
    p1 = raw.to(torch.float64).mean(dim=1).squeeze(0).cpu().numpy()
    t0 = time.time()
    p2 = predict_window_mm(model, past, itl, ns, seed, t_ids, t_mask, t_tt)
    ms = (time.time() - t0) * 1000
    repeat_bitwise = bool(np.array_equal(p1, p2))
    after = _model_state_digest(model)
    state_ok = before == after

    # cross-process bitwise: child recomputes the same window from scratch
    child = subprocess.run(
        [sys.executable, str(SUBDIR / "run_preflight_mm.py"), "--mm-c-child",
         "--domain", dom, "--seed", str(seed)],
        capture_output=True, text=True)
    cross_ok = False
    cross_detail = {"rc": child.returncode}
    if child.returncode == 0:
        child_pred = np.load(PREFLIGHT / "mm_c_child_pred.npy")
        cross_ok = bool(np.array_equal(child_pred, p1))
        cross_detail["bitwise"] = cross_ok

    ok = shape_ok and finite_ok and repeat_bitwise and state_ok and cross_ok
    save("MM-C", {
        "ok": ok, "raw_shape": list(raw.shape), "raw_finite": finite_ok,
        "same_seed_repeat_bitwise": repeat_bitwise,
        "weights_unchanged": state_ok,
        "n_tensors": after["n_tensors"], "n_bn_modules": after["n_bn_modules"],
        "cross_process_bitwise": cross_ok, "cross_process_detail": cross_detail,
        "ms_per_window_with_text": ms, "peak_vram_gb": peak_gb,
    })
    return ok


def mm_c_child(domain, seed):
    """Hidden child mode: recompute ONE trainval window with text, dump npy."""
    import torch
    from load_aurora import load_aurora
    from predict_mm import predict_window_mm
    model, _ = load_aurora(device="cuda")
    td = __import__("data_loader").TimesXData(PROJECT / "data")
    vk, sid, dom, past, tgt, dv = td.trainval_rows(domain, max_per_var=1,
                                                   splits=("train",))[0]
    ids, mask, meta, rec = _build_text_for(vk, sid)
    device = next(model.parameters()).device
    pred = predict_window_mm(
        model, past, PROTO["inference"]["itl_main"], PROTO["inference"]["num_samples"],
        seed,
        torch.from_numpy(ids.astype(np.int64)).unsqueeze(0).to(device),
        torch.from_numpy(mask.astype(np.int64)).unsqueeze(0).to(device),
        torch.zeros(1, len(ids), dtype=torch.long, device=device))
    np.save(PREFLIGHT / "mm_c_child_pred.npy", pred)
    print(f"[mm-c-child] wrote pred for {vk} {sid}")
    return 0


# --------------------------------------------------------------------- MM-D

def mm_d():
    """Budget traceability: cache rows EXACTLY equal re-derived expected IDs."""
    import text_cache as tc
    from text_builder import BLOCKS, build_tokens, find_bert_config
    from transformers import BertTokenizer
    tok = BertTokenizer.from_pretrained(find_bert_config(), local_files_only=True)
    index = tc.build_index()
    by_var = tc.group_by_var(index)
    from data_loader import TimesXData
    td = TimesXData(PROJECT / "data")
    rows = td.test_rows()

    z = np.load(SUBDIR / "cache" / "text_tokens_M48T512.npz", allow_pickle=False)
    cache = {(str(vk), str(sid)): (z["ids"][i], z["mask"][i])
             for i, (vk, sid) in enumerate(zip(z["var_keys"], z["sample_ids"]))}
    meta = {}
    with open(SUBDIR / "cache" / "text_meta_M48T512.jsonl", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            meta[(m["var_key"], m["sample_id"])] = m

    key_set_ok = set(cache.keys()) == {(vk, sid) for vk, sid, *_ in rows} == set(meta.keys())

    daily_unknown = [(vk, sid) for vk, sid, *_ in rows
                     if tc.resolve_independently(index, vk, sid, by_var)
                     ["fields"]["holiday_info"].strip().lower() == "unknown"]
    cal_skipped = [k for k, m in meta.items() if "Calendar" in m["blocks_skipped"]]
    unknown_ok = (len(daily_unknown) == 618 and len(cal_skipped) == 618
                  and set(daily_unknown) == set(cal_skipped))

    step = max(1, len(rows) // 40)
    sample = rows[::step][:40]
    # force-include a daily-Unknown and a weekly window in the sample
    sample.append(next(r for r in rows if (r[0], r[1]) in set(daily_unknown)))
    sample.append(next(r for r in rows
                       if td.manifest["variables"][r[0]]["frequency"] != "daily"))

    row_checks, all_ok = [], True
    for vk, sid, dom, _p, _t, _d in sample:
        rec = tc.resolve_independently(index, vk, sid, by_var)
        ids, mask, meta_b = build_tokens(rec["fields"], tok)
        cids, cmask = cache[(vk, sid)]
        ids_eq = bool(np.array_equal(ids, cids))
        mask_eq = bool(np.array_equal(mask, cmask))
        m = meta[(vk, sid)]
        bounds_ok = all(
            (bm["start"] is None and mm_["start"] is None) or
            (bm["start"] == mm_["start"] and bm["end"] == mm_["end"])
            for bm, mm_ in zip(meta_b["blocks"], m["blocks"]))
        decode_eq = tok.decode(cids[:int(cmask.sum())], skip_special_tokens=False) \
            == m["decoded_text"]
        raw_chars_eq = all(bm["raw_chars"] == mm_["raw_chars"]
                           for bm, mm_ in zip(meta_b["blocks"], m["blocks"]))
        no_double_trunc = all(bm["kept_tokens"] <= b and
                              (not bm["skipped"] or bm["kept_tokens"] == 0)
                              for bm, (_n, _f, b) in zip(meta_b["blocks"], BLOCKS))
        content_ok = meta_b["content_tokens"] == m["content_tokens"] <= 510
        rec_ok = ids_eq and mask_eq and bounds_ok and decode_eq and raw_chars_eq \
            and no_double_trunc and content_ok
        all_ok &= rec_ok
        row_checks.append({"var_key": vk, "sample_id": sid, "ids_exact": ids_eq,
                           "mask_exact": mask_eq, "bounds_ok": bounds_ok,
                           "decode_matches_meta": decode_eq,
                           "raw_chars_match_meta": raw_chars_eq,
                           "no_double_truncation": no_double_trunc,
                           "content_tokens": meta_b["content_tokens"]})

    save("MM-D", {
        "ok": bool(all_ok and key_set_ok and unknown_ok),
        "key_sets_equal": key_set_ok,
        "unknown_calendar_consistency": {"n_unknown": len(daily_unknown),
                                         "n_calendar_skipped": len(cal_skipped),
                                         "sets_equal": set(daily_unknown) == set(cal_skipped),
                                         "expected_618": True,
                                         "ok": unknown_ok},
        "n_sampled_rows": len(row_checks),
        "all_sampled_rows_exact": all_ok,
        "note": "token IDs are verified by EXACT equality against the frozen "
                "rules; decoded text equals meta and is for human inspection "
                "(BERT lowercases/punctuation-splits, so verbatim equality with "
                "source fields is NOT expected)",
        "rows": row_checks,
    })
    return bool(all_ok and key_set_ok and unknown_ok)


# --------------------------------------------------------------------- MM-E

def mm_e():
    """Resume/flag drills on a scratch store (no test-window data)."""
    import shutil

    from predict import ShardStore, ShardVerificationError, verify_shard_coverage
    from predict_mm import shard_fingerprint_mm
    from run_eval_mm import EXP010_TOKEN, protocol_mm_sha
    scratch = PREFLIGHT / "mm_e_scratch"
    store = ShardStore(out_dir=scratch)
    td = __import__("data_loader").TimesXData(PROJECT / "data")
    dom = "pets"
    rows = td.trainval_rows(dom, max_per_var=1, splits=("train",))[:4]
    expect = {k: (np.asarray(t, dtype=np.float64), float(d))
              for k, t, d in [((vk, sid), tgt, dv) for vk, sid, _dm, _p, tgt, dv in rows]}
    fp = shard_fingerprint_mm(PROTO, 48, 31337, "preflight-weights-sha")
    out_rows = [{"var_key": vk, "sample_id": sid, "domain": dom,
                 "pred": np.full(12, 0.5), "target": tgt, "d": dv}
                for vk, sid, _dm, _p, tgt, dv in rows]
    store.save("PREF", dom, 31337, fp, out_rows)

    drills = {}

    reuse = store.reuse_or_init("PREF", dom, 31337, fp, expect=expect)
    drills["clean_reuse"] = {"ok": reuse is True}

    fp_bad = dict(fp)
    fp_bad["text_cache_npz_sha256"] = "0" * 64  # simulated one-byte cache change
    reuse_bad = store.reuse_or_init("PREF", dom, 31337, fp_bad, expect=expect)
    quarantined = any("PREF" in p.name for p in
                      (scratch / "quarantine").glob("*"))
    drills["fingerprint_mismatch_quarantines"] = {"ok": reuse_bad is False and quarantined}
    store.save("PREF", dom, 31337, fp, out_rows)  # restore

    done_path = store._paths("PREF", dom, 31337)[2]
    dn = json.loads(done_path.read_text())
    dn["content_sha256"] = "b" * 64
    done_path.write_text(json.dumps(dn))
    reuse_done = store.reuse_or_init("PREF", dom, 31337, fp, expect=expect)
    drills["done_tamper_rejected"] = {"ok": reuse_done is False}
    store.save("PREF", dom, 31337, fp, out_rows)

    npz_path, _, _ = store._paths("PREF", dom, 31337)
    with np.load(npz_path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    data["pred"] = data["pred"] + 1.0  # byte-level content swap WITHOUT ShardStore.save
    with open(npz_path, "wb") as f:
        np.savez(f, **data)
    reuse_npz = store.reuse_or_init("PREF", dom, 31337, fp, expect=expect)
    drills["npz_content_swap_rejected"] = {"ok": reuse_npz is False}
    store.save("PREF", dom, 31337, fp, out_rows)

    dup_entries = [(vk, sid, tgt, dv) for vk, sid, tgt, dv in
                   [(r["var_key"], r["sample_id"], r["target"], r["d"]) for r in out_rows]]
    dup_entries.append(dup_entries[0])
    dup_ok = False
    try:
        verify_shard_coverage(dup_entries, expect)
    except ShardVerificationError:
        dup_ok = True
    missing_ok = False
    try:
        verify_shard_coverage(dup_entries[:-2], expect)
    except ShardVerificationError:
        missing_ok = True
    drills["coverage_dup_and_missing_rejected"] = {"ok": dup_ok and missing_ok}

    # dry-run: zero inference; flag matrix via subprocess
    before_files = sorted(p.name for p in (SUBDIR / "predictions").glob("*"))
    r = subprocess.run([sys.executable, str(SUBDIR / "run_eval_mm.py"), "--dry-run"],
                       capture_output=True, text=True)
    after_files = sorted(p.name for p in (SUBDIR / "predictions").glob("*"))
    drills["dry_run_zero_inference"] = {
        "ok": r.returncode == 0 and before_files == after_files,
        "rc": r.returncode}
    matrix = []
    for token, expect_rc, label in (
            (EXP010_TOKEN, 2, "exp010_token_rejected"),
            ("0" * 12, 2, "wrong_token_rejected"),
            (None, 1, "no_token_locked")):
        cmd = [sys.executable, str(SUBDIR / "run_eval_mm.py"), "--itl", "48"]
        if token:
            cmd += ["--i-approve-frozen-protocol", token]
        r2 = subprocess.run(cmd, capture_output=True, text=True)
        matrix.append({"case": label, "rc": r2.returncode, "expected_rc": expect_rc,
                       "ok": r2.returncode == expect_rc})
    r3 = subprocess.run([sys.executable, str(SUBDIR / "run_eval_mm.py"), "--itl", "48",
                         "--dry-run", "--i-approve-frozen-protocol",
                         protocol_mm_sha()[:12]], capture_output=True, text=True)
    matrix.append({"case": "dryrun_plus_token_rejected", "rc": r3.returncode,
                   "expected_rc": 2, "ok": r3.returncode == 2})
    drills["flag_matrix"] = {"ok": all(m["ok"] for m in matrix), "cases": matrix}

    shutil.rmtree(scratch, ignore_errors=True)
    ok = all(d["ok"] for d in drills.values())
    save("MM-E", {"ok": ok, "drills": drills})
    return ok


# -------------------------------------------------------------------- main

def main():
    global PROTO
    ap = argparse.ArgumentParser()
    ap.add_argument("--mm-c-child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--domain", default=None)
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument("--skip-gpu", action="store_true",
                    help="run only CPU sections (MM-DATA, MM-D, MM-E)")
    args = ap.parse_args()

    PROTO = json.loads((SUBDIR / "protocol_mm.json").read_text())
    t0 = time.time()

    if args.mm_c_child:
        return mm_c_child(args.domain, args.seed)

    ok = mm_data()
    ok &= mm_d()
    ok &= mm_e()

    if not args.skip_gpu:
        import torch
        from load_aurora import load_aurora
        model, _rep = load_aurora(device="cuda")
        ok &= mm_a(model)
        ok &= mm_b(model)
        ok &= mm_c(model)
        del model
        torch.cuda.empty_cache()

    RESULTS["total_seconds"] = time.time() - t0
    RESULTS["all_ok"] = bool(ok)
    (PREFLIGHT / "preflight_summary_mm.json").write_text(json.dumps(RESULTS, indent=2))
    print(f"[preflight] ALL {'OK' if ok else 'FAILED'} in {RESULTS['total_seconds']:.0f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
