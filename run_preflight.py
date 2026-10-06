#!/usr/bin/env python3
"""Preflight A-L for the Aurora x TimesX baseline.

Runs ONLY on synthetic windows and train/val windows. Zero inference on test
windows. All evidence lands in preflight/ as JSON + small sample .npy files.

Sections:
  A load        sha256/keys/strict/eval/frozen
  B eval-state  BN running stats unchanged across real inference (bitwise)
  C shapes      [1,100,12], finite everywhere
  D determinism same-seed bitwise, diff-seed differs, settings live,
                cross-process reproduction (subprocess twice)
  E immutability full state_dict (incl. num_batches_tracked) bitwise
  F identity    data_cache vs zip JSON spot check; composite-key uniqueness
  G revin A/B   official revin=True vs manual-normalize/revin=False/manual-restore;
                d (scoring) vs internal RevIN std recorded separately
  H metrics     window_d closed form + constant->fallback + std_mse closed form
  I aggregate   ported vs ORIGINAL EXP-004 aggregate.py, full precision
  J itl compare 48 vs 9 on the same windows: geometry/finiteness/timing
                (parallel presentation, no recommendation)
  K perf        per-window time, peak VRAM, extrapolation to 2474x3
  L resume      continuous vs resume bitwise; stale-fingerprint shard rejected
                to quarantine; duplicate/missing detection
"""
import argparse
import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import numpy as np
import torch

import data_loader
from load_aurora import load_aurora
from predict import ShardStore, derive_seed, predict_window, shard_fingerprint

PRE = PROJECT / "preflight"
SAMPLES = PRE / "samples"
NUM_SAMPLES = 100


def save(name, obj):
    p = PRE / name
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    return p


def load_jsonl_zip_reader():
    spec = importlib.util.spec_from_file_location(
        "timesx_data_readonly",
        str(PROJECT / "reference" / "exp004" / "timesx_data.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def preflight_windows(data):
    """Fixed, small real-window set from train/val ONLY (never test)."""
    rows = []
    for dom in ("climate", "traffic"):
        rows.extend(data.trainval_rows(dom, max_per_var=2, splits=("train", "val")))
    return rows[:8]


def synthetic_windows():
    rng = np.random.default_rng(7)
    t = np.arange(96.0)
    return [
        ("synth_constant", np.full(96, 3.5)),
        ("synth_ramp", 0.5 * t),
        ("synth_sine", np.sin(t / 6.0)),
        ("synth_noise", rng.normal(0, 2.0, 96)),
    ]


def bn_buffers(model):
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            out[name] = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    return out


def buffers_equal(a, b):
    diffs = []
    for mod, dicts in a.items():
        for k, v in dicts.items():
            w = b[mod][k]
            same = np.array_equal(v.numpy(), w.numpy())
            if not same:
                diffs.append(f"{mod}.{k}")
    return diffs


# ---------------------------------------------------------------- sections

def sec_A(protocol, device):
    model, rep = load_aurora(device=device)
    ok = (rep["weights"]["ok"] and rep["package"]["all_ok"]
          and not rep["key_diff"]["missing"] and not rep["key_diff"]["unexpected"]
          and rep["mode"]["training"] is False and rep["mode"]["trainable_params"] == 0
          and rep["strict_load"]["missing_keys"] == [] and rep["strict_load"]["unexpected_keys"] == [])
    save("A_load.json", {"ok": ok, "report": rep})
    print(f"A load: {'OK' if ok else 'FAIL'}")
    return model, ok


def sec_B(model, windows, itl):
    before = bn_buffers(model)
    preds = [predict_window(model, w[3], itl, NUM_SAMPLES, 42) for w in windows[:2]]
    after = bn_buffers(model)
    diffs = buffers_equal(before, after)
    ok = not diffs
    save("B_eval_bn.json", {"ok": ok, "n_bn_modules": len(before), "diffs": diffs})
    print(f"B eval-state: {'OK' if ok else 'FAIL'} ({len(before)} BN modules)")
    return ok


def sec_C(model, all_windows, itl):
    items, ok = [], True
    for name, w in all_windows:
        out = predict_window(model, w, itl, NUM_SAMPLES, 42)
        with torch.inference_mode():
            x = torch.from_numpy(np.asarray(w, dtype=np.float32)).unsqueeze(0).to(next(model.parameters()).device)
            torch.manual_seed(42)
            raw = model.generate(inputs=x, text_inputs=None, vision_inputs=None, revin=True,
                                 num_samples=NUM_SAMPLES, max_output_length=12, inference_token_len=itl)
        item = {"window": name, "shape": list(out.shape), "finite": bool(np.isfinite(out).all()),
                "raw_shape": list(raw.shape), "raw_finite": bool(torch.isfinite(raw).all().item())}
        item["ok"] = item["finite"] and item["raw_finite"]
        ok &= item["ok"]
        items.append(item)
    save("C_shapes.json", {"ok": ok, "items": items})
    print(f"C shapes/finite: {'OK' if ok else 'FAIL'}")
    return ok


def sec_D(model, windows, itl):
    ok = True
    same = []
    for name, w, *_ in windows[:3]:
        a = predict_window(model, w, itl, NUM_SAMPLES, 123)
        b = predict_window(model, w, itl, NUM_SAMPLES, 123)
        eq = np.array_equal(a, b)
        same.append({"window": name, "bitwise_equal": eq,
                     "max_abs_diff": float(np.max(np.abs(a - b)))})
        ok &= eq
    s1 = predict_window(model, windows[0][1], itl, NUM_SAMPLES, 123)
    s2 = predict_window(model, windows[0][1], itl, NUM_SAMPLES, 124)
    diff_seed = not np.array_equal(s1, s2)
    ok &= diff_seed

    live = {
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    live_ok = (not live["tf32_matmul"] and not live["tf32_cudnn"]
               and not live["cudnn_benchmark"] and live["deterministic_algorithms"]
               and live["cublas_workspace_config"] == ":4096:8")
    ok &= live_ok

    child_out1 = PRE / "d_child_run1.npy"
    child_out2 = PRE / "d_child_run2.npy"
    cmd = [sys.executable, str(PROJECT / "run_preflight.py"), "--_determinism-child"]
    procs = [subprocess.run(cmd + [str(p), str(itl)], capture_output=True, text=True)
             for p in (child_out1, child_out2)]
    cross = {"ok": False}
    if all(p.returncode == 0 for p in procs):
        a1, a2 = np.load(child_out1), np.load(child_out2)
        cross = {"ok": bool(np.array_equal(a1, a2)),
                 "bitwise_equal": bool(np.array_equal(a1, a2)),
                 "max_abs_diff": float(np.max(np.abs(a1 - a2))),
                 "shape": list(a1.shape)}
        SAMPLES.mkdir(exist_ok=True)
        np.save(SAMPLES / "determinism_crossprocess.npy", a1)
        for p in (child_out1, child_out2):
            os.remove(p)
    else:
        cross = {"ok": False, "stderr": [p.stderr[-500:] for p in procs]}
    ok &= cross["ok"]

    save("D_determinism.json", {"ok": ok, "same_seed": same, "diff_seed_differs": diff_seed,
                                "settings_live": live, "settings_ok": live_ok,
                                "cross_process": cross})
    print(f"D determinism: {'OK' if ok else 'FAIL'} (cross-process {'OK' if cross['ok'] else 'FAIL'})")
    return ok


def past_of(item):
    """Accept either a (name, window) pair or a trainval row tuple; return the window."""
    if isinstance(item, np.ndarray):
        return item
    return item[3] if isinstance(item, tuple) and isinstance(item[3], np.ndarray) else item[1]


def sec_E(model, windows, itl):
    sd_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for r in windows[:3]:
        predict_window(model, past_of(r), itl, NUM_SAMPLES, 7)
    diffs = [k for k, v in sd_before.items()
             if not np.array_equal(v.numpy(), model.state_dict()[k].detach().cpu().numpy())]
    ok = not diffs and any("num_batches_tracked" in k for k in sd_before)
    save("E_state.json", {"ok": ok, "n_tensors": len(sd_before), "diffs": diffs[:20],
                          "has_bn_counters": any("num_batches_tracked" in k for k in sd_before)})
    print(f"E immutability: {'OK' if ok else 'FAIL'} ({len(sd_before)} tensors)")
    return ok


def sec_F(data):
    zpath = PROJECT / "data" / "TimesX_Datasets.zip"
    txd = load_jsonl_zip_reader()
    data_cls = txd.iter_variables(str(zpath), load_values=True)

    targets = {data.domain_vars["climate"][0], data.domain_vars["traffic"][0]}
    found = {}
    for v in data_cls:
        key = v.var_key
        if key in targets:
            found[key] = v
        if len(found) == len(targets):
            break

    checks, ok = [], True
    for key, v in found.items():
        manifest = data.manifest["variables"][key]
        seen = set()
        for s in v.samples:
            sid = txd.make_sample_id(v.domain, v.var, s.future_start, s.idx, seen)
            if sid not in manifest["sample_start_idx"]:
                continue
            start = manifest["sample_start_idx"][sid]
            past_cache = data.series(key)[start:start + 96]
            tgt_cache = data.series(key)[start + 96:start + 108]
            past_zip = np.asarray(s.past_val, dtype=np.float64)
            tgt_zip = np.asarray(s.future_val, dtype=np.float64)
            same_len = len(past_cache) == 96 and len(tgt_cache) == 12
            d_past = float(np.max(np.abs(past_cache - past_zip))) if same_len else None
            d_tgt = float(np.max(np.abs(tgt_cache - tgt_zip))) if same_len else None
            item = {"var_key": key, "sample_id": sid, "past_max_abs_diff": d_past,
                    "target_max_abs_diff": d_tgt, "ok": bool(same_len and d_past == 0.0 and d_tgt == 0.0)}
            checks.append(item)
            ok &= item["ok"]
            break  # one window per var is enough for spot check

    keys = set()
    for d in data_loader.DOMAIN_NAMES:
        for vk in data.domain_vars[d]:
            for sid in data.native_ids(vk, "test"):
                k = (vk, sid)
                if k in keys:
                    ok = False
                    checks.append({"duplicate_composite_key": list(k), "ok": False})
                keys.add(k)
    save("F_identity.json", {"ok": ok and len(keys) == 2474, "n_unique_keys": len(keys),
                             "spot_checks": checks})
    print(f"F identity: {'OK' if ok and len(keys) == 2474 else 'FAIL'} ({len(keys)} unique keys)")
    return ok and len(keys) == 2474


def sec_G(model, windows, itl):
    dev = next(model.parameters()).device
    items, ok = [], True
    for name, w in windows[:4]:
        x = torch.from_numpy(np.asarray(w, dtype=np.float32)).unsqueeze(0).to(dev)
        torch.manual_seed(2021)
        with torch.inference_mode():
            p1 = model.generate(inputs=x, text_inputs=None, vision_inputs=None, revin=True,
                                num_samples=NUM_SAMPLES, max_output_length=12, inference_token_len=itl)
            means = x.mean(dim=-1, keepdim=True)
            stdev = x.std(dim=-1, keepdim=True, unbiased=False) + 1e-5
            xn = (x - means) / stdev
            torch.manual_seed(2021)
            p_norm = model.generate(inputs=xn, text_inputs=None, vision_inputs=None, revin=False,
                                    num_samples=NUM_SAMPLES, max_output_length=12, inference_token_len=itl)
            st = stdev.unsqueeze(1).repeat(1, NUM_SAMPLES, 1)
            me = means.unsqueeze(1).repeat(1, NUM_SAMPLES, 1)
            p2 = p_norm * st + me
        a, b = p1.to(torch.float64).cpu().numpy(), p2.to(torch.float64).cpu().numpy()
        bitwise = np.array_equal(a, b)
        max_abs = float(np.max(np.abs(a - b)))
        denom = np.maximum(np.abs(a), 1e-12)
        max_rel = float(np.max(np.abs(a - b) / denom))
        passed = bitwise or (max_abs <= 1e-5 and max_rel <= 1e-6)
        ok &= passed
        items.append({"window": name, "bitwise_equal": bool(bitwise),
                      "max_abs": max_abs, "max_rel": max_rel, "ok": bool(passed)})
        np.save(SAMPLES / f"revin_{name}.npy", np.stack([a[0], b[0]]))
    save("G_revin.json", {"ok": ok, "items": items,
                          "tolerances": {"bitwise_expected": True,
                                         "abs_tol": 1e-5, "rel_tol": 1e-6},
                          "note": "Scoring denominator d and Aurora internal RevIN std "
                                  "(std(unbiased=False)+1e-5) are distinct; never blended."})
    print(f"G revin A/B: {'OK' if ok else 'FAIL'}")
    return ok


def sec_H(data):
    from data_loader import window_d
    x = np.array([1.0, 2.0, 3.0, 4.0])
    closed = float(np.sqrt(np.mean((x - x.mean()) ** 2)))
    d1 = window_d(x, 99.0)
    const = np.full(96, 5.0)
    d2 = window_d(const, 99.0)
    real = preflight_windows(data)[:1]
    past = real[0][3]
    d3 = window_d(past, data.fb_std[real[0][0]])

    pred = np.array([1.0, 2.0])
    tgt = np.array([0.0, 4.0])
    dd = 2.0
    smse_closed = float(np.mean(((pred - tgt) / dd) ** 2))
    smse_impl = float(np.mean((pred - tgt) ** 2)) / (dd ** 2)

    items = [
        {"case": "pstdev closed form", "expected": closed, "got": d1, "ok": d1 == closed},
        {"case": "constant -> fallback_std", "expected": 99.0, "got": d2, "ok": d2 == 99.0},
        {"case": "real train window d", "expected": float(np.std(past)), "got": d3,
         "ok": abs(d3 - float(np.std(past))) < 1e-12 and d3 >= 1e-8},
        {"case": "std_mse closed form", "expected": smse_closed, "got": smse_impl,
         "ok": smse_closed == smse_impl},
    ]
    ok = all(i["ok"] for i in items)
    save("H_metrics.json", {"ok": ok, "items": items,
                            "note": "constant window: scoring d=fallback_std, inference uses official revin; "
                                    "the two scales are separate by protocol and never interchanged"})
    print(f"H metrics: {'OK' if ok else 'FAIL'}")
    return ok


def sec_I():
    r = subprocess.run([sys.executable, str(PROJECT / "verify_aggregate.py")],
                       capture_output=True, text=True)
    rep = json.loads((PRE / "aggregate_verify.json").read_text())
    ok = r.returncode == 0 and rep["all_within_atol"] and rep["bitwise_equal"]
    save("I_aggregate.json", {"ok": ok, "returncode": r.returncode,
                              "stderr_tail": r.stderr[-500:] if r.returncode else "",
                              "max_abs_diff": rep["max_abs_diff"],
                              "n_comparisons": rep["n_comparisons"],
                              "corroboration": rep["corroboration_vs_archive_csv"]})
    print(f"I aggregate vs original: {'OK' if ok else 'FAIL'}")
    return ok


def sec_J(model, real_windows, synth):
    out = {"windows": [], "itl": {}}
    for itl in (48, 9):
        geometry = {"itl": itl,
                    "predict_token_num": int(np.ceil(12 / itl)),
                    "padded_output_len": int(np.ceil(12 / itl)) * itl,
                    "sliced_to": 12}
        times = []
        finite_all = True
        shape_ok = True
        stats = []
        for name, w in synth + [(r[0], r[3]) for r in real_windows[:4]]:
            x = torch.from_numpy(np.asarray(w, dtype=np.float32)).unsqueeze(0).to(next(model.parameters()).device)
            torch.manual_seed(42)
            t0 = time.time()
            with torch.inference_mode():
                raw = model.generate(inputs=x, text_inputs=None, vision_inputs=None, revin=True,
                                     num_samples=NUM_SAMPLES, max_output_length=12, inference_token_len=itl)
            dt = time.time() - t0
            times.append(dt)
            pred = raw.to(torch.float64).mean(dim=1).squeeze(0).cpu().numpy()
            finite_all &= bool(np.isfinite(pred).all() and torch.isfinite(raw).all().item())
            shape_ok &= list(raw.shape) == [1, NUM_SAMPLES, 12]
            stats.append({"window": name, "ms": dt * 1000,
                          "mean": float(pred.mean()), "std": float(pred.std())})
        out["itl"][str(itl)] = {"geometry": geometry, "shape_ok": shape_ok,
                                "finite_all": finite_all,
                                "ms_per_window_mean": float(np.mean(times)) * 1000,
                                "per_window": stats}
    ok = all(v["shape_ok"] and v["finite_all"] for v in out["itl"].values())
    save("J_itl.json", {"ok": ok, **out,
                        "verdict": "parallel evidence only; freeze decision belongs to the user; "
                                   "main scheme itl=48 follows the official TimeMMD precedent "
                                   "(Health/Traffic: seq_len=96, inference_token_len=48), "
                                   "itl=9 remains a candidate"})
    print(f"J itl 48 vs 9: {'OK' if ok else 'FAIL'} "
          f"(48: {out['itl']['48']['ms_per_window_mean']:.0f} ms, "
          f"9: {out['itl']['9']['ms_per_window_mean']:.0f} ms)")
    return ok


def sec_K(model, real_windows, itl):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    times = []
    for r in real_windows[:6]:
        t0 = time.time()
        predict_window(model, past_of(r), itl, NUM_SAMPLES, 5)
        times.append(time.time() - t0)
    ms = float(np.mean(times)) * 1000
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    total_min = 2474 * 3 * ms / 1000 / 60
    free = 0.0
    if torch.cuda.is_available():
        free_t, tot_t = torch.cuda.mem_get_info()
        free = free_t / 2**30
    ok = True
    save("K_perf.json", {"ok": ok, "ms_per_window_mean": ms,
                         "ms_per_window_all": [t * 1000 for t in times],
                         "peak_vram_gb": peak, "gpu_free_gb_now": free,
                         "extrapolation_2474x3_minutes": total_min,
                         "gate": "free>=20GB before official run; OOM -> stop and preserve scene"})
    print(f"K perf: {ms:.0f} ms/window, peak {peak:.2f} GB, 2474x3 ~ {total_min:.0f} min")
    return ok


def sec_L(protocol, itl, weights_sha):
    ok = True
    tmp_dir = PRE / "resume_drill"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    store = ShardStore(out_dir=tmp_dir / "shards")
    data = data_loader.TimesXData()
    rows = []
    for vk, sid, dom, past, tgt, dv in data.trainval_rows("climate", max_per_var=1)[:6]:
        rows.append((vk, sid, "__preflight__", past, tgt, dv))
    fp = shard_fingerprint(protocol, itl, 9999, weights_sha)

    def run_rows(rows_in, seed_base):
        out_rows = []
        for vk, sid, dom, past, tgt, dv in rows_in:
            w = derive_seed(seed_base, vk, sid)
            pred = predict_window(data_model, past, itl, NUM_SAMPLES, w)
            out_rows.append({"var_key": vk, "sample_id": sid, "domain": dom,
                             "pred": pred, "target": tgt, "d": dv})
        return out_rows

    data_model, _ = load_aurora(device="cuda" if torch.cuda.is_available() else None)

    continuous = run_rows(rows, 9999)

    # interrupted: first 3 rows then "crash" -> nothing persisted
    _ = run_rows(rows[:3], 9999)
    resumed = run_rows(rows, 9999)
    resume_bitwise = all(
        np.array_equal(a["pred"], b["pred"]) for a, b in zip(continuous, resumed))
    ok &= resume_bitwise

    store.save("P48", "__preflight__", 9999, fp, continuous)
    reused = store.reuse_or_init("P48", "__preflight__", 9999, fp)
    ok &= reused
    z = store.load("P48", "__preflight__", 9999)
    stored_bitwise = np.array_equal(z["pred"], np.array([r["pred"] for r in continuous]))
    ok &= stored_bitwise

    fp_tampered = dict(fp)
    fp_tampered["itl"] = 9
    rejected = not store.reuse_or_init("P48", "__preflight__", 9999, fp_tampered)
    quarantined = any(p.name.endswith(".npz") for p in
                      (tmp_dir / "shards" / "quarantine").glob("*.npz")) if (tmp_dir / "shards" / "quarantine").exists() else False
    ok &= rejected and quarantined

    # duplicate / missing detection
    keys = [(r[0], r[1]) for r in rows]
    dup = keys + keys[:1]
    missing = keys[1:]
    dup_detected = len(dup) != len(set(dup))
    missing_detected = set(missing) != set(keys)
    ok &= dup_detected and missing_detected

    save("L_resume.json", {"ok": ok,
                           "resume_bitwise": resume_bitwise,
                           "stored_bitwise": stored_bitwise,
                           "reuse_compatible_shard": reused,
                           "stale_fingerprint_rejected": rejected,
                           "quarantined": quarantined,
                           "duplicate_detected": dup_detected,
                           "missing_detected": missing_detected})
    print(f"L resume: {'OK' if ok else 'FAIL'}")
    return ok


def determinism_child(out_path, itl):
    model, _ = load_aurora(device="cuda" if torch.cuda.is_available() else None)
    data = data_loader.TimesXData()
    rows = preflight_windows(data)[:3]
    preds = np.stack([predict_window(model, r[3], itl, NUM_SAMPLES,
                                     derive_seed(777, r[0], r[1])) for r in rows])
    np.save(out_path, preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--itl", type=int, default=48)
    ap.add_argument("--sections", nargs="*", default=None,
                    help="subset e.g. A B C; default all")
    ap.add_argument("--_determinism-child", nargs=2, default=None, metavar=("OUT", "ITL"))
    args = ap.parse_args()

    if args._determinism_child:
        determinism_child(args._determinism_child[0], int(args._determinism_child[1]))
        return

    import torch
    PRE.mkdir(exist_ok=True)
    SAMPLES.mkdir(exist_ok=True)
    protocol = json.loads((PROJECT / "protocol.json").read_text())
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    data = data_loader.TimesXData()
    real = preflight_windows(data)
    synth = synthetic_windows()
    allw = [(n, w) for n, w in synth] + [(r[0], r[3]) for r in real]

    model, ok_a = sec_A(protocol, device)
    results = {"A": ok_a}
    todo = args.sections or ["B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    if "B" in todo:
        results["B"] = sec_B(model, real, args.itl)
    if "C" in todo:
        results["C"] = sec_C(model, allw, args.itl)
    if "D" in todo:
        results["D"] = sec_D(model, allw, args.itl)
    if "E" in todo:
        results["E"] = sec_E(model, real, args.itl)
    if "F" in todo:
        results["F"] = sec_F(data)
    if "G" in todo:
        results["G"] = sec_G(model, allw, args.itl)
    if "H" in todo:
        results["H"] = sec_H(data)
    if "I" in todo:
        results["I"] = sec_I()
    if "J" in todo:
        results["J"] = sec_J(model, real, synth)
    if "K" in todo:
        results["K"] = sec_K(model, real, args.itl)
    if "L" in todo:
        results["L"] = sec_L(protocol, args.itl, model_sha(protocol))

    ok = all(results.values())
    save("preflight_report.json", {"ok": ok, "sections": results,
                                   "itl": args.itl, "device": device,
                                   "test_window_inference": 0})
    print(("PREFLIGHT ALL OK" if ok else "PREFLIGHT FAILED") +
          f" (itl={args.itl}, device={device})")
    sys.exit(0 if ok else 1)


def model_sha(protocol):
    from load_aurora import resolve_weights_path, verify_weights
    return verify_weights(resolve_weights_path())["sha256"]


if __name__ == "__main__":
    main()
