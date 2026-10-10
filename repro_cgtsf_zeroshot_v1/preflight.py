#!/usr/bin/env python3
"""Stage P: 230-window x 3-seed preflight on the executing GPU host.

Hard gates (any failure = nonzero exit, no full run):
  1. protocol/code/old-root/aurora-package/CSV/visionts_ref fingerprints
     (protocol.verify_frozen) + resolved weights sha256 before/after
  2. frozen-params gate: all requires_grad False, model.training False
  3. internal-behavior table: pad = (48 - L%48)%48 == {24,0,0},
     predict_token_num = ceil(P/48) == {1,1,2} (formula + frozen literal)
  4. output shape (P,) and finiteness on all 230 windows x 3 seeds
     (batch=1; includes the all-zero-history windows)
  5. same-env/same-order repeat inference (seed 2021) bitwise identical
  6. cross-seed sensitivity: 2021/2022/2023 predictions differ (RNG active)
  7. in-memory model state digest (params+buffers) unchanged per (ds, seed)
  8. shard roundtrip (atomic save npz -> reload) bitwise identical
  9. real resume drill on 32 PTF windows (seed 2021) through the production
     run_infer code paths (sandboxed predictions/quarantine dirs):
     continuous run vs fresh-process recompute bitwise identical;
     tampered stored guards -> HARD STOP (tier-1); corrupt npz ->
     quarantine + recompute bitwise identical (tier-2).
Record-only (NOT gates): explicit zero-prepend changes the prediction
(revin on the unpadded input), per-window timing / VRAM / full-run ETA.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for p in (REPO_ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

import common as C
import infer_aurora as IA
import load_aurora as LA
import protocol as P
from load_cgtsf import load_windows, select_preflight


def setup_determinism():
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == P.CUBLAS_WORKSPACE_CONFIG, \
        "CUBLAS_WORKSPACE_CONFIG not set (use wrapper_theta.sh)"
    LA.apply_determinism()
    torch.manual_seed(min(P.SEEDS))
    np.random.seed(min(P.SEEDS))


def frozen_gate(model):
    tg = [n for n, p in model.named_parameters() if p.requires_grad]
    return (not tg) and (model.training is False), tg


RESUME_RUNNER = r'''
import os, sys
HERE = r"{here}"
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)
import numpy as np
import common as C
import protocol as P
import run_infer as R
from load_cgtsf import load_windows

P.PRED_DIR = r"{pred_dir}"
P.QUARANTINE_DIR = r"{quarantine_dir}"
npz_check, _, _ = P.shard_paths("PTF", 2021, 0)
assert npz_check.startswith(r"{pred_dir}"), \
    f"sandbox leak: shard would land at {{npz_check}}"

R.setup_determinism()
import infer_aurora as IA
model, _ = IA.build_model(r"{weights}", "cuda")
windows = load_windows("PTF")[:32]
preds, ms = R.compute_shard(model, windows, "cuda", 2021, 24)
guards = C.current_guards(batch=1, preflight_report_sha256=None)
R.write_shard("PTF", 2021, 0, windows, preds, ms, guards)
print("RESUME_RUNNER_OK")
'''


def resume_test(model, weights_path, device):
    """Real interrupt->restart->resume drill on 32 PTF windows (seed 2021)
    through the production run_infer code paths, in a sandboxed predictions
    dir. Nothing here touches the real predictions/ or quarantine/ trees."""
    import run_infer as R
    import tempfile
    res = {"n_windows": 32, "seed": 2021}
    ok = True
    saved = (P.PRED_DIR, P.QUARANTINE_DIR)
    try:
        with tempfile.TemporaryDirectory(prefix="cgtsf_resume_") as td:
            P.PRED_DIR = os.path.join(td, "pred")
            P.QUARANTINE_DIR = os.path.join(td, "quarantine")
            Pl = P.SUBSETS["PTF"]["pred"]
            dg_a = C.model_state_digest(model)
            windows = load_windows("PTF")[:32]

            # 1) continuous run: in-process model computes and writes shard 0
            preds_a, _ = R.compute_shard(model, windows, device, 2021, Pl)
            guards = C.current_guards(batch=1, preflight_report_sha256=None)
            R.write_shard("PTF", 2021, 0, windows, preds_a,
                          np.zeros(len(windows)), guards)
            v, _ = R.shard_valid("PTF", 2021, 0, guards)
            res["baseline_shard_valid"] = bool(v)
            ok &= v

            # 2) simulate a crash: wipe the shard, resume in a fresh process
            for p in P.shard_paths("PTF", 2021, 0):
                if os.path.isfile(p):
                    os.remove(p)
            v2, _ = R.shard_valid("PTF", 2021, 0, guards)
            res["after_crash_invalid"] = not v2
            ok &= not v2
            runner = os.path.join(td, "resume_runner.py")
            with open(runner, "w") as f:
                f.write(RESUME_RUNNER.format(
                    here=HERE, weights=weights_path,
                    pred_dir=P.PRED_DIR,
                    quarantine_dir=P.QUARANTINE_DIR))
            rp = subprocess.run([sys.executable, runner], cwd=HERE,
                                capture_output=True, text=True, timeout=1800)
            res["resume_proc_returncode"] = rp.returncode
            if rp.returncode != 0:
                res["resume_proc_tail"] = (rp.stderr or rp.stdout)[-2000:]
                ok = False
            else:
                npz, _, _ = P.shard_paths("PTF", 2021, 0)
                with np.load(npz) as z:
                    preds_b = z["pred"]
                same = (preds_b.shape == preds_a.shape
                        and np.array_equal(preds_b, preds_a))
                res["resume_bitwise_ok"] = bool(same)
                ok &= same
                v3, _ = R.shard_valid("PTF", 2021, 0, guards)
                res["cross_process_guards_ok"] = bool(v3)
                ok &= bool(v3)

            # 3) tier-1: tampered stored guards must HARD STOP
            npz, hdr, done = P.shard_paths("PTF", 2021, 0)
            good_done = open(done).read()
            d = json.loads(good_done)
            d["guards"]["core"]["weights_sha256"] = "0" * 64
            C.dump_json(d, done)
            stopped = False
            try:
                R.shard_valid("PTF", 2021, 0, guards)
            except SystemExit as e:
                stopped = "HARD STOP" in str(e)
            res["tier1_hardstop_ok"] = bool(stopped)
            ok &= stopped
            with open(done, "w") as f:
                f.write(good_done)

            # 4) tier-2: corrupt npz -> detect -> quarantine -> recompute
            with open(npz, "r+b") as f:
                f.seek(100)
                f.write(b"\x00\x01\x02\x03")
            v4, why4 = R.shard_valid("PTF", 2021, 0, guards)
            res["tier2_detected"] = bool(not v4 and "payload" in why4)
            ok &= res["tier2_detected"]
            if res["tier2_detected"]:
                R.quarantine("PTF", 2021, 0, why4)
                qfiles = sorted(os.listdir(P.QUARANTINE_DIR))
                res["tier2_quarantined"] = qfiles
                q_ok = (any(x.startswith("PTF__s2021__shard000.npz.")
                            for x in qfiles)
                        and any(x.endswith(".reason.json") for x in qfiles)
                        and not os.path.isfile(npz))
                res["tier2_quarantine_ok"] = bool(q_ok)
                ok &= q_ok
                preds_c, _ = R.compute_shard(model, windows, device, 2021, Pl)
                R.write_shard("PTF", 2021, 0, windows, preds_c,
                              np.zeros(len(windows)), guards)
                v5, _ = R.shard_valid("PTF", 2021, 0, guards)
                with np.load(npz) as z:
                    same_c = np.array_equal(z["pred"], preds_a)
                res["tier2_recompute_bitwise_ok"] = bool(v5 and same_c)
                ok &= res["tier2_recompute_bitwise_ok"]

            dg_b = C.model_state_digest(model)
            res["model_state_unchanged"] = bool(dg_a == dg_b)
            ok &= res["model_state_unchanged"]
    finally:
        P.PRED_DIR, P.QUARANTINE_DIR = saved
    res["ok"] = bool(ok)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-approve-frozen-protocol", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", default=None,
                    help="explicit weights path (default: old-root HF-cache "
                         "resolution via load_aurora)")
    args = ap.parse_args()

    proto = P.check_approval(args.i_approve_frozen_protocol)
    P.verify_frozen(proto)
    setup_determinism()
    device = args.device
    assert device == "cuda" and torch.cuda.is_available()

    w_path = IA.resolve_weights(args.weights)
    w_sha_0 = C.sha256_file(w_path)
    assert w_sha_0 == P.WEIGHTS_SHA256
    print(f"[preflight] weights ok ({w_path}): {w_sha_0[:16]}...")

    model, load_report = IA.build_model(w_path, device)
    fz_ok, tg = frozen_gate(model)
    print(f"[preflight] frozen gate (params requires_grad all False, "
          f"eval mode) = {fz_ok}")
    if not fz_ok:
        sys.exit(f"HARD STOP: trainable params present: {tg[:10]}")

    windows_by_ds = {ds: load_windows(ds) for ds in P.ORDER}
    sel = select_preflight(windows_by_ds)
    per_ds = {ds: picks for ds, picks in sel}
    print("[preflight] selected windows:",
          {ds: len(p) for ds, p in per_ds.items()}, "total",
          sum(len(p) for p in per_ds.values()))

    report = {"stage": "preflight", "env": C.env_info(),
              "weights_path": w_path,
              "weights_sha256": w_sha_0,
              "load_report_weights_sha256":
                  load_report.get("weights", {}).get("sha256"),
              "per_subset": {}, "checks": {}}
    ok = True
    preds_by = {}   # (ds, seed) -> np.array [n, Pl]

    for ds in P.ORDER:
        cfg = P.SUBSETS[ds]
        pad_exp = IA.expected_pad(cfg["context"])
        tok_exp = IA.predict_token_num(cfg["pred"])
        table_ok = (pad_exp == P.EXPECTED_PAD[ds] == P.EXPECTED_PAD_LITERAL[ds]
                    and tok_exp == P.PREDICT_TOKEN_NUM[ds]
                    == P.PREDICT_TOKEN_NUM_LITERAL[ds])
        print(f"[preflight] {ds}: pad={pad_exp} predict_token_num={tok_exp} "
              f"ok={table_ok}")
        ok &= table_ok
        picks = per_ds[ds]
        hists = [windows_by_ds[ds][i]["hist"] for i in picks]
        Pl = cfg["pred"]
        n_zero = sum(1 for i in picks if not windows_by_ds[ds][i]["hist"].any())
        entry = {"n_windows": len(picks), "n_zero_hist_windows": n_zero,
                 "pad_expected": pad_exp, "predict_token_num": tok_exp,
                 "behavior_table_ok": bool(table_ok), "seeds": {}}
        dg_start = C.model_state_digest(model)

        # --- run 1: all seeds, batch=1, timed (seed 2021 timed in detail) ---
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t1 = []
        for seed in P.SEEDS:
            out = []
            for j, h in enumerate(hists):
                t0 = time.perf_counter()
                a = IA.predict_window(model, h, Pl,
                                      C.derive_seed(seed, ds,
                                                    windows_by_ds[ds][picks[j]]["idx"],
                                                    windows_by_ds[ds][picks[j]]["date"]))
                torch.cuda.synchronize()
                if seed == min(P.SEEDS):
                    t1.append(time.perf_counter() - t0)
                assert a.shape == (Pl,), f"{ds}: pred shape {a.shape} != ({Pl},)"
                if not C.all_finite(a):
                    print(f"[preflight] FAIL {ds} s{seed}: non-finite output")
                    ok = False
                out.append(a)
            preds_by[(ds, seed)] = np.stack(out)
        vram = torch.cuda.max_memory_allocated() / (1 << 20)

        # --- run 2: identical repeat (base seed), bitwise ---
        out2 = []
        for j, h in enumerate(hists):
            out2.append(IA.predict_window(
                model, h, Pl,
                C.derive_seed(min(P.SEEDS), ds,
                              windows_by_ds[ds][picks[j]]["idx"],
                              windows_by_ds[ds][picks[j]]["date"])))
        rep_ok = np.array_equal(preds_by[(ds, min(P.SEEDS))], np.stack(out2))
        print(f"[preflight] {ds}: repeat bitwise identical = {rep_ok}")
        ok &= rep_ok

        # --- cross-seed sensitivity gate ---
        s = list(P.SEEDS)
        cs_ok = (not np.array_equal(preds_by[(ds, s[0])], preds_by[(ds, s[1])])
                 and not np.array_equal(preds_by[(ds, s[0])], preds_by[(ds, s[2])])
                 and not np.array_equal(preds_by[(ds, s[1])], preds_by[(ds, s[2])]))
        print(f"[preflight] {ds}: cross-seed predictions differ = {cs_ok}")
        ok &= cs_ok

        # --- model state unchanged across the whole subset's forwards ---
        dg_end = C.model_state_digest(model)
        entry["seeds"] = {"start": dg_start, "end": dg_end}
        entry["state_unchanged"] = bool(dg_start == dg_end)
        print(f"[preflight] {ds}: model state unchanged = "
              f"{entry['state_unchanged']}")
        ok &= entry["state_unchanged"]

        # --- record-only: explicit zero-prepend changes the prediction ---
        demo = None
        if ds == "PTF":
            j = 0
            h = hists[j]
            seed_d = min(P.SEEDS)
            sd = C.derive_seed(seed_d, ds, windows_by_ds[ds][picks[j]]["idx"],
                               windows_by_ds[ds][picks[j]]["date"])
            hp = np.concatenate([np.zeros(pad_exp, dtype=np.float64), h])
            a0 = IA.predict_window(model, h, Pl, sd)
            a1 = IA.predict_window(model, hp, Pl, sd)
            demo = {"note": "revin stats are computed on the UNPADDED input; "
                            "explicit zero-prepend is NOT equivalent (record "
                            "only, never used in the experiment)",
                    "explicit_pad_equal": bool(np.array_equal(a0, a1))}
            entry["zero_prepend_demo"] = demo
            print(f"[preflight] {ds}: zero-prepend demo "
                  f"(expected not equal) = {demo['explicit_pad_equal']}")

        entry["ms_per_window_b1_mean"] = float(np.mean(t1)) * 1000.0
        entry["ms_per_window_b1_std"] = float(np.std(t1)) * 1000.0
        entry["vram_peak_mib"] = float(vram)
        entry["repeat_bitwise_ok"] = bool(rep_ok)
        entry["cross_seed_ok"] = bool(cs_ok)
        report["per_subset"][ds] = entry

    # --- roundtrip: atomic shard-style save/reload for first 8 PTF preds ---
    import tempfile
    arr = preds_by[("PTF", min(P.SEEDS))][:8]
    with tempfile.TemporaryDirectory() as td:
        pth = os.path.join(td, "rt.npz")
        np.savez(pth + ".tmp.npz", pred=arr)
        os.replace(pth + ".tmp.npz", pth)
        with np.load(pth) as z:
            rt_ok = np.array_equal(z["pred"], arr)
    print(f"[preflight] roundtrip save/reload bitwise identical = {rt_ok}")
    ok &= rt_ok
    report["checks"]["roundtrip_bitwise_ok"] = bool(rt_ok)

    # --- real resume drill (production run_infer paths, sandboxed dirs) ---
    print("[preflight] resume drill: 32 PTF windows s2021, fresh-process "
          "recompute + tier-1/tier-2 shard anomaly paths ...", flush=True)
    resume = resume_test(model, w_path, device)
    report["checks"]["resume"] = resume
    ok &= resume["ok"]
    print(f"[preflight] resume drill ok = {resume['ok']} "
          f"(bitwise={resume.get('resume_bitwise_ok')} "
          f"tier1={resume.get('tier1_hardstop_ok')} "
          f"tier2={resume.get('tier2_recompute_bitwise_ok')})", flush=True)

    w_sha_1 = C.sha256_file(w_path)
    w_ok = w_sha_0 == w_sha_1 == P.WEIGHTS_SHA256
    print(f"[preflight] weights file unchanged = {w_ok}")
    ok &= w_ok
    report["checks"]["weights_unchanged"] = bool(w_ok)
    report["ok"] = bool(ok)

    mean_ms = float(np.mean([report["per_subset"][ds]["ms_per_window_b1_mean"]
                             * report["per_subset"][ds]["n_windows"]
                             for ds in P.ORDER]))
    report["full_run_eta_hours"] = P.TOTAL_FORWARDS * mean_ms / 1000 / 3600
    print(f"[preflight] ETA for {P.TOTAL_FORWARDS} forwards: "
          f"{report['full_run_eta_hours']:.2f} h")

    os.makedirs(P.RESULTS_DIR, exist_ok=True)
    C.dump_json(report, os.path.join(P.RESULTS_DIR, "preflight_report.json"))
    print(f"[preflight] RESULT: {'PASS' if ok else 'FAIL'} "
          f"-> results/preflight_report.json")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
