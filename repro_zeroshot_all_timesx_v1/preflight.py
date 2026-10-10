#!/usr/bin/env python3
"""S2/S3 preflight.

--sample-only (local): freeze the comparison sample BEFORE any new inference:
  per domain 2 vars (random.Random('<protocol>|<domain>') over the sorted list
  of vars reusable in ALL three seeds), per var up to 5 evenly spaced
  old-cached LN-test windows (incl. first/last), plus ALL constant-history
  windows (new_only sanity; none of them is LN-test cached).
  -> preflight_sample.json (hash-pinned, bound to inventory + reuse_plan)

--run (GPU host): re-forward every sampled window under all three base seeds
  (per-window seed = frozen rule) and compare with the old cache under the
  frozen dual criteria:
    c1: max|new-old| / max(1e-8, max|old|) <= 1e-5
    c2: max|new-old| / d                    <= 1e-4   (d = frozen denominator)
  plus finiteness. A failing (window, seed) re-infers the WHOLE (var, seed).
  Thresholds are never loosened. Also re-asserts the aurora package md5 equals
  the one recorded in the old shard headers, and that the old derive_seed
  equals the local frozen formula. -> preflight_report.json + reuse_final.json
  (per-(seed,var) verdicts + forward_binding fingerprint).
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np

import common as C
import protocol as P

REPORT_PATH = os.path.join(C.HERE, "preflight_report.json")
FINAL_PATH = C.REUSE_FINAL
SAMPLE_PATH = C.PREFLIGHT_SAMPLE


def windows_hash(windows):
    return hashlib.sha256(
        json.dumps(windows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def candidate_vars_all_seeds(plan):
    out = None
    for s in P.SEEDS:
        vars_s = {vk for vk, v in plan["per_seed"][str(s)]["vars"].items()
                  if v["verdict"] == "reuse_candidate"}
        out = vars_s if out is None else (out & vars_s)
    return out or set()


def cached_sids_by_var(plan):
    """{var_key: {sample_id}} cached per seed via the verified shards."""
    out = {}
    for s in P.SEEDS:
        cand_s = {vk for vk, v in plan["per_seed"][str(s)]["vars"].items()
                  if v["verdict"] == "reuse_candidate"}
        files = sorted({v["file"] for v in plan["per_seed"][str(s)]["vars"].values()
                        if v["verdict"] == "reuse_candidate"})
        m = {}
        for fname in files:
            with np.load(os.path.join(P.OLD_PRED_DIR, fname),
                         allow_pickle=False) as z:
                for i in range(len(z["sample_ids"])):
                    vk, sid = str(z["var_keys"][i]), str(z["sample_ids"][i])
                    if vk in cand_s:
                        m.setdefault(sid, set()).add(vk)
        out[s] = m
    return out


def cached_sid_all_seeds(plan):
    """{var_key: set(sample_id)} cached for EVERY seed in the verified shards
    (a sid only counts when all seeds carry it, so the compare is 3-way)."""
    per_seed = cached_sids_by_var(plan)
    out = {}
    for vk in candidate_vars_all_seeds(plan):
        sets = [{sid for sid, vks in per_seed[s].items() if vk in vks}
                for s in P.SEEDS]
        out[vk] = set.intersection(*sets) if sets else set()
    return out


def local_seed(base_seed, var_key, sample_id):
    token = f"{P.SEED_PREFIX}|{base_seed}|{var_key}|{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % (2 ** 31 - 1)


def build_sample():
    inv = C.load_json(C.INV_PATH)
    plan = C.load_json(C.REUSE_PLAN)
    cand = candidate_vars_all_seeds(plan)
    n_all = len({w["var_key"] for w in inv["windows"]})
    if len(cand) < n_all:
        print(f"[preflight] note: {n_all - len(cand)} vars are not reusable in all "
              "seeds — sampling restricted to all-seed reusable vars")

    zero_ids = {z["sample_id"] for z in inv["meta"]["zero_variance_windows"]}
    cached = cached_sid_all_seeds(plan)
    ln = C.load_json(P.LN_MANIFEST)
    ln_keys = set()
    for vk, v in ln["variables"].items():
        for t in v["native"]["test"]:
            ln_keys.add(t if isinstance(t, str) else t["sample_id"])

    by_domain = {}
    for w in inv["windows"]:
        by_domain.setdefault(w["domain"], {}).setdefault(w["var_key"], []).append(w)

    sample_windows = []
    per_domain = {}
    n_cached_eligible = 0
    for domain in sorted(by_domain):
        vars_sorted = sorted(vk for vk in by_domain[domain] if vk in cand)
        rng = random.Random(f"{P.PROTOCOL_VERSION}|{domain}")
        chosen = rng.sample(vars_sorted, min(P.PF_VARS_PER_DOMAIN, len(vars_sorted)))
        per_domain[domain] = sorted(chosen)
        for vk in sorted(chosen):
            # sample ONLY from windows actually reusable from the old cache
            # (verified shards: LN-test sids present for all 3 seeds)
            rows = sorted((w for w in by_domain[domain][vk]
                           if w["sample_id"] in cached[vk]),
                          key=lambda w: w["flat_index"])
            n_cached_eligible += len(rows)
            if not rows:
                raise SystemExit(f"{vk}: no all-seed cached windows to sample")
            n = len(rows)
            idxs = sorted(set(int(round(i)) for i in
                              np.linspace(0, n - 1, P.PF_WINDOWS_PER_VAR)))
            for i in idxs:
                w = rows[i]
                if w["sample_id"] not in ln_keys:
                    raise SystemExit(f"{w['sample_id']} cached but outside LN test")
                sample_windows.append({"flat_index": w["flat_index"],
                                       "sample_id": w["sample_id"], "var_key": vk,
                                       "domain": domain, "mode": "compare"})
    # all constant-history windows; none is LN-test cached -> new_only sanity
    for w in inv["windows"]:
        if w["sample_id"] in zero_ids:
            sample_windows.append({"flat_index": w["flat_index"],
                                   "sample_id": w["sample_id"], "var_key": w["var_key"],
                                   "domain": w["domain"], "mode": "new_only"})
    sample_windows.sort(key=lambda r: r["flat_index"])
    seen, dedup = set(), []
    for r in sample_windows:
        if r["sample_id"] in seen:
            continue
        seen.add(r["sample_id"])
        dedup.append(r)
    sample = {
        "protocol_version": P.PROTOCOL_VERSION,
        "rule": (f"per domain {P.PF_VARS_PER_DOMAIN} vars via random.Random("
                 f"'<protocol>|<domain>') from sorted all-seed-reusable var list; "
                 f"per var {P.PF_WINDOWS_PER_VAR} evenly spaced windows drawn "
                 "ONLY from windows actually present in the verified old cache "
                 "for all 3 seeds (LN-test sids), incl. first/last; plus ALL "
                 "constant-history windows (new_only; none is LN-test cached); "
                 "every sampled window runs under all 3 base seeds"),
        "inventory_sha256": C.sha256_file(C.INV_PATH),
        "reuse_plan_sha256": C.sha256_file(C.REUSE_PLAN),
        "n_compare": sum(1 for r in dedup if r["mode"] == "compare"),
        "n_new_only": sum(1 for r in dedup if r["mode"] == "new_only"),
        "n_cached_eligible": n_cached_eligible,
        "n_forward_total": len(dedup) * len(P.SEEDS),
        "domains": per_domain,
        "windows": dedup,
    }
    sample["windows_sha256"] = windows_hash(sample["windows"])
    C.dump_json(sample, SAMPLE_PATH)
    print(f"[preflight] sample frozen: {len(dedup)} windows x {len(P.SEEDS)} seeds "
          f"= {sample['n_forward_total']} forwards "
          f"(compare={sample['n_compare']}, new_only={sample['n_new_only']})")
    C.mark_done("s2_sample_frozen", {
        "n_windows": len(dedup), "n_forward_total": sample["n_forward_total"],
        "sha": {"preflight_sample.json": C.sha256_file(SAMPLE_PATH)},
        "input_guards": C.stage_input_guards("sample")})


def load_sample():
    s = C.load_json(SAMPLE_PATH)
    if s["windows_sha256"] != windows_hash(s["windows"]):
        raise SystemExit("preflight sample hash mismatch — sampling list was modified")
    if s["inventory_sha256"] != C.sha256_file(C.INV_PATH):
        raise SystemExit("preflight sample frozen against a different inventory")
    if s["reuse_plan_sha256"] != C.sha256_file(C.REUSE_PLAN):
        raise SystemExit("preflight sample frozen against a different reuse_plan")
    return s


def run_preflight():
    t0 = time.time()
    sample = load_sample()
    plan = C.load_json(C.REUSE_PLAN)
    fp = C.forward_fingerprint()
    if plan["pins"]["aurora_package_md5_expected_on_gpu"] != fp["aurora_package_md5"]:
        raise SystemExit("current aurora package md5 != the one recorded in the old "
                         "shard headers — environment drift, refusing preflight")

    inv = C.load_json(C.INV_PATH)
    win_by_sid = {w["sample_id"]: w for w in inv["windows"]}
    arrays = np.load(C.ARRAYS_PATH)
    past_all, d_all = arrays["past"], arrays["d"]

    # old cached pred rows per (seed, sid), re-verifying shard hashes
    old_pred = {}
    for s in P.SEEDS:
        for vk, v in plan["per_seed"][str(s)]["vars"].items():
            if v["verdict"] != "reuse_candidate":
                continue
            path = os.path.join(P.OLD_PRED_DIR, v["file"])
            if C.sha256_file(path) != v["sha256"]:
                raise SystemExit(f"old shard {v['file']} changed since S1b")
            with np.load(path, allow_pickle=False) as z:
                for i, sid in enumerate(str(x) for x in z["sample_ids"]):
                    if str(z["var_keys"][i]) == vk:
                        old_pred[(s, sid)] = z["pred"][i]

    import torch
    if not torch.cuda.is_available() and "--device" not in sys.argv:
        print("cuda unavailable, falling back to cpu", file=sys.stderr)
    from load_aurora import load_aurora
    model, load_rep = load_aurora(None, None, verify_package=False)

    import predict as OP  # read-only old module (fingerprint-checked at S1b)
    for w in sample["windows"][:20]:
        for s in P.SEEDS:
            if OP.derive_seed(s, w["var_key"], w["sample_id"]) != \
                    local_seed(s, w["var_key"], w["sample_id"]):
                raise SystemExit("old derive_seed != frozen seed rule")

    def forward(sid, base_seed):
        w = win_by_sid[sid]
        return OP.predict_window(model, past_all[w["flat_index"]], P.ITL,
                                 P.NUM_SAMPLES,
                                 OP.derive_seed(base_seed, w["var_key"], sid))

    per_var = {}
    checked = []
    for r in sorted(sample["windows"], key=lambda x: x["flat_index"]):
        sid = r["sample_id"]
        w = win_by_sid[sid]
        for s in P.SEEDS:
            new = np.asarray(forward(sid, s), dtype=np.float64)
            rec = {"flat_index": r["flat_index"], "sample_id": sid,
                   "var_key": r["var_key"], "seed": s, "mode": r["mode"]}
            if not np.isfinite(new).all():
                rec.update(passed=False, reason="new pred non-finite")
            elif r["mode"] == "new_only":
                rec.update(passed=True, reason="new_only sanity (finite)")
            else:
                old = np.asarray(old_pred[(s, sid)], dtype=np.float64)
                diff = np.abs(new - old)
                c1 = float(diff.max() / max(1e-8, float(np.abs(old).max())))
                d = float(d_all[w["flat_index"]])
                c2 = float(diff.max() / d)
                ok = (c1 <= P.TOL_PRED_REL) and (c2 <= P.TOL_PRED_OVER_D) \
                    and np.isfinite(old).all()
                rec.update(passed=bool(ok), c1=c1, c2=c2, d=d,
                           reason=None if ok else f"c1={c1:.3e} c2={c2:.3e}")
            per_var.setdefault((s, r["var_key"]), []).append(rec)
            checked.append(rec)

    fail_keys = {k for k, recs in per_var.items() if any(not x["passed"] for x in recs)}
    c1s = [x["c1"] for x in checked if "c1" in x]
    c2s = [x["c2"] for x in checked if "c2" in x]

    final_vars = {}
    for s in P.SEEDS:
        for vk, v in plan["per_seed"][str(s)]["vars"].items():
            if v["verdict"] != "reuse_candidate":
                final_vars[f"{s}|{vk}"] = {"verdict": "reinfer", "seed": s,
                                           "reason": "S1b verification failed"}
            elif (s, vk) in fail_keys:
                final_vars[f"{s}|{vk}"] = {"verdict": "reinfer", "seed": s,
                                           "reason": "preflight dual-criteria failure"}
            elif any(x["var_key"] == vk for x in checked if x["seed"] == s):
                final_vars[f"{s}|{vk}"] = {"verdict": "reuse", "seed": s,
                                           "reason": None, "n_checked":
                                               sum(1 for x in checked
                                                   if x["seed"] == s and x["var_key"] == vk)}
            else:
                final_vars[f"{s}|{vk}"] = {"verdict": "reuse", "seed": s,
                                           "reason": "not sampled (S1b verified)"}

    per_seed_totals = {}
    for s in P.SEEDS:
        reuse_vars = []
        for key, v in final_vars.items():
            s_str, vk = key.split("|", 1)
            if int(s_str) == s and v["verdict"] == "reuse":
                reuse_vars.append(vk)
        covered = set()
        for vk in reuse_vars:
            v = plan["per_seed"][str(s)]["vars"][vk]
            if v["verdict"] == "reuse_candidate":
                with np.load(os.path.join(P.OLD_PRED_DIR, v["file"]),
                             allow_pickle=False) as z:
                    for i, sid in enumerate(str(x) for x in z["sample_ids"]):
                        if str(z["var_keys"][i]) == vk:
                            covered.add(sid)
        per_seed_totals[str(s)] = {
            "reuse_vars": len(reuse_vars),
            "reuse_windows": len(covered),
            "fill_windows": P.TOTAL_WINDOWS - len(covered),
        }
        print(f"[preflight] seed {s}: reuse_vars={len(reuse_vars)} "
              f"reuse_windows={len(covered)} fill_windows={per_seed_totals[str(s)]['fill_windows']}")

    report = {
        "stage": "S3_preflight",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sample_sha256": C.sha256_file(SAMPLE_PATH),
        "forward_binding": fp,
        "model_load_report": {k: load_rep[k] for k in
                              ("determinism", "weights", "mode", "device")},
        "criteria": {"c1_max_rel": P.TOL_PRED_REL, "c2_max_over_d": P.TOL_PRED_OVER_D},
        "stats": {
            "n_checked": len(checked),
            "n_compare": sum(1 for x in checked if x["mode"] == "compare"),
            "n_new_only": sum(1 for x in checked if x["mode"] == "new_only"),
            "n_failed_(seed,var)": len(fail_keys),
            "c1_max": max(c1s) if c1s else None,
            "c2_max": max(c2s) if c2s else None,
        },
        "failed_detail": [x for x in checked if not x["passed"]],
        "elapsed_s": round(time.time() - t0, 1),
    }
    final = {
        "stage": "S3_preflight_final",
        "created": report["created"],
        "forward_binding": fp,
        "variables": final_vars,
        "totals": {
            "per_seed": per_seed_totals,
            "pred_total": P.TOTAL_WINDOWS * len(P.SEEDS),
            "note": ("fill_windows = 8106 - actually-covered cached windows per "
                     "seed; the LN-test KEY SET stays fixed at 2,474 regardless"),
        },
    }
    C.dump_json(report, REPORT_PATH)
    C.dump_json(final, FINAL_PATH)
    C.mark_done("s3_preflight", {
        "n_checked": len(checked),
        "failed_(seed,var)": len(fail_keys),
        "fill_windows_per_seed": {s: per_seed_totals[s]["fill_windows"]
                                  for s in per_seed_totals},
        "sha": {"preflight_report.json": C.sha256_file(REPORT_PATH),
                "reuse_final.json": C.sha256_file(FINAL_PATH)},
        "input_guards": C.stage_input_guards("preflight")})
    print(f"[preflight] checked={len(checked)} "
          f"(compare={report['stats']['n_compare']}) "
          f"failed_(seed,var)={len(fail_keys)} "
          f"c1_max={report['stats']['c1_max']:.3e} "
          f"c2_max={report['stats']['c2_max']:.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-only", action="store_true")
    ap.add_argument("--run", action="store_true",
                    help="execute the preflight comparison (GPU host)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.sample_only == args.run:
        ap.error("exactly one of --sample-only / --run is required")
    if args.sample_only:
        build_sample()
    else:
        run_preflight()


if __name__ == "__main__":
    main()
