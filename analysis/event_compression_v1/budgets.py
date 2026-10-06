#!/usr/bin/env python3
"""S2: rebuild frozen D2 for the 76 diagnostic windows, verify bitwise against
trace_D2.npz, and compute per-event token budgets (analysis/event_compression_v1).

Outputs (gitignored):
  outputs/d2_rebuilt.npz   ids/mask int32 [76,512] + keys + set names
  outputs/budgets_all.json per-window: E, overhead, pool, per-event
                           need/base/budget/fit/zero, cleaned event prose
"""
import json
import sys

import numpy as np

from ec_common import (OUT, compute_event_budgets, event_pieces,
                       load_tokenizer, rebuild_d2, split_events_clean,
                       trace_npz_index, verify_bitwise_vs_trace)


def main():
    from text_cache import build_index

    tok = load_tokenizer()
    z, idx = trace_npz_index()
    sel = json.loads((OUT / "selected_windows.json").read_text())
    index = build_index()

    rows_ids, rows_mask, keys, set_names = [], [], [], []
    budgets = {}
    for set_name in ("debug", "val"):
        for w in sel[f"{set_name}_windows"]:
            vk, sid = w["var_key"], w["sample_id"]
            fields = index[(vk, sid)]["fields"]
            rb = rebuild_d2(fields, tok)

            assert rb["alloc"]["Events"] == w["events_alloc_E"], \
                f"{vk}|{sid}: recomputed Events alloc != trace E"
            assert verify_bitwise_vs_trace(rb["ids"], rb["mask"], vk, sid,
                                           z, idx), \
                f"{vk}|{sid}: D2 rebuild not bitwise vs trace_D2.npz"

            if rb["counts"]["Events"]["skipped"]:
                b = {"status": "events_absent"}
            else:
                split = split_events_clean(rb["cleaned"]["scenario"], tok)
                if split is None or not split["ok"]:
                    b = {"status": "parse_fail"}
                else:
                    head_ids, tag_ids, prose_ids = event_pieces(split, tok)
                    whole = rb["counts"]["Events"]["raw_ids"]
                    concat = ([i for i in head_ids]
                              + [t for pair in zip(tag_ids, prose_ids)
                                 for part in pair for t in part])
                    if concat != whole:
                        div = next((i for i, (x, y) in
                                    enumerate(zip(concat, whole)) if x != y),
                                   min(len(concat), len(whole)))
                        raise AssertionError(
                            f"{vk}|{sid}: event pieces != block ids; "
                            f"len(concat)={len(concat)} len(whole)={len(whole)} "
                            f"first_div={div} n_tags={len(tag_ids)} "
                            f"n_proses={len(prose_ids)} "
                            f"n_seg={len(split['segments'])}")
                    b = compute_event_budgets(split, head_ids, tag_ids,
                                              prose_ids, w["events_alloc_E"])
                    if b["status"] == "ok":
                        b["events"] = [
                            {"k": k + 1,
                             "tag": split["events"][k][0],
                             "prose_clean": split["events"][k][1],
                             "need": b["need"][k], "base": b["base"][k],
                             "budget": b["budget"][k], "fit": b["fit"][k],
                             "zero_budget": b["zero_budget"][k]}
                            for k in range(b["n"])]
                        b["head"] = split["head"]
            b.update({"var_key": vk, "sample_id": sid, "domain": w["domain"],
                      "scope": w["scope"], "freq": w["freq"],
                      "E": w["events_alloc_E"],
                      "events_raw_tokens": w["events_raw_tokens"],
                      "calendar_skipped": w["calendar_skipped"]})
            budgets.setdefault(set_name, []).append(b)
            rows_ids.append(rb["ids"])
            rows_mask.append(rb["mask"])
            keys.append((vk, sid))
            set_names.append(set_name)
            print(f"[{set_name}] {vk}|{sid}: E={b['E']} status={b['status']} "
                  f"n={b.get('n', 0)} fit={sum(b.get('fit', []))}", flush=True)

    ids_arr = np.asarray(rows_ids, dtype=np.int32)
    mask_arr = np.asarray(rows_mask, dtype=np.int32)
    np.savez(OUT / "d2_rebuilt.npz", ids=ids_arr, mask=mask_arr,
             var_keys=np.array([k[0] for k in keys]),
             sample_ids=np.array([k[1] for k in keys]),
             set_name=np.array(set_names))

    stats = {}
    for set_name, bs in budgets.items():
        st = {"n": len(bs)}
        for s in ("ok", "events_absent", "parse_fail", "overhead_overflow"):
            st[s] = sum(1 for b in bs if b["status"] == s)
        ok = [b for b in bs if b["status"] == "ok"]
        st["events_total"] = sum(b["n"] for b in ok)
        st["events_fit_verbatim"] = sum(sum(b["fit"]) for b in ok)
        st["events_zero_budget"] = sum(sum(b["zero_budget"]) for b in ok)
        stats[set_name] = st
    out = {"method": "event-compression-v1", "stats": stats, "windows": budgets}
    (OUT / "budgets_all.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("STATS:", json.dumps(stats))
    print("BITWISE GATE: all 76 windows passed vs trace_D2.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
