#!/usr/bin/env python3
"""Select the frozen diagnostic windows (analysis/event_compression_v1).

Debug set:  19 windows, 1 train window per domain = domain median by raw
            Events Aurora-token length (trace_D2.jsonl blocks.Events.raw_tokens,
            scope=='train'), lower-median index (n-1)//2, ties by composite key.
Val set:    57 windows, 3 val windows per domain = shortest / median / longest
            by the same ordering, distinct, ties by composite key.

Hard gates: every selected key is in manifest train/val; intersection with
test/excluded key sets is empty (test keys are read for exclusion ONLY).
"""
import json
import sys

from ec_common import OUT, PROJECT, iter_trace_jsonl, manifest_scope_sets


def order_key(r):
    return (r["blocks"]["Events"]["raw_tokens"], r["var_key"], r["sample_id"])


def pick(sorted_rows, idx):
    r = sorted_rows[idx]
    return {"var_key": r["var_key"], "sample_id": r["sample_id"],
            "domain": r["domain"], "scope": r["scope"], "freq": r["freq"],
            "events_raw_tokens": r["blocks"]["Events"]["raw_tokens"],
            "events_alloc_E": r["blocks"]["Events"]["alloc"],
            "events_skipped": r["blocks"]["Events"]["skipped"],
            "calendar_skipped": r["blocks"]["Calendar"]["skipped"],
            "n_events_trace": r["events"]["n_total"],
            "content_tokens_d2": r["content_tokens"]}


def main():
    by_dom = {}
    for r in iter_trace_jsonl(scope_filter={"train", "val"}):
        by_dom.setdefault((r["domain"], r["scope"]), []).append(r)

    trv, test, exc = manifest_scope_sets()
    debug, val = [], []
    domains = sorted({d for d, _s in by_dom})
    assert len(domains) == 19, f"expected 19 domains, got {len(domains)}"
    for d in domains:
        tr = sorted(by_dom[(d, "train")], key=order_key)
        va = sorted(by_dom[(d, "val")], key=order_key)
        assert tr, f"no train windows in domain {d}"
        assert len(va) >= 3, f"domain {d} has <3 val windows"
        debug.append(pick(tr, (len(tr) - 1) // 2))
        idxs = [0, (len(va) - 1) // 2, len(va) - 1]
        assert len(set(idxs)) == 3
        picked = [pick(va, i) for i in idxs]
        keys = {(w["var_key"], w["sample_id"]) for w in picked}
        assert len(keys) == 3, f"domain {d}: val picks not distinct"
        val.extend(picked)

    for w in debug + val:
        k = (w["var_key"], w["sample_id"])
        assert k in trv, f"selected key not in train/val: {k}"
        assert k not in test and k not in exc, f"selected key leaked: {k}"
    assert len(debug) == 19 and len(val) == 57
    all_keys = [(w["var_key"], w["sample_id"]) for w in debug + val]
    assert len(set(all_keys)) == 76, "debug/val sets overlap"

    OUT.mkdir(exist_ok=True)
    out = {
        "method": "event-compression-v1",
        "selection_rule": CONFIG_NOTES,
        "counts": {"debug": len(debug), "val": len(val)},
        "domains": domains,
        "debug_windows": debug,
        "val_windows": val,
    }
    (OUT / "selected_windows.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
    freqs = {}
    for w in debug + val:
        freqs[w["freq"]] = freqs.get(w["freq"], 0) + 1
    print(f"selected: debug={len(debug)} val={len(val)} freq={freqs}")
    print(f"calendar_skipped: {sum(1 for w in debug+val if w['calendar_skipped'])}/76")
    print(f"-> {OUT/'selected_windows.json'}")


CONFIG_NOTES = (
    "debug: per-domain train median of blocks.Events.raw_tokens "
    "(lower-median (n-1)//2, ties by (raw_tokens,var_key,sample_id)); "
    "val: per-domain shortest/median/longest by the same ordering, distinct")

if __name__ == "__main__":
    sys.exit(main())
