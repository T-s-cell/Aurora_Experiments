#!/usr/bin/env python3
"""E-Extract v3.3 three-seed prediction experiment: per-event budgets for the
frozen test split (2,474 windows).

Reuses the diagnostic pipeline read-only (analysis/event_compression_v1 via
sys.path): rebuild_d2 + bitwise gate vs trace_D2.npz, then the frozen
compute_event_budgets rule with E = trace blocks.Events.alloc.

Outputs (gitignored):
  outputs/budgets_test.jsonl   one record per test window, sorted composite
                               key order; record shape matches what
                               compress.process_window expects (status/E/
                               events[n]{k,need,base,budget,fit,zero_budget}/
                               head + identity fields)
  outputs/d2_rows_test.npz     rebuilt D2 ids/mask int32 [2474,512] + keys
                               (bitwise == trace rows; reference for every
                               later stage)

Hard gates: 2474 keys == manifest native.test; every rebuild bitwise-equal to
trace_D2.npz; events piece-concat == Events block ids.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SUBDIR = Path(__file__).resolve().parent
PROJECT = SUBDIR.parent
for p in (str(PROJECT), str(PROJECT / "analysis" / "event_compression_v1")):
    if p not in sys.path:
        sys.path.insert(0, p)

from ec_common import (compute_event_budgets, event_pieces, load_tokenizer,  # noqa: E402
                       manifest_scope_sets, rebuild_d2, split_events_clean,
                       trace_npz_index, verify_bitwise_vs_trace)

OUT = SUBDIR / "outputs"


def main():
    from text_cache import build_index

    OUT.mkdir(exist_ok=True)
    tok = load_tokenizer()
    z, idx = trace_npz_index()
    _trv, test_keys, _exc = manifest_scope_sets()
    assert len(test_keys) == 2474, f"manifest test keys {len(test_keys)} != 2474"
    sel = sorted(test_keys)

    # trace jsonl records for identity + E
    rec_by_key = {}
    with open(PROJECT / "analysis" / "text_budget_v1" / "outputs" /
              "trace_D2.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rec_by_key[(r["var_key"], r["sample_id"])] = r

    index = build_index()
    rows_ids, rows_mask = [], []
    n_ok = n_absent = n_parse = n_overhead = 0
    events_total = fit_total = zero_total = 0
    with open(OUT / "budgets_test.jsonl", "w", encoding="utf-8") as f:
        for i, key in enumerate(sel):
            vk, sid = key
            tr = rec_by_key[key]
            assert tr["scope"] == "test", f"{key}: trace scope {tr['scope']}"
            fields = index[key]["fields"]
            rb = rebuild_d2(fields, tok)
            assert verify_bitwise_vs_trace(rb["ids"], rb["mask"], vk, sid, z, idx), \
                f"{key}: D2 rebuild not bitwise vs trace_D2.npz"
            E = tr["blocks"]["Events"]["alloc"]

            if rb["counts"]["Events"]["skipped"]:
                b = {"status": "events_absent"}
            else:
                split = split_events_clean(rb["cleaned"]["scenario"], tok)
                if split is None or not split["ok"]:
                    b = {"status": "parse_fail"}
                else:
                    head_ids, tag_ids, prose_ids = event_pieces(split, tok)
                    whole = rb["counts"]["Events"]["raw_ids"]
                    concat = ([t for t in head_ids]
                              + [x for pair in zip(tag_ids, prose_ids)
                                 for part in pair for x in part])
                    assert concat == whole, \
                        f"{key}: event pieces != Events block ids"
                    b = compute_event_budgets(split, head_ids, tag_ids,
                                              prose_ids, E)
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

            b.update({"var_key": vk, "sample_id": sid,
                      "domain": tr["domain"], "scope": "test",
                      "freq": tr["freq"], "E": E,
                      "events_raw_tokens": tr["blocks"]["Events"]["raw_tokens"],
                      "calendar_skipped": tr["blocks"]["Calendar"]["skipped"]})
            f.write(json.dumps(b, ensure_ascii=False) + "\n")

            if b["status"] == "ok":
                n_ok += 1
                events_total += b["n"]
                fit_total += sum(b["fit"])
                zero_total += sum(b["zero_budget"])
            elif b["status"] == "events_absent":
                n_absent += 1
            elif b["status"] == "parse_fail":
                n_parse += 1
            else:
                n_overhead += 1
            rows_ids.append(rb["ids"])
            rows_mask.append(rb["mask"])
            if (i + 1) % 200 == 0:
                print(f"[budgets] {i + 1}/2474", flush=True)

    ids_arr = np.asarray(rows_ids, dtype=np.int32)
    mask_arr = np.asarray(rows_mask, dtype=np.int32)
    npz = OUT / "d2_rows_test.npz"
    tmp = Path(str(npz) + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, ids=ids_arr, mask=mask_arr,
                 var_keys=np.array([k[0] for k in sel], dtype=np.str_),
                 sample_ids=np.array([k[1] for k in sel], dtype=np.str_))
    tmp.replace(npz)

    stats = {"n_windows": len(sel), "ok": n_ok, "events_absent": n_absent,
             "parse_fail": n_parse, "overhead_overflow": n_overhead,
             "events_total": events_total, "events_fit_verbatim": fit_total,
             "events_zero_budget": zero_total}
    rep = {"stats": stats,
           "budgets_test_jsonl_sha256": hashlib.sha256(
               (OUT / "budgets_test.jsonl").read_bytes()).hexdigest(),
           "d2_rows_test_npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
           "bitwise_gate": "all 2474 windows passed vs trace_D2.npz"}
    (OUT / "budgets_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(stats, indent=1))
    print("[budgets] BITWISE GATE: all 2474 windows passed vs trace_D2.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
