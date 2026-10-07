#!/usr/bin/env python3
"""S5 QC: independent recomputation of coverage/failure stats from the frozen
artifacts (final_inputs npz + budgets_all.json + attempts jsonl + run_meta).

  python qc.py --set debug
  python qc.py --set val

Coverage is ALWAYS computed against the final actual input ids: per event,
its token ids must appear contiguously inside the final Events block.
Fallback windows are counted as D2 coverage (their unused LLM pieces are NOT
counted as scheme coverage). Fact-preservation regex screens are screening
only — they never claim zero hallucination.
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict

import numpy as np

from ec_common import HERE, OUT, RESULTS, load_tokenizer, trace_npz_index, \
    verify_bitwise_vs_trace

REFUSAL_RE = re.compile(
    r"\b(i cannot|i can't|cannot assist|no relevant|not applicable|"
    r"unable to|as an ai|i'm sorry|i am sorry)\b", re.IGNORECASE)

NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
MONTH_RE = re.compile(r"\b(january|february|march|april|may|june|july|"
                      r"august|september|october|november|december|"
                      r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b")
UNIT_RE = re.compile(r"%(?:\s|)|\b(?:mw|mwh|kwh|gw|gwh|twh|kw|v|kv|a|ma|"
                     r"barrels?|bbl|tons?|tonnes?|mt|kt|bcf|mcf|"
                     r"mcm|hm3|°c|°f|c|f|usd|usd/|eur|\$|€|£|"
                     r"billion|million|thousand|hectares?|acres?|"
                     r"percent)\b", re.IGNORECASE)
NEG_RE = re.compile(r"\b(?:not|no|cannot|can't|never|nor|without|"
                    r"cancel(?:l)?ed|cancel(?:l)?s|halted|halts|halt|"
                    r"suspended|suspends|suspend|postponed|postpones|"
                    r"postpone|delayed|delays|delay|shut|stopped|stops|"
                    r"stop|no longer|unavailable|offline|failed)\b",
                    re.IGNORECASE)
FUT_RE = re.compile(r"\b(?:expect(?:ed|s)?|plan(?:ned|s|ning)?|may|might|"
                    r"could|will|would|forecast(?:ed|s)?|project(?:ed|s)?|"
                    r"likely|aim(?:s|ed)?|seek(?:s)?|intend(?:s|ed)?|"
                    r"anticipat(?:e|es|ed)|outlook|guidance|estimat(?:e|es|"
                    r"ed)|scheduled|due to)\b", re.IGNORECASE)


def is_subseq_ids(needle, hay):
    """Contiguous subsequence check on token id lists."""
    return find_slice(needle, hay) is not None


def find_slice(needle, hay):
    """Start index of the first contiguous occurrence of needle in hay,
    or None."""
    n, h = len(needle), len(hay)
    if n == 0:
        return 0
    first = needle[0]
    for i in range(h - n + 1):
        if hay[i] == first and hay[i:i + n] == needle:
            return i
    return None


def screen_facts(src, outp):
    """Regex screens: preserved fraction per class + new-number flags."""
    s, o = src.lower(), outp.lower()
    res = {}
    for name, rx in (("numbers", NUM_RE), ("years", YEAR_RE),
                     ("months", MONTH_RE), ("units", UNIT_RE),
                     ("negation", NEG_RE), ("forecast", FUT_RE)):
        src_set = set(m.group(0).strip().lower() for m in rx.finditer(s))
        out_set = set(m.group(0).strip().lower() for m in rx.finditer(o))
        preserved = (len(src_set & out_set) / len(src_set)) if src_set \
            else None
        res[name] = {"src": sorted(src_set), "out": sorted(out_set),
                     "preserved": preserved,
                     "new": sorted(out_set - src_set)}
    return res


def slice_add(dst, **dims):
    key = tuple(sorted(dims.items()))
    return dst.setdefault(key, Counter())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_name", choices=("debug", "val"),
                    required=True)
    args = ap.parse_args()
    tag = args.set_name

    tok = load_tokenizer()
    z, idx = trace_npz_index()
    enc = lambda s: tok.encode(s, add_special_tokens=False)

    budgets = json.loads((OUT / "budgets_all.json").read_text())["windows"][tag]
    recs = [json.loads(l) for l in
            open(OUT / f"attempts_{tag}.jsonl")]
    npz = dict(np.load(OUT / f"final_inputs_{tag}.npz", allow_pickle=False))
    meta = json.loads((OUT / f"run_meta_{tag}.json").read_text())

    by_win = defaultdict(dict)
    for r in recs:
        by_win[(r["var_key"], r["sample_id"])][r["scheme"]] = r
    b_by_win = {(b["var_key"], b["sample_id"]): b for b in budgets}

    n_rows = len(npz["method"])
    assert n_rows == 3 * len(budgets), (n_rows, len(budgets))

    win_rows, ev_rows = [], []
    gate_total = gate_pass = 0
    fallback_windows = Counter()
    fact_flags = []

    for wi in range(len(budgets)):
        vk = str(npz["var_keys"][wi * 3])
        sid = str(npz["sample_ids"][wi * 3])
        b = b_by_win[(vk, sid)]
        recs3 = by_win[(vk, sid)]
        assert set(recs3) == {"E-Extract", "E-Summary"}

        base = {"var_key": vk, "sample_id": sid, "domain": b["domain"],
                "scope": b["scope"], "freq": b["freq"],
                "calendar_skipped": b["calendar_skipped"],
                "n_events": b.get("n", 0), "E": b["E"],
                "raw_events_tokens": b["events_raw_tokens"]}

        # D2 row: bitwise vs trace
        row = wi * 3
        ids_d2 = [int(x) for x in npz["ids"][row]]
        mask_d2 = [int(x) for x in npz["mask"][row]]
        assert verify_bitwise_vs_trace(ids_d2, mask_d2, vk, sid, z, idx), \
            f"{vk}|{sid}: D2 row != trace"

        # D2 per-event coverage (baseline) is computed inside the loop below
        # -- per scheme
        for si, scheme in enumerate(("D2", "E-Extract", "E-Summary")):
            row = wi * 3 + si
            ids = [int(x) for x in npz["ids"][row]]
            mask = [int(x) for x in npz["mask"][row]]
            assert len(ids) == 512, f"{vk}|{sid}|{scheme}: len != 512"
            assert ids[0] == tok.cls_token_id and ids[sum(mask) - 1] == \
                tok.sep_token_id, f"{vk}|{sid}|{scheme}: CLS/SEP"
            r = {"scheme": scheme, "outcome": "d2_reference",
                 "events_tokens": None, "covered_events": None}
            if scheme == "D2":
                cov, per = d2_coverage(b, ids, mask, enc)
                r.update({"outcome": "d2_reference",
                          "covered_events": cov,
                          "coverage_rate": cov / b["n"] if b.get("n") else
                          None})
                for k, covk in enumerate(per):
                    ev_rows.append(dict(
                        base, scheme=scheme, k=k + 1,
                        source="d2", covered=covk,
                        tokens=None, budget=b["events"][k]["budget"]))
                win_rows.append(dict(base, **{k2: v for k2, v in r.items()
                                              if k2 != "scheme"},
                                     scheme=scheme))
                continue
            rec = recs3[scheme]
            if rec["outcome"] == "compressed":
                gate_total += 1
                gate_pass += 1
                ev_ids = reconstruct_events_ids(b, rec, enc)
                # independent splice check: D2 row content with its Events
                # region replaced by the reconstructed Events ids must equal
                # the scheme row content exactly (verifies non-Events blocks
                # bitwise + Events content + placement in one shot)
                d2_ids = [int(x) for x in npz["ids"][wi * 3]]
                d2_mask = [int(x) for x in npz["mask"][wi * 3]]
                d2_content = d2_ids[1:sum(d2_mask) - 1]
                head_ids = enc("Events: " + b["head"])
                s = find_slice(head_ids, d2_content)
                assert s is not None, \
                    f"{vk}: Events head not found in D2 row"
                # D2 assemble truncates the Events block to alloc E, so the
                # in-row region length is min(raw, E)
                raw_len = min(b["events_raw_tokens"], b["E"])
                expected = (d2_content[:s] + ev_ids
                            + d2_content[s + raw_len:])
                content = ids[1:sum(mask) - 1]
                assert content == expected, \
                    f"{vk}|{sid}|{scheme}: content != D2-with-Events-spliced"
                assert len(ev_ids) <= b["E"]
                per, facts = scheme_coverage(b, rec, ev_ids, enc)
                cov = sum(1 for e in rec["events"]
                          if e.get("outcome") == "ok"
                          or e.get("source") == "fit_verbatim")
                nonempty = sum(per)
                comp = (1 - len(ev_ids) / b["events_raw_tokens"]
                        ) if b["events_raw_tokens"] else None
                r.update({"outcome": "compressed",
                          "covered_events": cov,
                          "coverage_rate": cov / b["n"] if b.get("n") else
                          None,
                          "events_tokens": len(ev_ids),
                          "nonempty_fragments": nonempty,
                          "compression_rate": comp})
                for k, (covk, fk) in enumerate(zip(per, facts)):
                    ev_rows.append(dict(
                        base, scheme=scheme, k=k + 1,
                        source=rec["events"][k].get("source"),
                        covered=covk, nonempty=fk["nonempty"],
                        tokens=rec["events"][k].get("tokens"),
                        budget=rec["events"][k]["budget"],
                        preserved_numbers=fk["preserved_numbers"],
                        new_numbers=fk["new_numbers"],
                        preserved_units=fk["preserved_units"],
                        preserved_negation=fk["preserved_negation"],
                        preserved_forecast=fk["preserved_forecast"],
                        evidence_ok=rec["events"][k].get("evidence_ok")))
                if any(f["new_numbers"] for f in facts):
                    fact_flags.append({"var_key": vk, "sample_id": sid,
                                       "scheme": scheme, "type": "new_number",
                                       "events": [
                                           {"k": k + 1,
                                            "prose": b["events"][k]
                                            ["prose_clean"],
                                            "piece": rec["events"][k]
                                            .get("piece"),
                                            "new": f["new"]}
                                           for k, f in enumerate(facts)
                                           if f["new_numbers"]]})
            else:
                fallback_windows[scheme] += 1
                cov, per = d2_coverage(b, ids, mask, enc)
                r.update({"outcome": "fallback_d2", "reason": rec["reason"],
                          "covered_events": cov,
                          "coverage_rate": cov / b["n"] if b.get("n") else
                          None})
                for k, covk in enumerate(per):
                    ev_rows.append(dict(
                        base, scheme=scheme, k=k + 1, source="d2_fallback",
                        covered=covk, tokens=None,
                        budget=b["events"][k]["budget"]))
            win_rows.append(dict(base, **{k2: v for k2, v in r.items()
                                          if k2 != "scheme"},
                                 scheme=scheme))

    # failure accounting
    fail_events = Counter()
    corr_attempts = 0
    for r in recs:
        for e in r["events"]:
            if e.get("outcome") == "failed":
                why = (e.get("attempts") or [{}])[-1].get("why", "unknown")
                fail_events[str(why).split("(")[0]] += 1
            for a in e.get("attempts") or []:
                if isinstance(a.get("attempt"), int) and a["attempt"] > 0:
                    corr_attempts += 1

    stats = {
        "set": tag,
        "n_windows": len(budgets),
        "outcomes": meta["outcomes"],
        "assembly_gates": {"compressed_windows": gate_total,
                           "gate_pass": gate_pass,
                           "pass_rate": (gate_pass / gate_total)
                           if gate_total else None},
        "fallback_windows": dict(fallback_windows),
        "events_total": sum(b.get("n", 0) for b in budgets),
        "fail_events": dict(fail_events),
        "correction_attempts": corr_attempts,
        "run_meta": meta,
        "fact_flag_cases": fact_flags,
    }
    RESULTS.mkdir(exist_ok=True)

    # aggregates by scheme
    agg = {}
    for scheme in ("D2", "E-Extract", "E-Summary"):
        rows = [w for w in win_rows if w["scheme"] == scheme]
        ok_rows = [w for w in rows if w["outcome"] == "compressed"]
        cov = [w["coverage_rate"] for w in rows
               if w.get("coverage_rate") is not None]
        evs = [e for e in ev_rows if e["scheme"] == scheme]
        a = {"windows": len(rows),
             "compressed_windows": len(ok_rows),
             "coverage_mean": sum(cov) / len(cov) if cov else None,
             "coverage_event_weighted": (
                 sum(e["covered"] for e in evs) / len(evs)) if evs else None,
             "events": len(evs)}
        if scheme != "D2":
            a["nonempty_fragments"] = sum(
                e.get("nonempty", 0) for e in evs)
            a["over_budget_events"] = sum(
                1 for e in evs if e.get("tokens") is not None
                and e["tokens"] > e["budget"])
            a["evidence_ok_events"] = sum(
                1 for e in evs if e.get("evidence_ok") is True)
            a["new_number_events"] = sum(
                1 for e in evs if e.get("new_numbers"))
            pres = [e["preserved_numbers"] for e in evs
                    if e.get("preserved_numbers") is not None]
            a["preserved_numbers_mean"] = (sum(pres) / len(pres)) if pres \
                else None
        agg[scheme] = a
    stats["aggregate"] = agg

    # slices: freq / calendar_skipped
    slices = {}
    for dim in ("freq", "calendar_skipped", "domain"):
        s = {}
        for scheme in ("E-Extract", "E-Summary"):
            keys = sorted(set((w[dim] if dim != "calendar_skipped"
                               else bool(w[dim]))
                              for w in win_rows))
            entry = {}
            for kv in keys:
                sub = [w for w in win_rows if w["scheme"] == scheme and
                       (w[dim] if dim != "calendar_skipped"
                        else bool(w[dim])) == kv]
                subc = [w for w in sub if w["outcome"] == "compressed"]
                cov = [w["coverage_rate"] for w in sub
                       if w.get("coverage_rate") is not None]
                entry[str(kv)] = {
                    "windows": len(sub), "compressed": len(subc),
                    "coverage_mean": (sum(cov) / len(cov)) if cov else None}
            s[scheme] = entry
        slices[dim] = s
    stats["slices"] = slices

    (RESULTS / f"stats_{tag}.json").write_text(json.dumps(
        stats, indent=1, ensure_ascii=False))

    # CSVs
    import csv
    with open(RESULTS / f"qc_windows_{tag}.csv", "w", newline="") as f:
        cols = ["var_key", "sample_id", "domain", "scope", "freq",
                "calendar_skipped", "n_events", "E", "raw_events_tokens",
                "scheme", "outcome", "reason", "events_tokens",
                "covered_events", "coverage_rate", "nonempty_fragments",
                "compression_rate"]
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(win_rows)
    with open(RESULTS / f"qc_events_{tag}.csv", "w", newline="") as f:
        cols = ["var_key", "sample_id", "domain", "freq", "scheme", "k",
                "source", "covered", "nonempty", "tokens", "budget",
                "preserved_numbers", "new_numbers", "preserved_units",
                "preserved_negation", "preserved_forecast", "evidence_ok"]
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(ev_rows)

    print(f"[qc:{tag}] windows={stats['n_windows']} "
          f"events={stats['events_total']} "
          f"gates={stats['assembly_gates']['pass_rate']} "
          f"fallbacks={dict(fallback_windows)}")
    for scheme, a in agg.items():
        print(f"  {scheme}: cov_mean={a['coverage_mean']} "
              f"ev_weighted={a['coverage_event_weighted']} "
              f"compressed={a['compressed_windows']}/{a['windows']}"
              + (f" nonempty={a['nonempty_fragments']} "
                 f"new_num_ev={a['new_number_events']}"
                 if scheme != "D2" else ""))
    return 0


def d2_coverage(b, ids, mask, enc):
    """Per-event D2 coverage: prose ids contiguous inside the final D2
    Events region. For fallback rows this is exactly the D2 coverage."""
    if b["status"] != "ok" or not b.get("n"):
        return 0, []
    # events region = ids between CLS..SEP excluding other blocks: use the
    # known D2 layout — Events is 2nd block; safer: find via head prefix.
    head_ids = enc("Events: " + b["head"])
    seq = ids[1:sum(mask) - 1]  # strip CLS/SEP
    start = None
    for i in range(len(seq) - len(head_ids) + 1):
        if seq[i:i + len(head_ids)] == head_ids:
            start = i
            break
    if start is None:
        return 0, [False] * b["n"]
    per = []
    for e in b["events"]:
        pid = enc(" " + e["prose_clean"].strip()) if e["prose_clean"] else []
        per.append(is_subseq_ids(pid, seq) if pid else True)
    return sum(per), per


def reconstruct_events_ids(b, rec, enc):
    """Rebuild the Events block ids from stored entries (independent of the
    driver's in-memory state)."""
    parts = []
    for k, e in enumerate(rec["events"]):
        tag = b["events"][k]["tag"]
        if e.get("source") == "fit_verbatim":
            piece = b["events"][k]["prose_clean"]
        else:
            piece = " " + e["piece"]
        parts.append(tag + piece)
    return enc("Events: " + b["head"] + "".join(parts))


def scheme_coverage(b, rec, ev_ids, enc):
    """Per-event: (covered, nonempty) + fact screens on the final pieces."""
    per, facts = [], []
    for k, e in enumerate(rec["events"]):
        if e.get("source") == "fit_verbatim":
            piece = b["events"][k]["prose_clean"]
            pid = enc(piece)
            per.append(is_subseq_ids(pid, ev_ids))
            facts.append({"nonempty": len(pid) >= 3,
                          "preserved_numbers": 1.0, "new_numbers": False,
                          "preserved_units": 1.0, "preserved_negation": 1.0,
                          "preserved_forecast": 1.0, "new": {"numbers": []}})
            continue
        piece = " " + e["piece"]
        pid = enc(piece)
        covered = is_subseq_ids(pid, ev_ids)
        nonempty = len(pid) >= 3 and bool(re.search(r"[a-z0-9]", piece.lower())
                                         ) and not REFUSAL_RE.search(piece)
        f = screen_facts(b["events"][k]["prose_clean"], piece)
        facts.append({"nonempty": bool(covered and nonempty),
                      "preserved_numbers": f["numbers"]["preserved"],
                      "new_numbers": bool(f["numbers"]["new"]),
                      "preserved_units": f["units"]["preserved"],
                      "preserved_negation": f["negation"]["preserved"],
                      "preserved_forecast": f["forecast"]["preserved"],
                      "new": f["numbers"]["new"]})
        per.append(covered)
    return per, facts


if __name__ == "__main__":
    sys.exit(main())
