#!/usr/bin/env python3
"""Statistics for analysis-text-budget-v1 (reads outputs/trace_{D0,D1,D2}.jsonl).

Scope grid: scheme x {full 8106, test 2474} x {overall, 19 domains}.
Boundary statement (report-level): coverage = entered BERT input; token
reduction != information loss; higher coverage does not imply better forecast.
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
RESULTS = HERE / "results"
SCHEMES = ("D0", "D1", "D2")
FIELDS = ("Background", "Events", "Calendar", "Covariates")
EXPECTED_REF = {"events_raw_median_test": 1313, "idle_median_test": 56,
                "events_alone_over510_test": 2464}


def load(scheme):
    recs = []
    with open(OUT / f"trace_{scheme}.jsonl", encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def p90(xs):
    return float(np.percentile(np.asarray(xs, dtype=float), 90)) if xs else None


def med(xs):
    return float(np.median(np.asarray(xs, dtype=float))) if xs else None


def field_rows(recs, scheme, scope):
    """Per-field token stats; D2 adds cleaned_tokens."""
    per = {f: {"raw": [], "alloc": [], "cleaned": []} for f in FIELDS}
    for r in recs:
        for f in FIELDS:
            b = r["blocks"][f]
            per[f]["raw"].append(b["raw_tokens"])
            per[f]["alloc"].append(b["alloc"])
            if scheme == "D2":
                per[f]["cleaned"].append(b.get("cleaned_tokens", b["raw_tokens"]))
    rows = []
    for f in FIELDS:
        d = per[f]
        row = {"scheme": scheme, "scope": scope, "field": f,
               "raw_median": med(d["raw"]), "raw_p90": p90(d["raw"]),
               "raw_max": max(d["raw"]), "raw_sum": sum(d["raw"]),
               "alloc_median": med(d["alloc"]), "alloc_p90": p90(d["alloc"]),
               "alloc_max": max(d["alloc"]), "alloc_sum": sum(d["alloc"])}
        row["kept_frac_by_sum"] = (row["alloc_sum"] / row["raw_sum"]
                                   if row["raw_sum"] else None)
        if scheme == "D2":
            row["cleaned_median"] = med(d["cleaned"])
            row["cleaned_p90"] = p90(d["cleaned"])
            row["cleaned_max"] = max(d["cleaned"])
            row["cleaned_sum"] = sum(d["cleaned"])
            row["shortening_by_sum"] = (1 - row["cleaned_sum"] / row["raw_sum"]
                                        if row["raw_sum"] else None)
            row["retention_by_sum"] = (row["alloc_sum"] / row["cleaned_sum"]
                                       if row["cleaned_sum"] else None)
        rows.append(row)
    return rows


def d2_ratios(recs, scope):
    """Per-window ratios (medians), separate from by-sum ratios: shortening =
    1 - cleaned/raw per field; retention = alloc/cleaned of the D2 final input."""
    per = {f: {"short": [], "ret": []} for f in FIELDS}
    for r in recs:
        for f in FIELDS:
            b = r["blocks"][f]
            if b["raw_tokens"] > 0:
                per[f]["short"].append(
                    1 - b.get("cleaned_tokens", b["raw_tokens"]) / b["raw_tokens"])
            if b.get("cleaned_tokens", 0) > 0:
                per[f]["ret"].append(b["alloc"] / b["cleaned_tokens"])
    return [{"scope": scope, "field": f,
             "shortening_median_w": med(per[f]["short"]),
             "retention_median_w": med(per[f]["ret"])} for f in FIELDS]


def agg_rows(scheme, recs, scope, domain):
    content = [r["content_tokens"] for r in recs]
    idle = [r["idle_tokens"] for r in recs]
    ev = {k: sum(r["events"][k] for r in recs)
          for k in ("n_total", "n_complete", "n_partial", "n_absent")}
    cv_ent = sum(r["covariates"]["n_entries"] for r in recs)
    cv_com = sum(r["covariates"]["n_complete"] for r in recs)
    cv_win_partial = sum(1 for r in recs if r["covariates"]["has_partial_entry"])
    return {
        "scheme": scheme, "scope": scope, "domain": domain, "n": len(recs),
        "content_median": med(content), "content_p90": p90(content),
        "content_max": max(content) if content else None,
        "idle_median": med(idle), "idle_p90": p90(idle),
        "idle_max": max(idle) if idle else None,
        "still_truncated_frac": (sum(1 for r in recs if r["still_truncated"])
                                 / len(recs)) if recs else None,
        "ev_total": ev["n_total"], "ev_complete": ev["n_complete"],
        "ev_partial": ev["n_partial"], "ev_absent": ev["n_absent"],
        "ev_complete_frac": ev["n_complete"] / ev["n_total"] if ev["n_total"] else None,
        "cv_entries": cv_ent, "cv_complete": cv_com,
        "cv_complete_frac": cv_com / cv_ent if cv_ent else None,
        "cv_win_partial_frac": cv_win_partial / len(recs) if recs else None,
    }


def reference_check(recs_test):
    """Recompute the three EXP-011 reference numbers from the D0 trace."""
    ev_raw = [r["blocks"]["Events"]["raw_tokens"] for r in recs_test]
    idle = [r["idle_tokens"] for r in recs_test]
    over = sum(1 for x in ev_raw if x > 510)
    got = {"events_raw_median_test": med(ev_raw), "idle_median_test": med(idle),
           "events_alone_over510_test": over, "n_test": len(recs_test)}
    check = {k: {"expected": EXPECTED_REF[k], "recomputed": got[k],
                 "match": abs(got[k] - EXPECTED_REF[k]) < 0.5 if isinstance(
                     got[k], float) else got[k] == EXPECTED_REF[k]}
             for k in EXPECTED_REF}
    return {"values": got, "check": check,
            "all_match": all(c["match"] for c in check.values())}


def write_csv(path, rows):
    if not rows:
        return
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[out] {path.name} ({len(rows)} rows)")


def domain_field_rows(recs, scheme, scope, domain):
    per = {f: {"raw": [], "alloc": [], "cleaned": []} for f in FIELDS}
    for r in recs:
        for f in FIELDS:
            b = r["blocks"][f]
            per[f]["raw"].append(b["raw_tokens"])
            per[f]["alloc"].append(b["alloc"])
            per[f]["cleaned"].append(
                b.get("cleaned_tokens", b["raw_tokens"]))
    rows = []
    for f in FIELDS:
        d = per[f]
        rows.append({
            "scheme": scheme, "scope": scope, "domain": domain, "field": f,
            "raw_median": med(d["raw"]), "raw_p90": p90(d["raw"]),
            "alloc_median": med(d["alloc"]),
            "alloc_sum": sum(d["alloc"]), "raw_sum": sum(d["raw"]),
            "cleaned_median": med(d["cleaned"]),
            "cleaned_sum": sum(d["cleaned"])})
    return rows


def main():
    RESULTS.mkdir(exist_ok=True)
    data = {s: load(s) for s in SCHEMES}
    key_sets = {s: {(r["var_key"], r["sample_id"]) for r in data[s]}
                for s in SCHEMES}
    assert len(key_sets["D0"]) == len(data["D0"]), "duplicate keys in D0"
    assert key_sets["D0"] == key_sets["D1"] == key_sets["D2"], "key set mismatch"
    by_scope = {s: {"full": data[s],
                    "test": [r for r in data[s] if r["scope"] == "test"]}
                for s in SCHEMES}
    print("[splits] " + str(dict(Counter(r["scope"] for r in data["D0"]))))
    print("[scope] " + ", ".join(
        f"{s}: full={len(by_scope[s]['full'])} test={len(by_scope[s]['test'])}"
        for s in SCHEMES))

    overall, domain, fields, ratios, dom_fields = [], [], [], [], []
    for s in SCHEMES:
        for scope in ("full", "test"):
            recs = by_scope[s][scope]
            overall.append(agg_rows(s, recs, scope, "__ALL__"))
            by_dom = defaultdict(list)
            for r in recs:
                by_dom[r["domain"]].append(r)
            for dom in sorted(by_dom):
                domain.append(agg_rows(s, by_dom[dom], scope, dom))
                dom_fields.extend(
                    domain_field_rows(by_dom[dom], s, scope, dom))
            fields.extend(field_rows(recs, s, scope))
            if s == "D2":
                ratios.extend(d2_ratios(recs, scope))

    write_csv(RESULTS / "stats_overall.csv", overall)
    write_csv(RESULTS / "stats_domain.csv", domain)
    write_csv(RESULTS / "stats_domain_fields.csv", dom_fields)
    write_csv(RESULTS / "stats_fields.csv", fields)
    write_csv(RESULTS / "stats_d2_ratios.csv", ratios)

    ref = reference_check(by_scope["D0"]["test"])
    (RESULTS / "reference_check.json").write_text(json.dumps(ref, indent=2))
    print("[ref] " + json.dumps(ref["check"]) + " all_match=" +
          str(ref["all_match"]))
    assert ref["all_match"], "reference values do not reconcile"

    # console summary of the four report questions
    for scope in ("full", "test"):
        d0, d1, d2 = (next(r for r in overall if r["scope"] == scope
                           and r["scheme"] == s) for s in SCHEMES)
        print(f"[{scope}] content med D0/D1/D2 = "
              f"{d0['content_median']:.0f}/{d1['content_median']:.0f}/"
              f"{d2['content_median']:.0f} | idle med "
              f"{d0['idle_median']:.0f}/{d1['idle_median']:.0f}/"
              f"{d2['idle_median']:.0f} | still_trunc "
              f"{d0['still_truncated_frac']:.3f}/{d1['still_truncated_frac']:.3f}"
              f"/{d2['still_truncated_frac']:.3f}")
        print(f"[{scope}] ev complete D0={d0['ev_complete']}/{d0['ev_total']} "
              f"D1={d1['ev_complete']}/{d1['ev_total']} "
              f"D2={d2['ev_complete']}/{d2['ev_total']} | cv complete "
              f"D0={d0['cv_complete']}/{d0['cv_entries']} "
              f"D1={d1['cv_complete']}/{d1['cv_entries']} "
              f"D2={d2['cv_complete']}/{d2['cv_entries']}")


if __name__ == "__main__":
    main()
