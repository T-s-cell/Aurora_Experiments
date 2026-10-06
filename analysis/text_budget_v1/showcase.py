#!/usr/bin/env python3
"""Showcase windows + fact-retention checks (analysis-text-budget-v1).

Deterministic ~20 train/val windows: >=1 per domain (19), both freqs, one
longest + one shortest content window, >=1 Calendar-skipped (Unknown).

Fact check (user correction 2026-10-07): compares RAW field text vs the
CLEANED UNTRUNCATED text (not the final truncated input). Facts = dates and
numbers extracted from raw after removing citation spans deleted by R2
(citation contents are exempt by definition). Both sides are normalized
(lowercase + whitespace collapse) before matching — no decoded-string
substring matching against raw punctuation spacing.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys_p = str(PROJECT / "repro_mm_timesx_v1")
import sys  # noqa: E402
sys.path.insert(0, sys_p)
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from text_builder import find_bert_config  # noqa: E402
from transformers import BertTokenizer  # noqa: E402

OUT = HERE / "outputs"
RESULTS = HERE / "results"
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
CIT_RE = re.compile(r"\[ *[0-9]{1,3}(?: *, *[0-9]{1,3})* *\]")

BACK_TPL = re.compile(
    r"^This time series records (?P<name>.+?) data in the (?P<domain>.+?) "
    r"domain, with a collection frequency of (?P<freq>daily|weekly), "
    r"focusing on monitoring and analyzing relevant indicators\.",
    re.IGNORECASE)
FIELDS = ("Background", "Events", "Calendar", "Covariates")


def norm(s):
    return " ".join(str(s).lower().split())


def facts_of(text):
    """Dates + numbers, excluding citation-list contents (R2 deletes them)."""
    without_cit = CIT_RE.sub(" ", str(text))
    dates = DATE_RE.findall(without_cit)
    nums = [n for n in NUM_RE.findall(without_cit)
            if not DATE_RE.fullmatch(n)]
    return dates + nums


def check_field(field_raw, field_clean):
    facts = facts_of(field_raw)
    nclean = norm(field_clean)
    lost = sorted({f for f in facts if f not in nclean})
    return {"n_facts": len(facts), "n_lost": len(lost), "lost": lost}


def check_background(field_raw, field_clean):
    m = BACK_TPL.match(norm(field_raw))
    if not m:
        # unmatched template is a FAILURE, not a pass (audit 2026-10-07)
        return {"template_matched": False, "n_facts": 0, "n_lost": 0,
                "lost_facts": []}
    nclean = norm(field_clean)
    facts = {"name": m.group("name"), "domain": m.group("domain"),
             "freq": m.group("freq")}
    lost = [k for k, v in facts.items() if norm(v) not in nclean]
    return {"template_matched": True, "n_facts": 3, "n_lost": len(lost),
            "lost_facts": lost}


def pick_windows(recs):
    """Deterministic: median-content window per domain over frozen train/val
    splits only (excluded windows are not eligible); force >=1
    Calendar-skipped; add global longest and shortest content windows."""
    tv = [r for r in recs if r["scope"] in ("train", "val")]
    by_dom = defaultdict(list)
    for r in tv:
        by_dom[r["domain"]].append(r)
    chosen = {}
    for dom in sorted(by_dom):
        cand = sorted(by_dom[dom],
                      key=lambda r: (r["content_tokens"], r["var_key"],
                                     r["sample_id"]))
        chosen[dom] = cand[len(cand) // 2]
    # force >=1 Unknown-Calendar window among the chosen (swap, not add)
    if not any(r["blocks"]["Calendar"]["skipped"] for r in chosen.values()):
        for dom in sorted(chosen,
                          key=lambda d: chosen[d]["content_tokens"]):
            cand = [r for r in by_dom[dom] if r["blocks"]["Calendar"]["skipped"]]
            if cand:
                cand.sort(key=lambda r: (r["content_tokens"], r["var_key"],
                                         r["sample_id"]))
                chosen[dom] = cand[len(cand) // 2]
                break
    out = list(chosen.values())
    for tag, rev in (("longest", True), ("shortest", False)):
        cand = sorted(tv, key=lambda r: (r["content_tokens"], r["var_key"],
                                         r["sample_id"]), reverse=rev)
        if cand[0] not in out:
            out.append(cand[0])
    return sorted(out, key=lambda r: (r["domain"], r["var_key"],
                                      r["sample_id"]))


def main():
    recs_d2 = [json.loads(l) for l in
               open(OUT / "trace_D2.jsonl", encoding="utf-8")]
    recs_d0 = [json.loads(l) for l in
               open(OUT / "trace_D0.jsonl", encoding="utf-8")]
    pos_d0 = {(r["var_key"], r["sample_id"]): i for i, r in enumerate(recs_d0)}
    chosen = pick_windows(recs_d2)
    doms = {r["domain"] for r in recs_d2 if r["scope"] in ("train", "val")}
    missing_doms = doms - {r["domain"] for r in chosen}
    assert not missing_doms, f"showcase misses domains: {missing_doms}"
    assert all(r["scope"] in ("train", "val") for r in chosen), \
        "showcase picked a non-train/val window"
    assert any(r["blocks"]["Calendar"]["skipped"] for r in chosen)
    freqs = {r["freq"] for r in chosen}
    print(f"[showcase] {len(chosen)} windows (train/val only), "
          f"domains={len(doms)}, freqs={sorted(freqs)}, "
          f"unknown_calendar={sum(1 for r in chosen if r['blocks']['Calendar']['skipped'])}")

    tok = BertTokenizer.from_pretrained(find_bert_config(),
                                        local_files_only=True)
    z2 = np.load(OUT / "trace_D2.npz", allow_pickle=False)
    z0 = np.load(OUT / "trace_D0.npz", allow_pickle=False)
    pos2 = {(r["var_key"], r["sample_id"]): i for i, r in enumerate(recs_d2)}
    from text_cache import build_index
    idx = build_index()

    lines = ["# Showcase windows (train/val, deterministic)\n"]
    all_ok = True
    jout = []
    for r in chosen:
        key = (r["var_key"], r["sample_id"])
        i2, i0 = pos2[key], pos_d0[key]
        cleaned = r["cleaned_text"]
        raw_fields = idx[key]["fields"]

        checks, blk = {}, {}
        checks["background"] = check_background(raw_fields["background"],
                                                cleaned["background"])
        for f in ("scenario", "covariates_info", "holiday_info"):
            checks[f] = check_field(raw_fields[f], cleaned[f])

        def field_text(z, rec, row, name):
            m_ = rec["blocks"][name]
            if m_["skipped"]:
                return "(skipped)"
            ids = z["ids"][row, m_["start"]:m_["end"]]
            return tok.decode(ids.tolist(), skip_special_tokens=False)

        for f in FIELDS:
            blk[f] = {"D0": field_text(z0, recs_d0[i0], i0, f),
                      "D2": field_text(z2, r, i2, f)}

        lost_any = any(v.get("n_lost", 0) > 0 for v in checks.values())
        bg_unmatched = not checks["background"]["template_matched"]
        bad = lost_any or bg_unmatched
        all_ok &= not bad
        verdict = "OK" if not bad else (
            "BG_TEMPLATE_UNMATCHED" if bg_unmatched else "FACT_LOST")
        lines.append(f"## {key[0]} / {key[1]}  [{r['domain']}, {r['freq']}, "
                     f"{r['scope']}, content={r['content_tokens']}, "
                     f"{verdict}]\n")
        for f, v in checks.items():
            lost = v.get("lost", v.get("lost_facts", ""))
            lines.append(f"- {f}: matched={v.get('template_matched', 'n/a')} "
                         f"facts={v.get('n_facts')} lost={v.get('n_lost')} "
                         f"{lost}")
        lines.append("")
        for f in FIELDS:
            lines.append(f"### {f}\n- **D0**: {blk[f]['D0']}\n- **D2**: "
                         f"{blk[f]['D2']}\n")
        jout.append({"var_key": key[0], "sample_id": key[1],
                     "domain": r["domain"], "freq": r["freq"],
                     "scope": r["scope"], "checks": checks,
                     "blocks_d0": {f: blk[f]["D0"] for f in FIELDS},
                     "blocks_d2": {f: blk[f]["D2"] for f in FIELDS}})
        print(f"[show] {key[0]}/{key[1]} {r['domain']} {verdict}")

    (RESULTS / "showcase.md").write_text("\n".join(lines), encoding="utf-8")
    (RESULTS / "showcase.json").write_text(
        json.dumps(jout, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[showcase] fact check {'ALL OK' if all_ok else 'HAS LOSSES'}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
