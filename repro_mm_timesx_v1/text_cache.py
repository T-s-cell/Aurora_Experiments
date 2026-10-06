#!/usr/bin/env python3
"""M48T512: (S2) frozen-data verification + (S3) fixed-text-budget cache builder.

Subcommands
  verify  — S2 checks over the frozen zip, read-only:
            mapping (2474 test windows resolve uniquely via the ORIGINAL
            sample-id logic), covariate span (start/end dates vs history /
            prediction start), Events date scan (strip the fixed
            "Prediction target period: from … to …" prefix first).
  build   — S3: token cache for the 2,474 test windows
            (cache/text_tokens_M48T512.npz) + per-window audit meta
            (cache/text_meta_M48T512.jsonl). Prints sha256 of both artifacts
            (these values are pinned into protocol_mm.json before freezing).

Reads only data/TimesX_Datasets.zip + the root frozen data artifacts; writes
only under repro_mm_timesx_v1/.
"""
import argparse
import hashlib
import importlib.util
import json
import re
import sys
import zipfile
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

SUBDIR = Path(__file__).resolve().parent
PROJECT = SUBDIR.parent
sys.path.insert(0, str(PROJECT))

from data_loader import TimesXData  # noqa: E402  (root, frozen)
from text_builder import BLOCKS, TOTAL_LEN, build_tokens, find_bert_config, normalize_field  # noqa: E402

# Original sample-id logic (vendored read-only module, loaded by path so the
# reference tree's import side effects are not triggered).
_spec = importlib.util.spec_from_file_location(
    "exp004_timesx_data", PROJECT / "reference" / "exp004" / "timesx_data.py")
_txd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_txd)
parse_ts = _txd.parse_ts
make_sample_id = _txd.make_sample_id
domain_of = _txd.domain_of

ZIP_PATH = PROJECT / "data" / "TimesX_Datasets.zip"
PREFLIGHT = SUBDIR / "preflight"
CACHE_DIR = SUBDIR / "cache"

FIELDS = ("background", "scenario", "holiday_info", "covariates_info")
SID_SUFFIX_RE = re.compile(r"^(\d{8}T\d{6})(?:__i(\d+))?$")
PRED_PREFIX_RE = re.compile(
    r"^Prediction target period: from \d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}\.\s*")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
COV_HEADER_RE = re.compile(
    r"^Covariate information from (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2}):")
WEEKEND_TOL = {"1D": timedelta(days=3), "1W": timedelta(days=7)}

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MON = r"(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
_SUFFIX = r"(?:st|nd|rd|th)?"
EN_DATE_MDY = re.compile(rf"\b({_MON})\.?\s+(\d{{1,2}}){_SUFFIX}\s*,?\s+(\d{{4}})\b", re.I)
EN_DATE_DMY = re.compile(rf"\b(\d{{1,2}}){_SUFFIX}\s+({_MON})\.?,?\s+(\d{{4}})\b", re.I)
EN_DATE_MD = re.compile(rf"\b({_MON})\.?\s+(\d{{1,2}}){_SUFFIX}(?!\s*,?\s*\d{{4}})\b", re.I)
EN_DATE_DM = re.compile(rf"\b(\d{{1,2}}){_SUFFIX}\s+({_MON})\.?(?!\s*,?\s*\d{{4}})\b", re.I)

# first-pass review-aid buckets (auto tags are NOT a leakage verdict)
_SCHED_KW = re.compile(
    r"\b(release[sd]?|schedul\w+|announc\w+|plan\w*|expect\w+|forecast\w*|"
    r"upcoming|set to|due (?:on|for|to)|will (?:be|release|publish|report)|"
    r"projected?|anticipated?|estimat\w+)\b", re.I)
_REALIZED_KW = re.compile(
    r"\b(report\w*|rose|fell|jump\w+|drop\w+|surge[ds]?|plunge[ds]?|climb\w+|"
    r"declin\w+|gained|lost|increase[ds]?|decrease[ds]?|soared|slumped|"
    r"posted|recorded|reached|according to|said|settled|closed|traded)\b", re.I)


def _month_of(tok):
    return _MONTHS[tok.lower().rstrip(".")]


def find_english_dates(text):
    """[(date, has_year, span, matched)] — deduplicated by (date, span-start)."""
    out = []
    seen = set()
    for m in EN_DATE_MDY.finditer(text):
        try:
            d = date(int(m.group(3)), _month_of(m.group(1)), int(m.group(2)))
        except ValueError:
            continue
        out.append((d, True, m.span(), m.group(0)))
        seen.add((d, m.start()))
    for m in EN_DATE_DMY.finditer(text):
        try:
            d = date(int(m.group(3)), _month_of(m.group(2)), int(m.group(1)))
        except ValueError:
            continue
        if (d, m.start()) not in seen:
            out.append((d, True, m.span(), m.group(0)))
            seen.add((d, m.start()))
    for m in EN_DATE_MD.finditer(text):
        try:
            d_tpl = (_month_of(m.group(1)), int(m.group(2)))
        except (KeyError, ValueError):
            continue
        out.append((d_tpl, False, m.span(), m.group(0)))
    for m in EN_DATE_DM.finditer(text):
        try:
            d_tpl = (_month_of(m.group(2)), int(m.group(1)))
        except (KeyError, ValueError):
            continue
        out.append((d_tpl, False, m.span(), m.group(0)))
    return out


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- zip index

def build_index(zip_path=ZIP_PATH):
    """(var_key, sample_id) -> record, replicating the original iter_variables
    ID rule verbatim (make_sample_id + future_start sort not needed here)."""
    index = {}
    with zipfile.ZipFile(zip_path) as z:
        jsons = sorted(n for n in z.namelist() if n.endswith(".json"))
        wrapper = jsons[0].split("/")[0]
        for n in jsons:
            rel = n[len(wrapper) + 1:]
            parts = rel.split("/")
            group, domain = domain_of(parts[:-1] if parts[-1] else parts)
            var = parts[-1][:-5]
            info = json.loads(z.read(n))
            seen = set()
            for s in info["samples"]:
                fts = [parse_ts(t) for t in s["future_time"]["timestamp"]]
                sid = make_sample_id(domain, var, fts[0], s.get("idx"), seen)
                key = (f"{domain}__{var}", sid)
                assert key not in index, f"duplicate composite key in zip index: {key}"
                index[key] = {
                    "zip_relpath": n,
                    "idx": s.get("idx"),
                    "freq": s.get("freq"),
                    "fields": {f: s.get(f) for f in FIELDS},
                    "future_start": fts[0],
                    "past_ts": s["past_time"]["timestamp"],
                    "date": s.get("date"),
                }
    return index


def group_by_var(index):
    g = {}
    for (vk, sid), rec in index.items():
        g.setdefault(vk, []).append((sid, rec))
    return g


def resolve_independently(index, var_key, sample_id, by_var=None):
    """Re-derive the sample from the parsed sid suffix (not by dict hit alone):
    parse trailing timestamp (+ optional __i<idx>), require EXACTLY one zip
    sample of this variable with that future_start (and idx). Also round-trips
    the original make_sample_id rule."""
    assert sample_id.startswith(var_key + "__"), \
        f"sample_id {sample_id} does not start with var_key {var_key}"
    suffix = sample_id[len(var_key) + 2:]
    m = SID_SUFFIX_RE.match(suffix)
    assert m, f"unparseable sample_id suffix: {sample_id}"
    ts = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")
    want_idx = int(m.group(2)) if m.group(2) is not None else None

    if by_var is None:
        by_var = group_by_var(index)
    candidates = [r for _sid, r in by_var.get(var_key, ())
                  if r["future_start"] == ts
                  and (want_idx is None or r["idx"] == want_idx)]
    assert len(candidates) == 1, \
        f"resolution not unique ({len(candidates)}) for {var_key} / {sample_id}"

    # round-trip: the original rule must regenerate this sample_id's base form
    regen = make_sample_id(var_key.split("__")[0], var_key.split("__", 1)[1],
                           ts, candidates[0]["idx"], set())
    if want_idx is None:
        assert regen == sample_id, f"round-trip mismatch: {regen} != {sample_id}"
    else:
        assert regen.startswith(sample_id.split("__i")[0]), \
            f"round-trip base mismatch for {sample_id}"
    return candidates[0]


# ------------------------------------------------------------------- checks

def check_fields(index, rows, by_var):
    bad = []
    for vk, sid, *_ in rows:
        rec = resolve_independently(index, vk, sid, by_var)
        for f in FIELDS:
            v = rec["fields"][f]
            if not isinstance(v, str) or not v.strip():
                bad.append({"var_key": vk, "sample_id": sid, "field": f,
                            "value": repr(v)[:120]})
    return bad


def check_covariate_span(index, rows, label, by_var):
    """start >= past[0] date; end strictly before prediction start (leakage
    guard); end within past[-1] date + weekend tolerance (3d daily / 7d weekly)."""
    violations, tol_hits = [], []
    for vk, sid, *_ in rows:
        rec = resolve_independently(index, vk, sid, by_var)
        cov = rec["fields"]["covariates_info"]
        m = COV_HEADER_RE.match(normalize_field(cov))
        if not m:
            violations.append({"var_key": vk, "sample_id": sid, "reason": "no parseable header",
                               "head": cov[:160]})
            continue
        d0 = datetime.fromisoformat(m.group(1)).date()
        d1 = datetime.fromisoformat(m.group(2)).date()
        past0 = datetime.fromisoformat(rec["past_ts"][0]).date()
        past1 = datetime.fromisoformat(rec["past_ts"][-1]).date()
        tol = WEEKEND_TOL.get(rec["freq"], timedelta(days=7))
        if d0 < past0:
            violations.append({"var_key": vk, "sample_id": sid, "reason": f"start {d0} < history start {past0}",
                               "freq": rec["freq"]})
        if d1 >= rec["future_start"].date():
            violations.append({"var_key": vk, "sample_id": sid,
                               "reason": f"end {d1} >= prediction start {rec['future_start'].date()}",
                               "freq": rec["freq"]})
        elif d1 > past1 + tol:
            tol_hits.append({"var_key": vk, "sample_id": sid,
                             "reason": f"end {d1} > history end {past1} + {tol}",
                             "freq": rec["freq"]})
    report = {"scope": label, "n_windows": len(rows), "n_violations": len(violations),
              "violations": violations, "n_tolerance_hits": len(tol_hits),
              "tolerance_hits": tol_hits}
    out = PREFLIGHT / f"covariate_span_{label}.json"
    PREFLIGHT.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report, out


def scan_events(index, rows, label, by_var, ctx_chars=300):
    """ISO-date hits >= prediction start in the Events text, AFTER stripping
    the fixed 'Prediction target period: from … to …' prefix. Reported for
    manual reading — NOT auto-deleted."""
    hits = []
    n_iso_in_bodies = 0
    for vk, sid, *_ in rows:
        rec = resolve_independently(index, vk, sid, by_var)
        body = PRED_PREFIX_RE.sub("", rec["fields"]["scenario"])
        pred0 = rec["future_start"].date()
        n_iso_in_bodies += len(DATE_RE.findall(body))
        for dm in DATE_RE.finditer(body):
            d = datetime.fromisoformat(dm.group(0)).date()
            if d >= pred0:
                s = max(0, dm.start() - 60)
                hits.append({"var_key": vk, "sample_id": sid, "date": str(d),
                             "context": body[s:dm.end() + ctx_chars]})
                break  # one sample hit per window is enough for manual reading
    report = {"scope": label, "n_windows": len(rows), "n_flagged": len(hits),
              "n_iso_dates_in_stripped_bodies": n_iso_in_bodies,
              "note": "flagged for manual reading; publication timing NOT independently "
                      "audited; calendar references are expected false positives. "
                      "Event bodies use natural-language dates (e.g. 'On September 13, "
                      "2023'), which this ISO-format scan cannot judge — recorded as a "
                      "limitation, not evidence of absence.",
              "hits": hits}
    out = PREFLIGHT / f"events_scan_{label}.json"
    PREFLIGHT.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report, out


def scan_events_english(index, rows, label, by_var, ctx_chars=140):
    """English-language date scan of the Events text, two scopes:
      scope=input : the ACTUAL model input — decode of the kept (<=270-token)
                    Events block from the frozen cache, by composite key
      scope=full  : the full raw scenario field (prefix stripped)
    Primary review set (written to CSV): windows whose input Events contain a
    YEAR-EXPLICIT English date ("September 13, 2023") on/after the prediction
    start. Year-less dates ("October 20") are ambiguous — the same text recurs
    yearly and historical mentions dominate — so they are counted
    INFORMATIONALLY (future-inferred over {pred.year, pred.year+1}), never as
    review-blocking flags. Auto keyword tags are review aids, NOT a verdict."""
    z = np.load(CACHE_DIR / "text_tokens_M48T512.npz", allow_pickle=False)
    cache_ids = {((str(vk), str(sid))): z["ids"][i]
                 for i, (vk, sid) in enumerate(zip(z["var_keys"], z["sample_ids"]))}
    meta = {}
    with open(CACHE_DIR / "text_meta_M48T512.jsonl", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            meta[(m["var_key"], m["sample_id"])] = m
    from transformers import BertTokenizer
    from text_builder import find_bert_config
    tok = BertTokenizer.from_pretrained(find_bert_config(), local_files_only=True)

    def _valid(y, mo, dy):
        try:
            datetime(y, mo, dy)
            return True
        except ValueError:
            return False

    hits_input, hits_full = [], []      # primary: year-explicit >= pred start
    n_win_input = n_win_full = 0
    n_info_noyear_input = 0             # informational: year-less, future-inferred
    n_info_noyear_hits = 0
    for vk, sid, *_ in rows:
        key = (vk, sid)
        rec = resolve_independently(index, vk, sid, by_var)
        pred0 = rec["future_start"].date()
        years = (pred0.year, pred0.year + 1)

        def _scan(text, scope, store):
            primary = info = False
            for d, has_year, span, matched in find_english_dates(text):
                s = max(0, span[0] - 80)
                context = text[s:span[1] + ctx_chars]
                if has_year:
                    if d >= pred0:
                        primary = True
                        store.append({
                            "var_key": vk, "sample_id": sid, "scope": scope,
                            "matched": matched, "date": str(d),
                            "pred_start": str(pred0),
                            "tag_schedule_like": bool(_SCHED_KW.search(context)),
                            "tag_realized_like": bool(_REALIZED_KW.search(context)),
                            "context": context,
                        })
                else:
                    mo, dy = d
                    if any(datetime(y, mo, dy).date() >= pred0
                           for y in years if _valid(y, mo, dy)):
                        info = True
            return primary, info

        # scope=input: decode the frozen kept Events block
        mblk = next(b for b in meta[key]["blocks"] if b["block"] == "Events")
        if mblk["skipped"]:
            input_text = ""
        else:
            ids = cache_ids[key]
            input_text = tok.decode(ids[mblk["start"]:mblk["end"]].tolist(),
                                    skip_special_tokens=False)
        p_in, i_in = _scan(input_text, "input", hits_input)
        n_win_input += int(p_in)
        n_info_noyear_input += int(i_in)
        full_text = PRED_PREFIX_RE.sub("", rec["fields"]["scenario"])
        p_full, _ = _scan(full_text, "full", hits_full)
        n_win_full += int(p_full)

    report = {
        "scope_label": label,
        "n_windows": len(rows),
        "n_windows_flagged_input": n_win_input,
        "n_windows_flagged_full": n_win_full,
        "n_hits_input": len(hits_input),
        "n_hits_full": len(hits_full),
        "n_windows_info_noyear_input": n_info_noyear_input,
        "note": "PRIMARY flags (CSV) = input Events decode contains a "
                "year-explicit English date on/after the prediction start. "
                "Advance-published schedules are NOT leakage; every primary "
                "window needs manual classification sign-off before the full "
                "run. Year-less month-day references are counted "
                "informationally only (n_windows_info_noyear_input): the same "
                "text recurs yearly so year inference is unreliable. The "
                "full scope covers the raw field including text beyond the "
                "kept 270-token budget that the model never consumes. "
                "Independent reference count from the reviewer's own scan: "
                "157 input windows; our pattern set (full+abbreviated month "
                "names, sept., optional ordinal suffix/comma) is a superset.",
        "hits_input": hits_input,
        "hits_full": hits_full,
    }
    out = PREFLIGHT / f"events_english_{label}.json"
    PREFLIGHT.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    # compact CSV for manual review (input scope only — this is what the model sees)
    import csv as _csv
    csv_path = PREFLIGHT / f"events_english_{label}_input.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["var_key", "sample_id", "matched", "date", "pred_start",
                    "tag_schedule_like", "tag_realized_like", "context"])
        for h in hits_input:
            w.writerow([h["var_key"], h["sample_id"], h["matched"], h["date"],
                        h["pred_start"], h["tag_schedule_like"],
                        h["tag_realized_like"], h["context"]])
    return report, out


def run_verify():
    print("[verify] building zip index ...")
    index = build_index()
    by_var = group_by_var(index)
    print(f"[verify] indexed {len(index)} samples, {len(by_var)} variables")

    td = TimesXData(PROJECT / "data")
    rows = td.test_rows()
    print(f"[verify] {len(rows)} test windows from frozen manifest")

    bad = check_fields(index, rows, by_var)
    (PREFLIGHT / "mapping_check.json").write_text(json.dumps(
        {"n_windows": len(rows), "n_unresolved_or_missing_field": len(bad), "bad": bad},
        indent=2, ensure_ascii=False))
    print(f"[verify] mapping+fields: {len(bad)} problems")

    cov_test, p1 = check_covariate_span(index, rows, "test", by_var)
    print(f"[verify] covariate span (test): {cov_test['n_violations']} violations, "
          f"{cov_test['n_tolerance_hits']} tolerance hits -> {p1.name}")

    ev_test, p2 = scan_events(index, rows, "test", by_var)
    print(f"[verify] events scan (test): {ev_test['n_flagged']} flagged -> {p2.name}")

    ev_en, p3 = scan_events_english(index, rows, "test", by_var)
    print(f"[verify] events English-date scan (test): "
          f"input-scope {ev_en['n_windows_flagged_input']} windows flagged, "
          f"full-scope {ev_en['n_windows_flagged_full']} -> {p3.name}")

    ok = not bad and cov_test["n_violations"] == 0
    print(f"[verify] RESULT: {'OK' if ok else 'FAILED — inspect preflight/ reports'}")


# -------------------------------------------------------------------- build

def run_build():
    print("[build] loading zip index ...")
    index = build_index()
    by_var = group_by_var(index)
    td = TimesXData(PROJECT / "data")
    rows = td.test_rows()
    print(f"[build] {len(rows)} test windows")

    from transformers import BertTokenizer
    tok = BertTokenizer.from_pretrained(find_bert_config(), local_files_only=True)

    ids_arr = np.zeros((len(rows), TOTAL_LEN), dtype=np.int32)
    mask_arr = np.zeros((len(rows), TOTAL_LEN), dtype=np.int32)
    vk_arr, sid_arr, dom_arr = [], [], []
    block_stats = {name: {"n_windows": 0, "n_skipped": 0, "n_truncated": 0,
                          "kept_ratio_min": 1.0, "kept_ratio_max": 0.0}
                   for name, _f, _b in BLOCKS}
    content_lens = []

    meta_path = CACHE_DIR / "text_meta_M48T512.jsonl"
    CACHE_DIR.mkdir(exist_ok=True)
    tmp_meta = meta_path.with_suffix(".jsonl.tmp")
    with open(tmp_meta, "w", encoding="utf-8") as mf:
        for i, (vk, sid, dom, _past, _target, _d) in enumerate(rows):
            rec = resolve_independently(index, vk, sid, by_var)
            ids, mask, meta = build_tokens(rec["fields"], tok)
            ids_arr[i], mask_arr[i] = ids, mask
            vk_arr.append(vk)
            sid_arr.append(sid)
            dom_arr.append(dom)
            content_lens.append(meta["content_tokens"])
            for bm in meta["blocks"]:
                st = block_stats[bm["block"]]
                st["n_windows"] += 1
                st["n_skipped"] += int(bm["skipped"])
                if not bm["skipped"] and bm["raw_tokens"] > bm["budget"]:
                    st["n_truncated"] += 1
                if not bm["skipped"] and bm["raw_tokens"]:
                    r = bm["kept_tokens"] / bm["raw_tokens"]
                    st["kept_ratio_min"] = min(st["kept_ratio_min"], r)
                    st["kept_ratio_max"] = max(st["kept_ratio_max"], r)
            mf.write(json.dumps({
                "var_key": vk, "sample_id": sid, "domain": dom,
                "freq": rec["freq"], "zip_relpath": rec["zip_relpath"],
                "zip_idx": rec["idx"],
                "blocks": meta["blocks"],
                "content_tokens": meta["content_tokens"],
                "pad_tokens": meta["pad_tokens"],
                "decoded_text": meta["decoded_text"],
                "decoded_sha256": meta["decoded_sha256"],
                "blocks_skipped": meta["blocks_skipped"],
            }, ensure_ascii=False) + "\n")
            if (i + 1) % 500 == 0:
                print(f"[build] {i + 1}/{len(rows)}")
    tmp_meta.rename(meta_path)

    npz_path = CACHE_DIR / "text_tokens_M48T512.npz"
    tmp_npz = CACHE_DIR / ".text_tokens_M48T512.npz.tmp"
    with open(tmp_npz, "wb") as f:
        np.savez(f, ids=ids_arr, mask=mask_arr,
                 var_keys=np.array(vk_arr), sample_ids=np.array(sid_arr),
                 domains=np.array(dom_arr))
    tmp_npz.replace(npz_path)

    stats = {
        "n_windows": len(rows),
        "content_tokens_min": int(min(content_lens)),
        "content_tokens_max": int(max(content_lens)),
        "blocks": block_stats,
        "text_tokens_npz_sha256": sha256_file(npz_path),
        "text_meta_jsonl_sha256": sha256_file(meta_path),
    }
    (CACHE_DIR / "build_stats_M48T512.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"[build] wrote {npz_path.name} and {meta_path.name}; pin these sha256 "
          f"values into protocol_mm.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["verify", "build"])
    args = ap.parse_args()
    {"verify": run_verify, "build": run_build}[args.cmd]()
