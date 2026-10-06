#!/usr/bin/env python3
"""D2 deterministic cleanup rules (analysis-text-budget-v1).

clean_fields() receives ONLY the four field strings and returns cleaned FIELD
TEXT WITHOUT block names — the assembly adds '{BlockName}: ' uniformly (user
correction 2026-10-07). Rules are frozen in d2_rules.json; anything uncertain
is kept. No LLM, no semantic merging, no relevance filtering.
"""
import json
import re
from pathlib import Path

SUBDIR = Path(__file__).resolve().parent
RULES = json.loads((SUBDIR / "d2_rules.json").read_text())
_CFG = {r["id"]: r for r in RULES["rules"]}

_R1 = re.compile(_CFG["R1"]["pattern"])
_R2 = re.compile(_CFG["R2"]["pattern"])
_R3 = re.compile(_CFG["R3"]["pattern"])
_R4 = re.compile(_CFG["R4"]["pattern"])
_R5 = re.compile(_CFG["R5"]["pattern"])
_SPACES = re.compile(r"  +")
_SPACE_COMMA = re.compile(r" ,")
_SPACE_DOT = re.compile(r" \.")


def normalize(text):
    """Same whitespace collapse as the frozen D0 builder."""
    return " ".join(str(text).split())


def clean_scenario(text, record):
    t = normalize(text)
    m = _R1.match(t)
    record["R1_prefix_matched"] = bool(m)
    if m:
        out = f"Prediction period: {m.group(1)} to {m.group(2)}. " + t[m.end():]
    else:
        out = t
    n_cit = len(_R2.findall(out))
    record["R2_citations_deleted"] = n_cit
    out = _R2.sub(" ", out)
    n_sc = len(_SPACE_COMMA.findall(out))
    n_sd = len(_SPACE_DOT.findall(out))
    record["R6_space_comma_fixed"] = n_sc
    record["R6_space_dot_fixed"] = n_sd
    out = _SPACES.sub(" ", _SPACE_COMMA.sub(",", _SPACE_DOT.sub(".", out)))
    return out.strip()


def clean_background(text, record):
    t = normalize(text)
    m = _R3.match(t)
    record["R3_template_matched"] = bool(m)
    if not m:
        return t  # keep original, recorded as unmatched
    return f"{m.group('name')}; domain: {m.group('domain')}; frequency: {m.group('freq')}."


def clean_holiday(text, record):
    t = normalize(text)
    if t.lower() in ("", "unknown"):
        record["skipped_value"] = t.lower()
        return t
    m = _R5.match(t)
    record["R5_prefix_matched"] = bool(m)
    out = t[m.end():] if m else t
    return _SPACES.sub(" ", out).strip()


def clean_covariates(text, record):
    t = normalize(text)
    m = _R4.match(t)
    record["R4_prefix_matched"] = bool(m)
    out = f"{m.group(1)} to {m.group(2)}: " + t[m.end():] if m else t
    return _SPACES.sub(" ", out).strip()


def clean_fields(background, scenario, holiday_info, covariates_info):
    """Only the four field strings go in; returns ({field: cleaned_text},
    {field: deletion_record}). Cleaned text never contains block names."""
    rec_b, rec_s, rec_h, rec_c = {}, {}, {}, {}
    cleaned = {
        "background": clean_background(background, rec_b),
        "scenario": clean_scenario(scenario, rec_s),
        "holiday_info": clean_holiday(holiday_info, rec_h),
        "covariates_info": clean_covariates(covariates_info, rec_c),
    }
    return cleaned, {"background": rec_b, "scenario": rec_s,
                     "holiday_info": rec_h, "covariates_info": rec_c}
