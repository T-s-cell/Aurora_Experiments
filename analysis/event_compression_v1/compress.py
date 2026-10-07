#!/usr/bin/env python3
"""S3 driver: run D2 / E-Extract / E-Summary over the selected windows.

  python compress.py --set debug  [--offline]
  python compress.py --set val    [--offline]

--offline: no service contact; every non-fit LLM event fails => window falls
back to the frozen D2 input (sanitized path used when the tunnel/service or
key is unavailable).

Hard gates per compressed window: Events token length <= E on the final
assembled string; Background/Calendar/Covariates kept token content bitwise
equal to D2 at the recomputed (shifted) boundaries; content <= 510; CLS/SEP/
mask correct. Any gate violation => that scheme's window falls back to D2 and
is counted (never silently patched).
"""
import argparse
import hashlib
import json
import re
import sys

from cache import LLMCache, cache_key
from ec_common import (HERE, OUT, PROJECT, event_pieces, load_tokenizer,
                       rebuild_d2, split_events_clean, trace_npz_index,
                       verify_bitwise_vs_trace)

SCHEMES = ("E-Extract", "E-Summary")
REFUSAL_RE = re.compile(
    r"\b(i cannot|i can't|cannot assist|no relevant|not applicable|"
    r"unable to|as an ai|i'm sorry|i am sorry)\b", re.IGNORECASE)

# Grounding screen for E-Summary: every token of these classes in the
# summary must also appear (same notation) in the verbatim evidence spans.
NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
MONTH_RE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|"
    r"nov|dec)\b", re.IGNORECASE)
QUARTER_RE = re.compile(r"\bq[1-4]\b", re.IGNORECASE)
UNIT_RE = re.compile(
    r"%|\$|€|£|°c|°f|\b(?:mw|mwh|kwh|gw|gwh|twh|kw|kv|ma|barrels?|bbl|"
    r"tons?|tonnes?|mt|kt|bcf|mcf|mcm|hm3|usd|eur|billion|million|thousand|"
    r"hectares?|acres?|percent)\b", re.IGNORECASE)
FUT_RE = re.compile(
    r"\b(?:expect(?:ed|s)?|plan(?:ned|s|ning)?|may|might|could|will|would|"
    r"forecast(?:ed|s)?|project(?:ed|s)?|likely|aim(?:s|ed)?|seek(?:s)?|"
    r"intend(?:s|ed)?|anticipat(?:e|es|ed)|outlook|guidance|estimat(?:e|es|"
    r"ed)|scheduled|due to)\b", re.IGNORECASE)
GROUND_RES = (("numbers", NUM_RE), ("years", YEAR_RE), ("months", MONTH_RE),
              ("quarters", QUARTER_RE), ("units", UNIT_RE),
              ("forecast", FUT_RE))

# Content-word grounding: every non-function word of the summary must be
# traceable to an evidence word (exact, or mild suffix variation like
# restarts/restart). Catches paraphrase drift ("delaying" introduced for
# "expected to return") that token-class grounding cannot see.
STOPWORDS = frozenset("""a an the this that these those of in on at to by for
with from as is are was were be been being am it its their his her our your
my me you he she they them we us and or but while than then so such if
because has have had will would can could may might must shall should do
does did done not no nor s t don now here there when where why how which
who whom whose what says said say according into onto out up down off over
under about after before during again further once all any both each few
more most other same too very just also only own""".split())


def content_words(text):
    return [w for w in re.findall(r"[a-z][a-z'-]*", text.lower())
            if w not in STOPWORDS and len(w) > 1]


def word_covered(w, ground_set):
    if w in ground_set:
        return True
    if len(w) >= 5:
        for g in ground_set:
            if len(g) >= 4 and w.startswith(g) and len(w) - len(g) <= 3:
                return True
    return False

NEG_RE = re.compile(
    r"\b(?:not|no|nor|never|cannot|can't|without|cancel(?:l)?ed|halted|"
    r"suspended|postponed|delayed|shut|stopped|unavailable|failed)\b",
    re.IGNORECASE)

# v3.2 whole-clause gate for E-Extract: every span must be exactly one
# complete trimmed sentence (final punctuation may be dropped). Whole
# sentences retain subject, scope qualifiers ("for employers with 51 or
# more employees") and forecast/negation wording by construction, and
# facts from different sentences can no longer be spliced into one
# reading; assembled pieces separate non-adjacent spans with " ... ".
# Sentence boundaries are abbreviation-aware: news text shatters at
# "U.S." / "D.C." / "Inc." / "Jan." periods, so a boundary only counts
# when the token before the punctuation is not an abbreviation/initials
# and the next non-space char starts a sentence (upper/digit/quote).
BOUND_CAND_RE = re.compile(r"([.!?;])(\s+)")
NEXT_OK_RE = re.compile(r"""["'(\[]?[A-Z0-9]""")
INITIALS_RE = re.compile(r"(?:[A-Za-z]\.)+")
ABBREV_WORDS = frozenset("""mr mrs ms dr prof vs etc al inc ltd llc corp co
jr sr st mt ft gov sen rep capt col gen rev hon no vol pp approx est dept
univ assn jan feb mar apr jun jul aug sep sept oct nov dec mon tue tues wed
thu thur thurs fri sat sun min max ave blvd""".split())


def clause_index(event_text):
    """Complete-sentence ranges [(cs, ce)] in original char coords."""
    out, start, n = [], 0, len(event_text)
    for m in BOUND_CAND_RE.finditer(event_text):
        p, pe = m.start(), m.end()
        if pe >= n:
            break
        if not NEXT_OK_RE.match(event_text[pe:pe + 2]):
            continue
        wm = re.search(r"([A-Za-z][A-Za-z.]*)\.$", event_text[:p + 1])
        if wm and (INITIALS_RE.fullmatch(wm.group(1) + ".")
                   or wm.group(1).lower() in ABBREV_WORDS):
            continue
        seg = event_text[start:p + 1]
        s = start + (len(seg) - len(seg.lstrip()))
        e = start + len(seg.rstrip())
        if e > s:
            out.append((s, e))
        start = pe
    seg = event_text[start:]
    s = start + (len(seg) - len(seg.lstrip()))
    e = start + len(seg.rstrip())
    if e > s:
        out.append((s, e))
    return out


def clause_gate(event_text, located):
    """Returns None or a reject reason: each span must lie within exactly
    one clause and equal it in full (optional final-punctuation drop)."""
    clauses = clause_index(event_text)
    for a, b in located:
        hit = [c for c in clauses if a < c[1] and c[0] < b]
        if len(hit) > 1:
            return "span_crosses_clauses"
        if not hit:
            return "partial_clause"
        cs, ce = hit[0]
        if (a, b) == (cs, ce):
            continue
        if a == cs and b == ce - 1 and event_text[ce - 1] in ".!?;":
            continue
        return "partial_clause"
    return None


def assemble_spans(event_text, merged):
    """Deterministic piece from original-text ranges: ranges separated by
    more than one whitespace char (i.e. skipped source material) are joined
    with an explicit ' ... ' separator so facts from different clauses
    cannot read as one claim; gap==1 means the clauses are adjacent in the
    original text and keep a plain space."""
    parts = [event_text[a:b].strip() for a, b in merged]
    out = [parts[0]]
    for i in range(1, len(merged)):
        gap = merged[i][0] - merged[i - 1][1]
        out.append(" ... " if gap > 1 else " ")
        out.append(parts[i])
    return " " + "".join(out).strip()


def apply_tpl(tpl, **kw):
    """Substitute {name} placeholders only; literal JSON braces in the
    template (e.g. {"event_id": ...}) are left untouched."""
    return re.sub(r"\{(\w+)\}", lambda m: kw.get(m.group(1), m.group(0)),
                  tpl)


def sha_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_prompt_parts(rel_path):
    """Parse the frozen prompt markdown: fenced blocks in order ->
    (system, user_template, correction_template)."""
    blocks = re.findall(r"```[^\n]*\n(.*?)```",
                        (HERE / rel_path).read_text(), re.DOTALL)
    assert len(blocks) == 3, f"{rel_path}: expected 3 fenced blocks"
    return blocks[0].strip(), blocks[1], blocks[2]


def parse_model_json(content):
    """Strip code fences, extract the outermost JSON object."""
    s = content.strip()
    s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return None


def norm_index_map(text):
    """Whitespace-normalized view of `text` with a map back to original
    positions: returns (norm_str, norm_pos_to_orig_idx)."""
    out, pos = [], []
    prev_space = True
    for i, ch in enumerate(text):
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                pos.append(i)
                prev_space = True
            continue
        out.append(ch)
        pos.append(i)
        prev_space = False
    return "".join(out), pos


def span_to_original(span, event_text):
    """Locate span in event_text: exact byte match first; fallback to
    whitespace-normalized match MAPPED BACK to original offsets. Returns
    (start, end) in original coords or None (never returns normalized text)."""
    p = event_text.find(span)
    if p >= 0:
        return (p, p + len(span))
    norm, pos = norm_index_map(event_text)
    nspan, _ = norm_index_map(span)
    if not nspan:
        return None
    q = norm.find(nspan)
    if q < 0:
        return None
    start, end = pos[q], pos[q + len(nspan) - 1] + 1
    assert event_text[start:end].strip() == span.strip()
    return (start, end)


def merge_spans(spans):
    """Sort by start, merge overlapping/adjacent-touched regions."""
    spans = sorted(spans)
    out = []
    for s, e in spans:
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


class WindowCompressor:
    def __init__(self, cfg, tok, client, caches, prompt_shas, prompt_parts,
                 offline):
        self.cfg = cfg
        self.tok = tok
        self.client = client
        self.caches = caches        # method -> LLMCache
        self.prompt_shas = prompt_shas  # method -> sha256 hex
        self.prompts = prompt_parts  # method -> (system, user_tpl, corr_tpl)
        self.offline = offline
        self.enc = lambda s: tok.encode(s, add_special_tokens=False)

    def char_cap(self, prose, need, budget):
        ratio = max(len(prose) / max(need, 1), 1.0)
        return max(int(budget * ratio), 32)

    def _cap(self, method, prose_piece, budget):
        need = len(self.enc(prose_piece))
        return self.char_cap(prose_piece, need, budget)

    def compress_event(self, method, k, prose_piece, budget, attempts_log):
        """Returns (piece_text, source, meta) or (None, 'failed', meta).
        piece_text replaces the original prose piece (leading-space
        convention). attempts_log accumulates attempt records."""
        prose = prose_piece.strip()
        system, user_tpl, corr_tpl = self.prompts[method]
        event_text = prose
        messages = [{"role": "system", "content": system},
                    {"role": "user",
                     "content": apply_tpl(user_tpl, event_id=str(k),
                                          event_text=event_text,
                                          char_cap=str(self._cap(
                                              method, prose_piece, budget)),
                                          max_words=str(max(
                                              int(budget * 0.9), 8)))}]
        key = cache_key(prose, method, budget,
                        self.prompt_shas[method],
                        self.client.params_fingerprint(),
                        self.client.service_version)
        cached = self.caches[method].get(key)
        if cached is not None:
            payload = dict(cached["payload"])
            payload["event_id"] = str(k)
            ok, why, spans = self.validate(method, payload, event_text, budget)
            if ok:
                attempts_log.append({"k": k, "source": "cache", "ok": True})
                piece, extra = self.build_piece(method, payload, event_text)
                return piece, "cache", {"cache_key": key, **extra}
            attempts_log.append({"k": k, "source": "cache", "ok": False,
                                 "why": why})
        if self.offline:
            attempts_log.append({"k": k, "source": "offline", "ok": False,
                                 "why": "offline mode"})
            return None, "failed", {"cache_key": key}
        for attempt in range(3):
            resp = self.client.chat_messages(messages)
            if not resp.get("ok"):
                attempts_log.append({"k": k, "attempt": attempt,
                                     "ok": False, "why": resp.get("error")})
                return None, "failed", {"cache_key": key}
            content = resp["content"]
            rec = {"k": k, "attempt": attempt, "finish": resp["finish_reason"],
                   "model": resp.get("model")}
            if resp.get("truncated"):
                rec["ok"] = False
                rec["why"] = f"truncated (finish_reason={resp['finish_reason']})"
                attempts_log.append(rec)
            else:
                obj = parse_model_json(content)
                if obj is None:
                    rec["ok"] = False
                    rec["why"] = "json_parse"
                    attempts_log.append(rec)
                else:
                    obj.pop("event_id", None)
                    probe = dict(obj)
                    probe["event_id"] = str(k)
                    ok, why, spans = self.validate(method, probe, event_text,
                                                   budget)
                    if ok:
                        rec["ok"] = True
                        attempts_log.append(rec)
                        self.caches[method].put(key, method, obj)
                        piece, extra = self.build_piece(method, probe,
                                                        event_text)
                        return piece, "llm", {"cache_key": key, **extra}
                    rec["ok"] = False
                    rec["why"] = why
                    attempts_log.append(rec)
            if attempt < 2:
                messages = messages + [
                    {"role": "assistant", "content": content},
                    {"role": "user",
                     "content": apply_tpl(
                         corr_tpl,
                         reject_reason=str(rec.get("why", "invalid")),
                         char_cap=str(self._cap(method, prose_piece, budget)),
                         span_hint="Each span must be ONE COMPLETE clause "
                                   "copied verbatim in full, from its first "
                                   "word through its last word (only the "
                                   "final period may be dropped). Do not "
                                   "cross or cut clauses: a partial "
                                   "fragment, a number without its subject "
                                   "or scope (e.g. 'for employers with 51 "
                                   "or more employees'), or forecast/"
                                   "negation wording stripped from its "
                                   "clause is rejected. If several clauses "
                                   "are needed, they are joined with "
                                   "' ... ' separators automatically."
                                   if method == "E-Extract" else
                                   "Ensure the evidence spans are exact "
                                   "verbatim substrings of the event text "
                                   "and contain every number, date, unit and "
                                   "modality word used in the summary; "
                                   "match the event text's notation exactly "
                                   "and keep all scope qualifiers.")}]
        return None, "failed", {"cache_key": key}

    def validate(self, method, obj, event_text, budget):
        """Returns (ok, why, spans_or_None). obj['event_id'] must already be
        rebound to the current numbering."""
        if obj.get("event_id") is None:
            return False, "missing_event_id", None
        if method == "E-Extract":
            spans = obj.get("spans")
            if not isinstance(spans, list) or not spans or \
                    not all(isinstance(s, str) and s.strip() for s in spans):
                return False, "bad_spans", None
            located = []
            for s in spans:
                r = span_to_original(s, event_text)
                if r is None:
                    return False, "span_not_verbatim", None
                located.append(r)
            merged = merge_spans(located)
            qd = clause_gate(event_text, located)
            if qd:
                return False, qd, None
            text = assemble_spans(event_text, merged)
            if len(self.enc(text)) > budget:
                return False, f"over_budget({len(self.enc(text))}>{budget})", None
            if REFUSAL_RE.search(text):
                return False, "refusal_text", None
            return True, "ok", merged
        else:
            summ = obj.get("summary")
            if not isinstance(summ, str) or not summ.strip():
                return False, "bad_summary", None
            ev = obj.get("evidence")
            if not isinstance(ev, list) or not ev:
                return False, "missing_evidence", None
            ground_parts = []
            for e in ev:
                if not isinstance(e, str) or not e.strip():
                    return False, "bad_evidence", None
                r = span_to_original(e, event_text)
                if r is None:
                    return False, "evidence_not_verbatim", None
                ground_parts.append(event_text[r[0]:r[1]])
            ground = " ".join(ground_parts).lower()
            s_low = summ.lower()
            for name, rx in GROUND_RES:
                need = set(m.group(0).strip().lower()
                           for m in rx.finditer(s_low))
                have = set(m.group(0).strip().lower()
                           for m in rx.finditer(ground))
                missing = sorted(need - have)
                if missing:
                    return False, f"not_grounded({name}:{missing[:3]})", None
            ground_set = set(content_words(ground))
            ungrounded = sorted(w for w in content_words(s_low)
                                if not word_covered(w, ground_set))
            if ungrounded:
                return False, f"ungrounded_words:{ungrounded[:4]}", None
            if NEG_RE.search(ground) and not NEG_RE.search(summ):
                return False, "negation_dropped", None
            if REFUSAL_RE.search(summ):
                return False, "refusal_text", None
            if len(self.enc(" " + summ.strip())) > budget:
                n = len(self.enc(" " + summ.strip()))
                return False, f"over_budget({n}>{budget})", None
            obj["_evidence_ok"] = True
            obj["_evidence_loc"] = ground_parts
            return True, "ok", ground_parts

    def build_piece(self, method, obj, event_text):
        """Returns (piece_text, extra_meta); piece uses the leading-space
        convention. Assembly must use original-text fragments."""
        if method == "E-Extract":
            merged = merge_spans([span_to_original(s, event_text)
                                  for s in obj["spans"]])
            piece = assemble_spans(event_text, merged)
            return piece, {"span_ranges": [[a, b] for a, b in merged]}
        evidence_loc = obj["_evidence_loc"]
        return (" " + obj["summary"].strip(),
                {"evidence_ok": True, "evidence": evidence_loc})


def process_window(w, b, wc, tok, z, idx):
    """Returns per-(window, scheme) records + final ids/mask arrays."""
    from allocate import assemble
    vk, sid = w["var_key"], w["sample_id"]
    fields = w["_fields"]
    rb = rebuild_d2(fields, tok)
    assert verify_bitwise_vs_trace(rb["ids"], rb["mask"], vk, sid, z, idx)
    meta_d2 = rb["meta"]
    recs = {}
    finals = {}

    base = {"var_key": vk, "sample_id": sid, "domain": w["domain"],
            "scope": w["scope"], "freq": w["freq"], "E": b["E"]}
    if b["status"] != "ok":
        for m in SCHEMES:
            recs[m] = dict(base, scheme=m, outcome="kept_d2",
                           reason=b["status"], events=[])
            finals[m] = (rb["ids"], rb["mask"])
        return recs, finals, rb

    split = split_events_clean(rb["cleaned"]["scenario"], tok)
    head_ids, tag_ids, prose_ids = event_pieces(split, tok)

    for m in SCHEMES:
        events_rec, pieces, failed = [], {}, None
        for i, ev in enumerate(b["events"]):
            k = ev["k"]
            if ev["fit"]:
                pieces[i] = split["events"][i][1]
                events_rec.append({"k": k, "source": "fit_verbatim",
                                   "need": ev["need"], "budget": ev["budget"],
                                   "tokens": ev["need"]})
                continue
            if ev["zero_budget"]:
                failed = (k, "zero_budget")
                break
            log = []
            piece, source, meta = wc.compress_event(
                m, k, split["events"][i][1], ev["budget"], log)
            entry = {"k": k, "source": source, "need": ev["need"],
                     "budget": ev["budget"], "attempts": log}
            if piece is None:
                entry["outcome"] = "failed"
                events_rec.append(entry)
                failed = (k, "event_failed:" +
                          str(log[-1].get("why") if log else "unknown"))
                break
            entry["outcome"] = "ok"
            entry["tokens"] = len(wc.enc(piece))
            entry["piece"] = piece.strip()
            if m == "E-Extract":
                entry["piece_chars"] = len(piece)
                entry["span_ranges"] = meta.get("span_ranges")
            else:
                entry["evidence_ok"] = meta.get("evidence_ok")
                entry["evidence"] = meta.get("evidence")
            events_rec.append(entry)
            pieces[i] = piece
        if failed is not None:
            recs[m] = dict(base, scheme=m, outcome="fallback_d2",
                           reason=failed, events=events_rec)
            finals[m] = (rb["ids"], rb["mask"])
            continue

        events_str = ("Events: " + split["head"]
                      + "".join(split["events"][i][0] + pieces[i]
                                for i in range(b["n"])))
        ev_ids = wc.enc(events_str)
        if len(ev_ids) > b["E"]:
            recs[m] = dict(base, scheme=m, outcome="fallback_d2",
                           reason=f"assembly_overflow({len(ev_ids)}>{b['E']})",
                           events=events_rec)
            finals[m] = (rb["ids"], rb["mask"])
            continue
        counts_new = dict(rb["counts"])
        counts_new["Events"] = {"skipped": False, "raw_ids": ev_ids,
                                "raw_tokens": len(ev_ids)}
        ids, mask, meta_new = assemble(counts_new, rb["alloc"], tok)
        gates = {
            "events_len_le_E": len(ev_ids) <= b["E"],
            "content_le_510": sum(mask) - 2 <= 510,
            "non_events_bitwise": all(
                meta_new[n]["kept_ids"] == meta_d2[n]["kept_ids"]
                for n in ("Background", "Calendar", "Covariates")),
            "cls_sep_mask": (ids[0] == tok.cls_token_id
                             and mask[0] == 1 and sum(mask) == len(
                                 [x for x in mask if x == 1])
                             and ids[sum(mask) - 1] == tok.sep_token_id),
        }
        if not all(gates.values()):
            bad = [k2 for k2, v in gates.items() if not v]
            recs[m] = dict(base, scheme=m, outcome="fallback_d2",
                           reason="gate_fail:" + ",".join(bad),
                           events=events_rec)
            finals[m] = (rb["ids"], rb["mask"])
            continue
        recs[m] = dict(base, scheme=m, outcome="compressed", reason="ok",
                       events=events_rec, events_tokens=len(ev_ids))
        finals[m] = (ids, mask)
    return recs, finals, rb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_name", choices=("debug", "val"),
                    required=True)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import hashlib

    from text_cache import build_index
    tok = load_tokenizer()
    z, idx = trace_npz_index()
    sel = json.loads((OUT / "selected_windows.json").read_text())
    buds = json.loads((OUT / "budgets_all.json").read_text())["windows"][args.set_name]
    wins = sel[f"{args.set_name}_windows"]
    if args.limit:
        wins, buds = wins[:args.limit], buds[:args.limit]
    index = build_index()
    for w in wins:
        w["_fields"] = index[(w["var_key"], w["sample_id"])]["fields"]

    cfg = json.loads((HERE / "config.json").read_text())
    if args.offline:
        class _OfflineClient:
            service_version = None
            dropped_params = []

            def params_fingerprint(self):
                return "offline"
        client = _OfflineClient()
    else:
        from llm_client import LLMClient
        svc = cfg["llm_service"]
        client = LLMClient(svc["base_url"], svc["model"],
                           svc.get("temperature", 0), svc.get("seed", 2021),
                           svc.get("max_tokens", 1024),
                           svc.get("timeout_s", 120),
                           svc.get("transport_retries", 2))
        print(f"[service] chat_path={client.chat_path} "
              f"service_version={client.service_version!r} "
              f"dropped={client.dropped_params}")

    caches, prompt_parts, prompt_shas = {}, {}, {}
    prompt_files = {"E-Extract": "prompts/extract_v3.md",
                    "E-Summary": "prompts/summarize_v3.md"}
    for m in SCHEMES:
        caches[m] = LLMCache(OUT / "llm_cache.jsonl")
        pf = prompt_files[m]
        prompt_parts[m] = load_prompt_parts(pf)
        prompt_shas[m] = hashlib.sha256((HERE / pf).read_bytes()).hexdigest()

    wc = WindowCompressor(cfg, tok, client, caches, prompt_shas,
                          prompt_parts, args.offline)

    out_recs, ids_rows, mask_rows, methods = [], [], [], []
    for i, (w, b) in enumerate(zip(wins, buds)):
        recs, finals, rb = process_window(w, b, wc, tok, z, idx)
        for m in SCHEMES:
            out_recs.append(recs[m])
            r = out_recs[-1]
            print(f"[{args.set_name}] {w['var_key'][:40]}.. {m}: "
                  f"{r['outcome']} ({r['reason']})", flush=True)
        ids_rows.append(rb["ids"])
        mask_rows.append(rb["mask"])
        methods.append("D2")
        for m in SCHEMES:
            ids_rows.append(finals[m][0])
            mask_rows.append(finals[m][1])
            methods.append(m)

    import numpy as np
    tag = args.set_name
    np.savez(OUT / f"final_inputs_{tag}.npz",
             ids=np.asarray(ids_rows, dtype=np.int32),
             mask=np.asarray(mask_rows, dtype=np.int32),
             method=np.array(methods),
             var_keys=np.array([w["var_key"] for w in wins for _ in range(3)]),
             sample_ids=np.array([w["sample_id"] for w in wins
                                  for _ in range(3)]))
    with open(OUT / f"attempts_{tag}.jsonl", "w") as f:
        for r in out_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {}
    for r in out_recs:
        key = (r["scheme"], r["outcome"])
        summary[key] = summary.get(key, 0) + 1
    print("SUMMARY:", json.dumps({f"{s}|{o}": c for (s, o), c in
                                  sorted(summary.items())}, indent=1))
    run_meta = {
        "set": args.set_name, "offline": args.offline, "n_windows": len(wins),
        "prompt_shas": prompt_shas,
        "params_fingerprint": wc.client.params_fingerprint(),
        "service_version": wc.client.service_version,
        "dropped_params": wc.client.dropped_params,
        "cache": {m: {"hits": caches[m].hits, "misses": caches[m].misses}
                  for m in SCHEMES},
        "outcomes": {f"{s}|{o}": c for (s, o), c in sorted(summary.items())},
    }
    if not args.offline:
        run_meta["requests_made"] = client.requests_made
        run_meta["usage"] = client.usage
        run_meta["elapsed_s"] = round(client.elapsed_s, 2)
    (OUT / f"run_meta_{args.set_name}.json").write_text(
        json.dumps(run_meta, indent=1, ensure_ascii=False))
    if not args.offline:
        print(f"[service] requests={client.requests_made} "
              f"usage={client.usage} elapsed={client.elapsed_s:.1f}s "
              f"dropped={client.dropped_params}")
    print(f"cache hits/misses: " + "; ".join(
        f"{m}: {caches[m].hits}/{caches[m].misses}" for m in SCHEMES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
