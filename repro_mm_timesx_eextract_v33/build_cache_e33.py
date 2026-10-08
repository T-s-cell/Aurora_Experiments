#!/usr/bin/env python3
"""E-Extract v3.3 three-seed prediction experiment: build the frozen test text
cache (2,474 windows, E-Extract only) on theta.

Phases (run in order; every phase is resumable):
  --phase compress  LLM whole-sentence extraction per event, reusing the frozen
                    diagnostic v3.3 code path (compress.process_window with
                    SCHEMES pinned to ("E-Extract",)).
  --phase assemble  OFFLINE: re-assemble every window from its persisted record
                    (no LLM), enforce all assembly gates, write the npz.
  --phase freeze    write protocol_e33.json from outputs/build_report.json.

Terminal-event ledger (outputs/terminal_ledger_test.jsonl) is the authority on
per-event model-request budgets: each cache key gets AT MOST one terminal
record ("success" or "failed"); the per-event budget is one initial request +
at most 2 correction requests (model turns), persisted across restarts and
NEVER reset or regenerated. A terminal record is always reused as-is; there is
no last-wins overwrite. Transport-layer failures (tunnel blips) are NOT
terminal: the window is left unrecorded and retried on resume; the event's
turn budget is untouched because no model turn completed.

Window records (outputs/e33_windows_test.jsonl): one line per window with the
final outcome, per-event pieces/span ranges/attempt logs, and sha256 of the
final ids/mask row. A window is persisted only when its outcome is final
(compressed, or fallback where every failed event is terminal).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SUBDIR = Path(__file__).resolve().parent
PROJECT = SUBDIR.parent
for _p in (str(PROJECT), str(PROJECT / "analysis" / "event_compression_v1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import compress as diag_compress  # noqa: E402
from cache import LLMCache, cache_key  # noqa: E402
from ec_common import (HERE as DIAG_HERE, load_tokenizer, manifest_scope_sets,  # noqa: E402
                       rebuild_d2, split_events_clean, trace_npz_index,
                       verify_bitwise_vs_trace)

OUT = SUBDIR / "outputs"
CACHE_DIR = SUBDIR / "cache"
BUDGETS = OUT / "budgets_test.jsonl"
D2_ROWS = OUT / "d2_rows_test.npz"
RECORDS = OUT / "e33_windows_test.jsonl"
LEDGER = OUT / "terminal_ledger_test.jsonl"
LLM_CACHE_TEST = OUT / "llm_cache_test.jsonl"
TURN_LEDGER = OUT / "model_turns_test.jsonl"
DIAG_LLM_CACHE = DIAG_HERE / "outputs" / "llm_cache.jsonl"
STATE = OUT / "build_state.json"
BUILD_FP_FILES = ["build_budgets_e33.py", "build_cache_e33.py"]

PROMPT_REL = "prompts/extract_v3.md"
MAX_CONSECUTIVE_RETRYABLE = 50


class BuildStop(RuntimeError):
    pass


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def build_fingerprint(params_fp, service_version):
    h = hashlib.sha256()
    for name in BUILD_FP_FILES:
        h.update((SUBDIR / name).read_bytes())
    h.update((DIAG_HERE / "compress.py").read_bytes())
    h.update((DIAG_HERE / "ec_common.py").read_bytes())
    h.update((DIAG_HERE / "cache.py").read_bytes())
    h.update((DIAG_HERE / "llm_client.py").read_bytes())
    h.update((DIAG_HERE / "prompts" / "extract_v3.md").read_bytes())
    h.update(BUDGETS.read_bytes())
    h.update(json.dumps({"params_fp": params_fp,
                         "service_version": service_version},
                        sort_keys=True).encode())
    return h.hexdigest()


def row_sha(ids, mask):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(ids, dtype=np.int32).tobytes())
    h.update(np.ascontiguousarray(mask, dtype=np.int32).tobytes())
    return h.hexdigest()


class TerminalLedger:
    """key -> terminal record, first-wins, immutable once written."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.map = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    assert r["key"] not in self.map, \
                        f"duplicate terminal key {r['key'][:24]}…"
                    self.map[r["key"]] = r
        self.n_success = sum(1 for r in self.map.values()
                             if r["status"] == "success")
        self.n_failed = len(self.map) - self.n_success

    def get(self, key):
        return self.map.get(key)

    def put(self, key, rec):
        assert key not in self.map, "terminal records are immutable"
        self.map[key] = rec
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if rec["status"] == "success":
            self.n_success += 1
        else:
            self.n_failed += 1


class TurnLedger:
    """Crash-safe reservation count for model turns; max three per cache key."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.counts = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    key = r["key"]
                    self.counts.setdefault(key, 0)
                    if "refund" in r:
                        assert self.counts[key] == r["refund"]
                        self.counts[key] -= 1
                    else:
                        assert r["turn"] == self.counts[key] + 1
                        self.counts[key] = r["turn"]

    def refund(self, key):
        used = self.counts.get(key, 0)
        assert used > 0, "cannot refund an unreserved turn"
        turn = used
        rec = {"key": key, "refund": turn,
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.counts[key] = used - 1

    def reserve(self, key):
        used = self.counts.get(key, 0)
        if used >= 3:
            return False
        turn = used + 1
        rec = {"key": key, "turn": turn,
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.counts[key] = turn
        return True


class LedgerWindowCompressor(diag_compress.WindowCompressor):
    """compress_event wrapped with terminal and persistent-turn ledgers."""

    def __init__(self, *a, ledger=None, turns=None, **kw):
        super().__init__(*a, **kw)
        self.ledger = ledger
        self.turns = turns
        self._active_key = None
        self._turn_http_start = 0
        self._turn_http_count = 0
        self._chat_messages = self.client.chat_messages
        self.client.chat_messages = self._chat_with_turn_ledger

    def _chat_with_turn_ledger(self, *args, **kwargs):
        if not self.turns.reserve(self._active_key):
            return {"ok": False, "error": "turn budget exhausted (3 model turns)"}
        self._turn_http_start = self.client.requests_made
        resp = self._chat_messages(*args, **kwargs)
        self._turn_http_count += self.client.requests_made - self._turn_http_start
        if not resp.get("ok") and str(resp.get("error", "")).startswith(
                "transport failed"):
            self.turns.refund(self._active_key)
            self._turn_http_count = max(0, self._turn_http_count - 1)
        return resp

    def compress_event(self, method, k, prose_piece, budget, attempts_log):
        prose = prose_piece.strip()
        key = cache_key(prose, method, budget,
                        self.prompt_shas[method],
                        self.client.params_fingerprint(),
                        self.client.service_version)
        led = self.ledger.get(key)
        if led is not None:
            if led["status"] == "success":
                merged = [tuple(r) for r in led["span_ranges"]]
                text = diag_compress.assemble_spans(prose, merged)
                qd = diag_compress.clause_gate(prose, merged)
                n_tok = len(self.enc(text))
                if qd is None and n_tok <= budget \
                        and not diag_compress.REFUSAL_RE.search(text):
                    attempts_log.append({"k": k, "source": "terminal",
                                         "ok": True,
                                         "attempts_used": led["attempts"]})
                    return text, "cache", {"cache_key": key,
                                           "span_ranges": led["span_ranges"],
                                           "ledger_reuse": True}
                raise BuildStop(
                    f"terminal success for key {key[:24]}… fails the frozen "
                    f"gate (qd={qd}, tokens={n_tok}>{budget}?); stop and report")
            attempts_log.append({"k": k, "source": "terminal", "ok": False,
                                 "why": led["why"],
                                 "attempts_used": led["attempts"]})
            return None, "failed", {"cache_key": key,
                                    "attempts_used": led["attempts"]}

        turns_before = self.turns.counts.get(key, 0)
        self._turn_http_count = 0
        self._active_key = key
        try:
            piece, source, meta = super().compress_event(
                method, k, prose_piece, budget, attempts_log)
        finally:
            self._active_key = None
        turns = self.turns.counts.get(key, 0) - turns_before
        http_requests = self._turn_http_count
        if piece is not None:
            self.ledger.put(key, {
                "key": key, "status": "success", "source": source,
                "attempts": turns,
                "piece": piece.strip(),
                "span_ranges": meta.get("span_ranges"),
                "http_requests": http_requests,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
            return piece, source, meta
        # classify the failure
        last = attempts_log[-1] if attempts_log else {}
        why = str(last.get("why", "unknown"))
        if why.startswith("transport failed"):
            return None, "failed_retryable", {"cache_key": key, "why": why}
        if why.startswith("auth rejected"):
            raise BuildStop(f"auth rejected by service: {why}")
        if why.startswith("HTTP"):
            raise BuildStop(f"service HTTP error: {why}")
        self.ledger.put(key, {
            "key": key, "status": "failed", "attempts": turns, "why": why,
            "http_requests": http_requests,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        return None, "failed", {"cache_key": key, "attempts_used": turns}


def load_budget_records():
    recs = []
    with open(BUDGETS, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    assert len(recs) == 2474
    keys = [(r["var_key"], r["sample_id"]) for r in recs]
    assert keys == sorted(keys) and len(set(keys)) == 2474
    return recs


def load_records_map():
    m = {}
    if RECORDS.exists():
        with open(RECORDS, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                m[(r["var_key"], r["sample_id"])] = r
    return m


def setup_service(cfg):
    from llm_client import LLMClient
    svc = cfg["llm_service"]
    client = LLMClient(svc["base_url"], svc["model"],
                       svc.get("temperature", 0), svc.get("seed", 2021),
                       svc.get("max_tokens", 1024),
                       svc.get("timeout_s", 120),
                       svc.get("transport_retries", 2))
    return client


def phase_compress():
    from text_cache import build_index

    cfg = json.loads((DIAG_HERE / "config.json").read_text())
    tok = load_tokenizer()
    z, idx = trace_npz_index()
    budget_recs = load_budget_records()
    done = load_records_map()

    state = {}
    if STATE.exists():
        state = json.loads(STATE.read_text())
    client = setup_service(cfg)
    print(f"[service] chat_path={client.chat_path} "
          f"service_version={client.service_version!r} "
          f"dropped={client.dropped_params}", flush=True)
    if "service_version" in state and \
            state["service_version"] != str(client.service_version):
        raise BuildStop(
            f"service_version changed across resumes: "
            f"{state['service_version']!r} -> {client.service_version!r}")
    fp = build_fingerprint(client.params_fingerprint(),
                           str(client.service_version))
    state.update({"service_version": str(client.service_version),
                  "params_fp": client.params_fingerprint(),
                  "build_fp": fp,
                  "model": cfg["llm_service"]["model"]})
    STATE.write_text(json.dumps(state, indent=2))

    caches = LLMCache(LLM_CACHE_TEST)
    if DIAG_LLM_CACHE.exists():
        seeded = 0
        with open(DIAG_LLM_CACHE, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["key"] not in caches.map and \
                        r.get("method") == "E-Extract":
                    caches.map[r["key"]] = r
                    seeded += 1
        print(f"[cache] seeded {seeded} E-Extract entries from the diagnostic "
              f"cache (read-only)", flush=True)

    prompt_shas = {"E-Extract": hashlib.sha256(
        (DIAG_HERE / PROMPT_REL).read_bytes()).hexdigest()}
    prompt_parts = {"E-Extract": diag_compress.load_prompt_parts(PROMPT_REL)}
    ledger = TerminalLedger(LEDGER)
    turns = TurnLedger(TURN_LEDGER)
    wc = LedgerWindowCompressor(cfg, tok, client,
                                {"E-Extract": caches}, prompt_shas,
                                prompt_parts, False, ledger=ledger,
                                turns=turns)
    diag_compress.SCHEMES = ("E-Extract",)

    index = build_index()
    current = sum(1 for r in budget_recs
                  if done.get((r["var_key"], r["sample_id"]), {})
                  .get("build_fp") == fp)
    todo = [r for r in budget_recs
            if done.get((r["var_key"], r["sample_id"]), {})
            .get("build_fp") != fp]
    print(f"[compress] records on disk: {len(done)} "
          f"(current fingerprint: {current}); to process: {len(todo)}; "
          f"ledger S/F={ledger.n_success}/{ledger.n_failed}", flush=True)

    consecutive_retryable = 0
    t0 = time.time()
    for n, b in enumerate(todo):
        key = (b["var_key"], b["sample_id"])
        w = {"var_key": b["var_key"], "sample_id": b["sample_id"],
             "domain": b["domain"], "scope": "test", "freq": b["freq"],
             "_fields": index[key]["fields"]}
        recs, finals, _rb = diag_compress.process_window(w, b, wc, tok, z, idx)
        rec = recs["E-Extract"]
        retryable = any(ev.get("source") == "failed_retryable"
                        for ev in rec.get("events", []))
        if retryable:
            consecutive_retryable += 1
            print(f"[compress] {key[0][:36]}..|{key[1]} RETRYABLE "
                  f"(transport); not recorded", flush=True)
            if consecutive_retryable >= MAX_CONSECUTIVE_RETRYABLE:
                raise BuildStop(f"{consecutive_retryable} consecutive windows "
                                f"hit transport failures; tunnel/service down?")
            continue
        consecutive_retryable = 0
        rec.update({"build_fp": fp,
                    "service_version": str(client.service_version),
                    "params_fp": client.params_fingerprint(),
                    "prompt_sha": prompt_shas["E-Extract"],
                    "ids_sha256": row_sha(finals["E-Extract"][0],
                                          finals["E-Extract"][1]),
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        with open(RECORDS, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
        if (n + 1) % 20 == 0 or n + 1 == len(todo):
            dt = time.time() - t0
            print(f"[compress] {n + 1}/{len(todo)} done in {dt / 60:.1f}min "
                  f"({dt / (n + 1):.1f}s/win) req={client.requests_made} "
                  f"ledger S/F={ledger.n_success}/{ledger.n_failed}",
                  flush=True)

    done = load_records_map()
    missing = [k for k in ((r["var_key"], r["sample_id"])
                           for r in budget_recs) if k not in done]
    if missing:
        raise BuildStop(f"{len(missing)} windows still unrecorded (retryable "
                        f"windows remain); rerun --phase compress")
    print("[compress] all 2474 windows recorded", flush=True)
    return 0


def phase_assemble():
    from allocate import assemble

    tok = load_tokenizer()
    z, idx = trace_npz_index()
    budget_recs = load_budget_records()
    b_by_key = {(r["var_key"], r["sample_id"]): r for r in budget_recs}
    records = load_records_map()
    assert len(records) == 2474, f"{len(records)} records != 2474"

    zd = np.load(D2_ROWS, allow_pickle=False)
    d2_index = {(str(v), str(s)): i for i, (v, s) in
                enumerate(zip(zd["var_keys"], zd["sample_ids"]))}
    from text_cache import build_index
    index = build_index()

    ids_rows = np.zeros((2474, 512), dtype=np.int32)
    mask_rows = np.zeros((2474, 512), dtype=np.int32)
    outcomes = {"compressed": 0, "fallback_d2": 0, "kept_d2": 0}
    changed_keys = []
    for i, b in enumerate(budget_recs):
        key = (b["var_key"], b["sample_id"])
        rec = records[key]
        assert rec["scope"] == "test" and rec["E"] == b["E"]
        rb = rebuild_d2(index[key]["fields"], tok)
        assert verify_bitwise_vs_trace(rb["ids"], rb["mask"], key[0], key[1],
                                       z, idx)
        assert np.array_equal(rb["ids"], zd["ids"][d2_index[key]]) and \
            np.array_equal(rb["mask"], zd["mask"][d2_index[key]])

        if rec["outcome"] in ("kept_d2", "fallback_d2"):
            ids, mask = rb["ids"], rb["mask"]
        else:
            assert rec["outcome"] == "compressed", rec["outcome"]
            split = split_events_clean(rb["cleaned"]["scenario"], tok)
            assert split["ok"] and len(split["events"]) == b["n"]
            pieces = {}
            for ev in rec["events"]:
                j = ev["k"] - 1
                if ev["source"] == "fit_verbatim":
                    pieces[j] = split["events"][j][1]
                else:
                    pieces[j] = " " + ev["piece"]
            assert len(pieces) == b["n"]
            events_str = ("Events: " + split["head"]
                          + "".join(split["events"][j][0] + pieces[j]
                                    for j in range(b["n"])))
            ev_ids = tok.encode(events_str, add_special_tokens=False)
            assert len(ev_ids) <= b["E"], f"{key}: Events {len(ev_ids)} > E={b['E']}"
            assert len(ev_ids) == rec["events_tokens"]
            counts_new = dict(rb["counts"])
            counts_new["Events"] = {"skipped": False, "raw_ids": ev_ids,
                                    "raw_tokens": len(ev_ids)}
            ids, mask, meta_new = assemble(counts_new, rb["alloc"], tok)
            gates = {
                "content_le_510": sum(mask) - 2 <= 510,
                "non_events_bitwise": all(
                    meta_new[n]["kept_ids"] == rb["meta"][n]["kept_ids"]
                    for n in ("Background", "Calendar", "Covariates")),
                "cls_sep_mask": (ids[0] == tok.cls_token_id and mask[0] == 1
                                 and ids[sum(mask) - 1] == tok.sep_token_id),
            }
            assert all(gates.values()), f"{key}: gates {gates}"
        assert row_sha(ids, mask) == rec["ids_sha256"], \
            f"{key}: re-assembled row sha != recorded sha"
        ids_rows[i] = ids
        mask_rows[i] = mask
        outcomes[rec["outcome"]] += 1
        if not (np.array_equal(ids, zd["ids"][d2_index[key]])
                and np.array_equal(mask, zd["mask"][d2_index[key]])):
            changed_keys.append(list(key))
        if (i + 1) % 200 == 0:
            print(f"[assemble] {i + 1}/2474", flush=True)

    CACHE_DIR.mkdir(exist_ok=True)
    npz = CACHE_DIR / "text_tokens_M48T512_E33.npz"
    tmp = Path(str(npz) + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, ids=ids_rows, mask=mask_rows,
                 var_keys=np.array([r["var_key"] for r in budget_recs],
                                   dtype=np.str_),
                 sample_ids=np.array([r["sample_id"] for r in budget_recs],
                                     dtype=np.str_),
                 domains=np.array([r["var_key"].split("__", 1)[0]
                                   for r in budget_recs], dtype=np.str_))
    tmp.replace(npz)
    zchk = np.load(npz, allow_pickle=False)
    assert np.array_equal(zchk["ids"], ids_rows)
    assert zchk["ids"].dtype == np.int32

    rec_copy = CACHE_DIR / "e33_windows_test.jsonl"
    with open(rec_copy, "w", encoding="utf-8") as f:
        for b in budget_recs:
            f.write(json.dumps(records[(b["var_key"], b["sample_id"])],
                               ensure_ascii=False) + "\n")

    report = {
        "n_windows": 2474,
        "outcomes": outcomes,
        "windows_input_changed_vs_d2": len(changed_keys),
        "npz_sha256": sha256_bytes(npz.read_bytes()),
        "records_sha256": sha256_bytes(rec_copy.read_bytes()),
        "ledger_sha256": sha256_bytes(LEDGER.read_bytes()) if LEDGER.exists() else None,
        "ledger_counts": {"success": ledger_counts()[0],
                          "failed": ledger_counts()[1]},
        "budgets_test_jsonl_sha256": sha256_bytes(BUDGETS.read_bytes()),
        "d2_rows_test_npz_sha256": sha256_bytes(D2_ROWS.read_bytes()),
        "prompt_extract_v3_sha256": sha256_bytes(
            (DIAG_HERE / "prompts" / "extract_v3.md").read_bytes()),
        "compress_py_sha256": sha256_bytes((DIAG_HERE / "compress.py").read_bytes()),
        "source_hashes": {
            name: sha256_bytes((SUBDIR / name).read_bytes())
            for name in ("build_budgets_e33.py", "build_cache_e33.py",
                         "predict_e33.py", "run_eval_e33.py",
                         "run_preflight_e33.py", "aggregate_e33.py",
                         "wrapper_theta.sh", "README.md")
        } | {
            f"analysis/event_compression_v1/{name}": sha256_bytes(
                (DIAG_HERE / name).read_bytes())
            for name in ("compress.py", "ec_common.py", "cache.py", "llm_client.py",
                         "config.json", "prompts/extract_v3.md")
        },
        "build_fp": json.loads(STATE.read_text())["build_fp"],
        "service_version": json.loads(STATE.read_text())["service_version"],
        "params_fp": json.loads(STATE.read_text())["params_fp"],
        "assembled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (OUT / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "build_fp"},
                     indent=1))
    return 0


def ledger_counts():
    n_s = n_f = 0
    if LEDGER.exists():
        with open(LEDGER, encoding="utf-8") as f:
            for line in f:
                if json.loads(line)["status"] == "success":
                    n_s += 1
                else:
                    n_f += 1
    return n_s, n_f


def phase_freeze():
    report = json.loads((OUT / "build_report.json").read_text())
    d2p = json.loads((PROJECT / "repro_mm_timesx_d2" / "protocol_d2.json").read_text())
    proto = {
        "protocol_name": "aurora-timesx-mm-eextract-v33-v1",
        "frozen": time.strftime("%Y-%m-%d"),
        "experiment": "EXP-013",
        "purpose": "Three-seed test-set prediction with the frozen E-Extract "
                   "v3.3 whole-sentence Events input (2,474 windows, 19 "
                   "domains), controlled against EXP-012 M48T512_D2. ONLY the "
                   "Events token ids/mask change; model, weights, windows, "
                   "seeds, itl, scoring and aggregation are EXP-012-identical.",
        "scope_statement": "This round evaluates the fixed E-Extract v3.3 "
                           "input scheme AS A WHOLE. The diagnostic "
                           "(analysis-event-compression-v1 @ 8fe2320) measured "
                           "coverage separately: val 27/57 windows compressed, "
                           "event entry rate 58.8%, model screening did not "
                           "flag distortion (screening is not an absolute "
                           "fidelity proof). Higher text coverage does NOT "
                           "imply better prediction.",
        "declaration": "Aurora is NOT trained or fine-tuned on TimesX. No "
                       "model modification and no hyperparameter search: the "
                       "E-Extract v3.3 prompt, whole-sentence gate, budgets, "
                       "terminal-event policy and all inference settings were "
                       "frozen a priori; no rule may be re-tuned on this "
                       "round's test results.",
        "data": d2p["data"],
        "model": d2p["model"],
        "inference": d2p["inference"],
        "text": {
            "method_tag": "M48T512_E33",
            "definition": "E-Extract v3.3 = frozen D2 cleanup, per-event "
                          "budgets from original lengths (E = window's D2 "
                          "Events alloc), whole-sentence verbatim span "
                          "selection via the frozen extract_v3 prompt + "
                          "clause gate (abbreviation-aware boundaries), "
                          "' ... ' separator for non-adjacent sentences, "
                          "Events quota = E, freed budget to end padding; "
                          "Background/Calendar/Covariates tokens bitwise "
                          "equal to D2. Any event final failure => window "
                          "falls back to the frozen D2 input.",
            "source_hashes": report["source_hashes"],
            "v33_provenance": {
                "source_branch": "analysis-event-compression-v1",
                "source_commit": "8fe232098181c3f158596dd526ee890584fc39e5",
                "diagnostic_report": "analysis/event_compression_v1/results/report.md",
                "prompt": "prompts/extract_v3.md",
                "prompt_sha256": report["prompt_extract_v3_sha256"],
                "compress_py_sha256": report["compress_py_sha256"],
                "llm_params_fp": report["params_fp"],
                "service_version": report["service_version"],
                "terminal_ledger": "per cache key, model request budget "
                                   "is one initial + at most two corrections, "
                                   "persisted across restarts; terminal "
                                   "success and terminal failure are both "
                                   "immutable and always reused",
                "llm_service": "http://localhost:8020 on theta via SSH tunnel; "
                               "key via TIMESX_LLM_API_KEY env only",
            },
            "tokenizer": d2p["text"]["tokenizer"],
            "cache": {
                "npz": "repro_mm_timesx_eextract_v33/cache/text_tokens_M48T512_E33.npz",
                "npz_sha256": report["npz_sha256"],
                "records_jsonl": "repro_mm_timesx_eextract_v33/cache/e33_windows_test.jsonl",
                "records_jsonl_sha256": report["records_sha256"],
                "budgets_jsonl_sha256": report["budgets_test_jsonl_sha256"],
                "d2_rows_npz_sha256": report["d2_rows_test_npz_sha256"],
                "outcomes": report["outcomes"],
                "windows_input_changed_vs_d2": report["windows_input_changed_vs_d2"],
                "schema": "ids int32 [2474,512], mask int32 [2474,512], "
                          "explicit key arrays var_keys/sample_ids/domains; "
                          "one record line per npz row; composite-key sorted",
                "provenance": "extracted and verified against trace_D2.npz and "
                              "EXP-012/EXP-011 frozen text cache keys",
            },
        },
        "seeds": d2p["seeds"],
        "determinism": d2p["determinism"],
        "scoring": {
            "window_d": d2p["scoring"]["window_d"],
            "std_mse": d2p["scoring"]["std_mse"],
            "std_mae": d2p["scoring"]["std_mae"],
            "paired_delta_ladder": "PAIRED COMPARISON = M48T512_E33 - "
                                   "M48T512_D2 (same seed, same composite "
                                   "key): window-level paired difference -> "
                                   "mean per variable -> equal weight within "
                                   "domain -> equal weight across 19 domains "
                                   "-> overall. NEVER a plain average over "
                                   "all windows. Negative delta = E-Extract "
                                   "improves. All 2,474 windows stay in the "
                                   "main results; no seed selection; no rule "
                                   "tuning on test error.",
        },
        "shard_fingerprint": {
            "per_shard_header": d2p["shard_fingerprint"]["per_shard_header"],
            "resume_policy": "On resume, every header field is re-verified "
                             "against the current protocol; ANY mismatch -> "
                             "shard moved to predictions/quarantine/ (never "
                             "deleted) and re-generated.",
            "done_marker": d2p["shard_fingerprint"]["done_marker"],
        },
        "outputs": {
            "npz_schema": "sample_ids/var_keys/domains (str), pred(N,12 float64), "
                           "target(N,12 float64), d(N, float64), sharded by "
                           "(base_seed, domain) under predictions/ with "
                           "method tag M48T512_E33",
            "aggregate": "repro_mm_timesx_eextract_v33/results/: per-seed, "
                         "variable-level, 19-domain and overall "
                         "(mean +/- std ddof=1) for M48T512_E33 and "
                         "M48T512_D2 (recomputed at full precision from the "
                         "frozen EXP-012 shards); paired-difference tables "
                         "E33 - D2 per the frozen ladder; fallback-window "
                         "bitwise pred equality vs D2 (same seed) validated; "
                         "input-change ratio reported",
            "coverage_check": d2p["outputs"]["coverage_check"],
        },
        "eval_gate": {
            "approval_flag": "--i-approve-frozen-protocol <first 12 hex of "
                             "sha256(protocol_e33.json)>; the EXP-010 "
                             "(580eb3c3f51d), EXP-011 (74b08ecac1ed) and "
                             "EXP-012 (7220a9aff32b) tokens MUST be rejected",
            "runtime_checks": d2p["eval_gate"]["runtime_checks"],
            "status": "official full run (2,474 windows x seeds "
                      "2021/2022/2023) pre-authorized by the user AFTER all "
                      "preflight sections pass on this frozen configuration",
        },
        "e33_provenance": {
            "build_report_sha256": sha256_bytes(
                (OUT / "build_report.json").read_bytes()),
            "d2_baseline_protocol": "repro_mm_timesx_d2/protocol_d2.json",
            "d2_protocol_sha256": sha256_bytes(
                (PROJECT / "repro_mm_timesx_d2" / "protocol_d2.json").read_bytes()),
        },
    }
    text = json.dumps(proto, indent=2, ensure_ascii=False)
    (SUBDIR / "protocol_e33.json").write_text(text)
    token = hashlib.sha256((SUBDIR / "protocol_e33.json").read_bytes()).hexdigest()[:12]
    print(f"[freeze] protocol_e33.json written; approval token = {token}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=("compress", "assemble", "freeze"))
    args = ap.parse_args()
    try:
        if args.phase == "compress":
            return phase_compress()
        if args.phase == "assemble":
            return phase_assemble()
        return phase_freeze()
    except BuildStop as e:
        print(f"[build_cache_e33] STOP: {e}", flush=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())
