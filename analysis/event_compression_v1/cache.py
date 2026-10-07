#!/usr/bin/env python3
"""Append-only JSONL cache for LLM compression results.

Key = sha256(event_text) + method + str(event_budget) + prompt_sha + model
params fingerprint + service version (or null). Values store the raw model
JSON payload fields WITHOUT any event_id binding — the consumer rebinds
event_id to the current window's numbering on reuse.
"""
import hashlib
import json
from pathlib import Path


def cache_key(event_text, method, budget, prompt_sha, params_fp,
              service_version):
    h = hashlib.sha256(event_text.encode("utf-8")).hexdigest()
    return "|".join([h, method, str(budget), prompt_sha, params_fp,
                     str(service_version)])


class LLMCache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.map = {}
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    r = json.loads(line)
                    self.map[r["key"]] = r
        self.hits = 0
        self.misses = 0

    def get(self, key):
        r = self.map.get(key)
        if r is not None:
            self.hits += 1
        else:
            self.misses += 1
        return r

    def put(self, key, method, payload):
        rec = {"key": key, "method": method, "payload": payload}
        self.map[key] = rec
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
