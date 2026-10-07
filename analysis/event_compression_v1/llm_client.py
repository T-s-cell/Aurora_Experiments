#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat client for the zeta service (theta-side).

Key from env TIMESX_LLM_API_KEY only. Route discovery: GET /v1/models then
/models; chat path = <prefix>/chat/completions. Unsupported params (seed,
response_format) are dropped on 400 and recorded, never silently swallowed.
finish_reason != 'stop' => output marked truncated (never treated as
complete). Transport errors: timeout + at most 2 retries. Logs never contain
the Authorization header or the key.
"""
import json
import os
import time

import requests


class LLMClient:
    def __init__(self, base_url, model, temperature=0, seed=2021,
                 max_tokens=1024, timeout_s=120, transport_retries=2):
        self.base = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.timeout = timeout_s
        self.retries = transport_retries
        self.dropped_params = []
        self.service_version = None
        self.requests_made = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0}
        self.elapsed_s = 0.0
        self.key = os.environ.get("TIMESX_LLM_API_KEY")
        if not self.key:
            raise RuntimeError(
                "TIMESX_LLM_API_KEY not set; inject it via the environment "
                "(e.g. source a 0600 file outside the repo)")
        self.prefix = self._discover_prefix()
        self.chat_path = f"{self.prefix}/chat/completions"

    def _headers(self):
        return {"Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json"}

    def _discover_prefix(self):
        for prefix in ("/v1", ""):
            try:
                r = requests.get(self.base + prefix + "/models",
                                 headers=self._headers(), timeout=15)
                if r.status_code == 200:
                    try:
                        data = r.json()
                        self.service_version = str(
                            data.get("version", r.headers.get("server")))
                    except Exception:
                        self.service_version = r.headers.get("server")
                    return prefix
                if r.status_code in (401, 403):
                    raise RuntimeError(
                        f"auth rejected by service (HTTP {r.status_code}); "
                        "check TIMESX_LLM_API_KEY")
            except requests.RequestException:
                continue
        raise RuntimeError(
            f"service unreachable at {self.base} (both /v1/models and "
            "/models failed); is the SSH tunnel up?")

    def chat(self, system, user, force_json=True):
        return self.chat_messages([{"role": "system", "content": system},
                                   {"role": "user", "content": user}],
                                  force_json=force_json)

    def chat_messages(self, messages, force_json=True):
        """One completion over an explicit message list (correction turns are
        appended by the caller). Returns dict(ok, content, finish_reason,
        model, usage, meta, truncated). Transport failures after retries
        return ok=False; auth/HTTP-400 handled inside."""
        body = {"model": self.model, "messages": messages,
                "temperature": self.temperature, "max_tokens": self.max_tokens}
        if force_json:
            body["response_format"] = {"type": "json_object"}
        if self.seed is not None:
            body["seed"] = self.seed
        return self._post(body)

    def _post(self, body):
        meta = {"dropped": [], "transport_retries_used": 0, "http_status": None}
        order = [dict(body)]
        last_err = None
        for attempt in range(self.retries + 1):
            payload = order[-1]
            try:
                t0 = time.time()
                r = requests.post(self.base + self.chat_path,
                                  headers=self._headers(), json=payload,
                                  timeout=self.timeout)
                self.elapsed_s += time.time() - t0
                self.requests_made += 1
                meta["http_status"] = r.status_code
                if r.status_code == 400:
                    dropped = self._drop_param_on_400(payload, r)
                    if dropped:
                        meta["dropped"].append(dropped)
                        if dropped not in self.dropped_params:
                            self.dropped_params.append(dropped)
                        order.append(payload)
                        continue
                    return {"ok": False, "error": f"HTTP 400: {r.text[:300]}",
                            "meta": meta}
                if r.status_code in (401, 403):
                    return {"ok": False,
                            "error": f"auth rejected (HTTP {r.status_code})",
                            "meta": meta}
                r.raise_for_status()
                data = r.json()
                choice = data["choices"][0]
                content = choice["message"]["content"]
                finish = choice.get("finish_reason")
                usage = data.get("usage") or {}
                for k in self.usage:
                    self.usage[k] += usage.get(k, 0) or 0
                return {"ok": True, "content": content, "finish_reason": finish,
                        "model": data.get("model", self.model),
                        "usage": usage, "meta": meta,
                        "truncated": finish not in (None, "stop")}
            except requests.RequestException as e:
                last_err = type(e).__name__
                meta["transport_retries_used"] = attempt + 1
                if attempt < self.retries:
                    time.sleep(2 * (attempt + 1))
                    continue
        return {"ok": False,
                "error": f"transport failed after retries ({last_err})",
                "meta": meta}

    def _drop_param_on_400(self, payload, resp):
        for name in ("seed", "response_format"):
            if name in payload:
                txt = resp.text[:400]
                if name in txt or "unsupported" in txt or "unknown" in txt \
                        or "invalid" in txt:
                    payload.pop(name)
                    return name
        # unknown 400: drop response_format then seed as a blind fallback
        if "response_format" in payload:
            payload.pop("response_format")
            return "response_format"
        if "seed" in payload:
            payload.pop("seed")
            return "seed"
        return None

    def params_fingerprint(self):
        return json.dumps({"model": self.model,
                           "temperature": self.temperature,
                           "seed": self.seed if "seed"
                           not in self.dropped_params else None,
                           "max_tokens": self.max_tokens,
                           "json_mode": "response_format"
                           not in self.dropped_params},
                          sort_keys=True)
