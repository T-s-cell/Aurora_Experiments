"""TimesX frozen-protocol data access (read-only copy of EXP-004 dcd2425 logic).

Window identity: composite key (var_key, sample_id) where sample_id is the
native id string. Uniqueness is required on the composite key only.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "data"

CTX, PRED, STD_EPS = 96, 12, 1e-8

EXPECTED_TEST_TOTAL = 2474
EXPECTED_VARS = 190
EXPECTED_DOMAINS = 19

DOMAIN_NAMES = sorted([
    'CropsAndStaples', 'Currency', 'EnergyAndFuels', 'LivestockAndFoodProducts',
    'RawMaterialsAndConstruction', 'SpecialtyAndAdvancedMaterials',
    'StrategicAndHighValueMaterials', 'arts', 'climate', 'economy',
    'electronic_technology', 'finance', 'pets', 'public_health',
    'public_policy', 'science', 'shopping', 'society', 'traffic',
])
assert len(DOMAIN_NAMES) == EXPECTED_DOMAINS


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_data_fingerprints(protocol=None):
    protocol = protocol or json.loads((PROJECT / "protocol.json").read_text())
    d = protocol["data"]
    checks = {
        "split_manifest": (DATA / "split_manifest.json", d["split_manifest_sha256"]),
        "data_cache": (DATA / "data_cache.npz", d["data_cache_sha256"]),
        "z0_reference": (DATA / "Z0__test.npz", d["z0_reference_sha256"]),
    }
    out = {}
    for name, (p, expect) in checks.items():
        got = sha256_file(p)
        out[name] = {"path": str(p), "sha256": got, "expected": expect, "ok": got == expect}
    return out


def window_d(x, fb_std):
    s = float(np.std(x))
    return s if s >= STD_EPS else fb_std


class TimesXData:
    """Read-only view over data_cache.npz + split_manifest.json."""

    def __init__(self, data_dir=DATA):
        data_dir = Path(data_dir)
        with open(data_dir / "split_manifest.json") as f:
            self.manifest = json.load(f)
        cache = np.load(data_dir / "data_cache.npz", allow_pickle=False)
        self._keys = [str(k) for k in cache["var_keys"]]
        self._series_off = cache["series_off"]
        self._series_len = cache["series_len"]
        self._series = cache["series"]

        assert len(self._keys) == EXPECTED_VARS, f"var count {len(self._keys)} != {EXPECTED_VARS}"
        assert len({v["domain"] for v in self.manifest["variables"].values()}) == EXPECTED_DOMAINS

        self.domain_vars = {d: sorted(vk for vk in self._keys
                                      if self.manifest["variables"][vk]["domain"] == d)
                            for d in DOMAIN_NAMES}
        self.fb_std = {vk: self.manifest["variables"][vk]["fallback_std"] for vk in self._keys}

    def series(self, vk):
        i = self._keys.index(vk)
        o, n = int(self._series_off[i]), int(self._series_len[i])
        return self._series[o:o + n]

    def start_of(self, vk, sample_id):
        return self.manifest["variables"][vk]["sample_start_idx"][sample_id]

    def native_ids(self, vk, split):
        return self.manifest["variables"][vk]["native"][split]

    def window(self, vk, s):
        ser = self.series(vk)
        return ser[s:s + CTX], ser[s + CTX:s + CTX + PRED]

    def split_window_by_id(self, vk, sample_id):
        return self.window(vk, self.start_of(vk, sample_id))

    def d_of(self, vk, sample_id):
        x, _ = self.split_window_by_id(vk, sample_id)
        return window_d(x, self.fb_std[vk])

    def test_rows(self, domain=None):
        """[(var_key, sample_id, domain, past, target, d)] over the 2474 test windows,
        deterministic order (DOMAIN_NAMES order, sorted var within domain, native test order)."""
        domains = [domain] if domain else DOMAIN_NAMES
        rows, seen = [], set()
        for d in domains:
            for vk in self.domain_vars[d]:
                for sid in self.native_ids(vk, "test"):
                    key = (vk, sid)
                    if key in seen:
                        raise AssertionError(f"duplicate composite key in test split: {key}")
                    seen.add(key)
                    x, y = self.split_window_by_id(vk, sid)
                    rows.append((vk, sid, d, x, y, self.d_of(vk, sid)))
        assert len(rows) == EXPECTED_TEST_TOTAL, f"test rows {len(rows)} != {EXPECTED_TEST_TOTAL}"
        return rows

    def trainval_rows(self, domain, max_per_var=None, splits=("train", "val")):
        """Rows from train/val windows only (never test) for preflight use."""
        rows = []
        for vk in self.domain_vars[domain]:
            ids = []
            for sp in splits:
                ids.extend(self.native_ids(vk, sp))
            if max_per_var:
                ids = ids[:max_per_var]
            for sid in ids:
                x, y = self.split_window_by_id(vk, sid)
                rows.append((vk, sid, domain, x, y, self.d_of(vk, sid)))
        return rows
