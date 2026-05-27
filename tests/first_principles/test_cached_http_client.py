from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

import pytest

from src.first_principles.sources import CachedHttpClient, _hash_key


def _write_cache(cache_dir, url: str, params: dict, payload: dict) -> None:
    cache_key = _hash_key(f"{url}|{json.dumps(params, sort_keys=True)}")
    cache_path = cache_dir / f"{cache_key}.json"
    cache_path.write_text(json.dumps({"fetched_at_utc": datetime.utcnow().isoformat(), "data": payload}))


def test_cached_http_client_uses_stale_cache_on_fetch_failure(monkeypatch, tmp_path):
    client = CachedHttpClient(cache_dir=tmp_path, force_refresh=False)
    url = "https://example.test/data"
    params = {"a": 1}
    cached_payload = {"ok": True, "source": "cache"}
    _write_cache(tmp_path, url, params, cached_payload)

    cache_key = _hash_key(f"{url}|{json.dumps(params, sort_keys=True)}")
    cache_path = tmp_path / f"{cache_key}.json"
    old_ts = (datetime.utcnow() - timedelta(hours=48)).timestamp()
    os.utime(cache_path, (old_ts, old_ts))

    def _raise(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("src.first_principles.sources.requests.get", _raise)

    result = client.get_json(url, params=params, ttl_hours=1)
    assert result == cached_payload


def test_cached_http_client_raises_when_no_cache_and_fetch_fails(monkeypatch, tmp_path):
    client = CachedHttpClient(cache_dir=tmp_path, force_refresh=False)

    def _raise(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("src.first_principles.sources.requests.get", _raise)

    with pytest.raises(RuntimeError):
        client.get_json("https://example.test/no-cache", ttl_hours=1)
