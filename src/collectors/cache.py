"""Local parquet cache, rate limiting and HTTP retry helpers shared by all collectors.

Every raw API pull is written to ``data/raw/<namespace>/<key>.parquet`` the first time it
is fetched. Re-running any pipeline stage therefore never touches the network for data it
has already seen, which makes results reproducible and keeps us well inside public rate
limits. Only immutable data is cached: we only pull *resolved* markets, whose history can
no longer change.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


class RateLimiter:
    """Blocking limiter that enforces a minimum interval between consecutive calls."""

    def __init__(self, calls_per_second: float) -> None:
        """Create a limiter allowing at most ``calls_per_second`` calls per second."""
        if calls_per_second <= 0:
            raise ValueError("calls_per_second must be positive")
        self.min_interval = 1.0 / calls_per_second
        self._last_call = 0.0

    def wait(self) -> None:
        """Sleep just long enough to respect the configured rate."""
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()


class HttpClient:
    """Thin ``requests`` wrapper with rate limiting and exponential backoff.

    Retries on HTTP 429 and 5xx responses and on connection errors. Client errors other
    than 429 (e.g. 404 for a market that lives in a different API tier) are raised
    immediately so callers can fall back.
    """

    def __init__(
        self,
        calls_per_second: float = 5.0,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        timeout: float = 30.0,
    ) -> None:
        """Configure rate, retry budget, backoff base (seconds) and request timeout."""
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "robust-kelly-prediction-markets-research/0.1"
        self.limiter = RateLimiter(calls_per_second)
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.timeout = timeout

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        """GET ``url`` and return the decoded JSON body.

        Raises:
            requests.HTTPError: on a non-retryable status or after retries are exhausted.
        """
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt == self.max_retries:
                    raise
                delay = self.backoff_base * 2**attempt
                logger.warning("network error %s on %s, retrying in %.1fs", exc, url, delay)
                time.sleep(delay)
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == self.max_retries:
                    resp.raise_for_status()
                retry_after = resp.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else self.backoff_base * 2**attempt
                logger.warning("HTTP %s on %s, retrying in %.1fs", resp.status_code, url, delay)
                time.sleep(delay)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError("unreachable")  # pragma: no cover


def _safe_key(key: str) -> str:
    """Map an arbitrary cache key to a short, filesystem-safe file stem."""
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
    if len(stem) > 80:
        digest = hashlib.sha1(key.encode()).hexdigest()[:12]
        stem = f"{stem[:60]}_{digest}"
    return stem


class ParquetCache:
    """Namespace/key -> parquet file store for DataFrames."""

    def __init__(self, root: Path | str = DEFAULT_CACHE_DIR) -> None:
        """Use ``root`` as the cache directory (created on demand)."""
        self.root = Path(root)

    def path(self, namespace: str, key: str) -> Path:
        """Return the parquet path for ``(namespace, key)``."""
        return self.root / namespace / f"{_safe_key(key)}.parquet"

    def get(self, namespace: str, key: str) -> pd.DataFrame | None:
        """Return the cached frame, or ``None`` on a cache miss."""
        p = self.path(namespace, key)
        return pd.read_parquet(p) if p.exists() else None

    def put(self, namespace: str, key: str, df: pd.DataFrame) -> None:
        """Write ``df`` to the cache atomically (write to temp file, then rename)."""
        p = self.path(namespace, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(p)

    def get_or_fetch(
        self, namespace: str, key: str, fetch: Callable[[], pd.DataFrame]
    ) -> pd.DataFrame:
        """Return the cached frame or call ``fetch()``, cache its result and return it."""
        cached = self.get(namespace, key)
        if cached is not None:
            return cached
        df = fetch()
        self.put(namespace, key, df)
        return df
