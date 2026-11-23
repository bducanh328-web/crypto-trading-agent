"""Lịch sử dài cho kiểm định: tệp nén công khai trên data.binance.vision (USDⓈ-M Futures).

Dùng tệp tĩnh thay cho API fapi.binance.com vì API này chặn nhiều vùng (kể cả máy GitHub Actions đặt ở Mỹ).
Tháng đã qua lấy tệp tháng; tháng hiện tại lấy tệp ngày (có sau khoảng 1 ngày).
Ví dụ: data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip
"""

from __future__ import annotations

import io
import logging
import zipfile
from collections.abc import Callable
from datetime import date, timedelta

import pandas as pd
import requests

from .bitunix import OHLCV

log = logging.getLogger(__name__)

BASE = "https://data.binance.vision/data/futures/um"
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
    "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]  # fmt: skip


def _http_bytes(url: str) -> bytes | None:
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.content


def _read_zip_csv(blob: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = zf.namelist()[0]
        raw = zf.read(name).decode()
    first = raw.split("\n", 1)[0]
    has_header = not first.split(",")[0].strip().lstrip("-").isdigit()
    return pd.read_csv(io.StringIO(raw), header=0 if has_header else None)


def parse_klines(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=OHLCV)
    if all(isinstance(c, int) for c in df.columns):
        df.columns = KLINE_COLS[: len(df.columns)]
    t = pd.to_numeric(df["open_time"])
    unit = "us" if t.max() > 1e14 else "ms"  # từ 2025 Binance chuyển một số tệp sang micro giây
    out = pd.DataFrame(
        {
            "open": pd.to_numeric(df["open"]),
            "high": pd.to_numeric(df["high"]),
            "low": pd.to_numeric(df["low"]),
            "close": pd.to_numeric(df["close"]),
            "volume": pd.to_numeric(df["volume"]),
            "turnover": pd.to_numeric(df["quote_volume"]),
        }
    )
    out.index = pd.DatetimeIndex(pd.to_datetime(t, unit=unit, utc=True), name="open_time")
    return out[~out.index.duplicated(keep="last")].sort_index()


def parse_funding(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    if all(isinstance(c, int) for c in df.columns):
        df.columns = ["calc_time", "funding_interval_hours", "last_funding_rate"][: len(df.columns)]
    t = pd.to_numeric(df["calc_time"])
    idx = pd.to_datetime(t, unit="us" if t.max() > 1e14 else "ms", utc=True)
    return pd.Series(pd.to_numeric(df["last_funding_rate"]).to_numpy(), index=idx).sort_index()


def _months(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


class BinanceVision:
    def __init__(self, fetch: Callable[[str], bytes | None] = _http_bytes):
        self._fetch = fetch

    def _get(self, url: str) -> pd.DataFrame | None:
        blob = self._fetch(url)
        return None if blob is None else _read_zip_csv(blob)

    def klines(self, symbol: str, interval: str, start: date, end: date) -> pd.DataFrame:
        frames = []
        this_month = (end.year, end.month)
        for y, m in _months(start, end):
            if (y, m) < this_month:
                df = self._get(f"{BASE}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{y}-{m:02d}.zip")
                if df is not None:
                    frames.append(parse_klines(df))
                continue
            d = date(y, m, 1)
            while d <= end:
                df = self._get(f"{BASE}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{d:%Y-%m-%d}.zip")
                if df is not None:
                    frames.append(parse_klines(df))
                d += timedelta(days=1)
        if not frames:
            return pd.DataFrame(columns=OHLCV)
        out = pd.concat(frames)
        return out[~out.index.duplicated(keep="last")].sort_index()

    def funding(self, symbol: str, start: date, end: date) -> pd.Series:
        frames = []
        for y, m in _months(start, end):
            df = self._get(f"{BASE}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{y}-{m:02d}.zip")
            if df is not None:
                frames.append(parse_funding(df))
        return pd.concat(frames).sort_index() if frames else pd.Series(dtype=float)
