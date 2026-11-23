"""Dữ liệu công khai của Bitunix Futures (không cần khóa API): nến, danh sách hợp đồng, ticker, funding.

Tài liệu: https://www.bitunix.com/api-docs/futures/market/ . Giới hạn 10 lượt/giây/IP; nến tối đa 200 cây mỗi lượt,
chỉ có endTime (lùi về quá khứ), chỉ trả nến đã đóng.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://fapi.bitunix.com"
KLINE_LIMIT = 200
FUNDING_LIMIT = 200
INTERVAL_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
OHLCV = ["open", "high", "low", "close", "volume", "turnover"]


class BitunixError(RuntimeError):
    pass


def _http_get(url: str, params: dict) -> dict:
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


class BitunixPublic:
    def __init__(self, get: Callable[[str, dict], dict] = _http_get, base_url: str = BASE_URL, pause: float = 0.12):
        self._get, self._base, self._pause = get, base_url, pause

    def _call(self, path: str, params: dict) -> list | dict:
        last: Exception | None = None
        for attempt in range(3):
            try:
                body = self._get(self._base + path, params)
                if body.get("code") != 0:
                    raise BitunixError(f"{path}: code={body.get('code')} msg={body.get('msg')}")
                time.sleep(self._pause)
                return body.get("data") or []
            except (requests.RequestException, BitunixError) as exc:
                last = exc
                time.sleep(1.5 * (attempt + 1) if self._pause else 0)
        raise BitunixError(f"Bitunix {path} lỗi sau 3 lần thử: {last}")

    # --- Nến ---------------------------------------------------------------------------------------------------
    def klines(self, symbol: str, interval: str, bars: int, end_ms: int | None = None) -> pd.DataFrame:
        """`bars` nến đã đóng gần nhất (lật trang lùi theo endTime). Chỉ số = giờ mở nến (UTC)."""
        rows: list[dict] = []
        end = end_ms
        while len(rows) < bars:
            params = {"symbol": symbol, "interval": interval, "limit": min(KLINE_LIMIT, bars - len(rows))}
            if end is not None:
                params["endTime"] = int(end)
            page = self._call("/api/v1/futures/market/kline", params)
            if not page:
                break
            rows.extend(page)
            oldest = min(int(r["time"]) for r in page)
            if len(page) < params["limit"] or (end is not None and oldest >= end):
                break
            end = oldest  # endTime loại trừ: trang sau kết thúc trước nến cũ nhất của trang này
        return normalize_klines(rows)

    # --- Hợp đồng, ticker, funding ----------------------------------------------------------------------------
    def trading_pairs(self) -> pd.DataFrame:
        df = pd.DataFrame(self._call("/api/v1/futures/market/trading_pairs", {}))
        if df.empty:
            return df
        for col in ("launchTime", "delistTime"):
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def tickers(self) -> pd.DataFrame:
        df = pd.DataFrame(self._call("/api/v1/futures/market/tickers", {}))
        for col in ("lastPrice", "markPrice", "quoteVol", "baseVol", "open", "high", "low"):
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def funding_history(self, symbol: str, start_ms: int, end_ms: int | None = None) -> pd.Series:
        """Funding đã thanh toán từ start_ms; chỉ số = giờ thanh toán (UTC), giá trị = tỷ lệ mỗi kỳ."""
        params = {"symbol": symbol, "startTime": int(start_ms), "limit": FUNDING_LIMIT}
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        data = self._call("/api/v1/futures/market/get_funding_rate_history", params)
        if not data:
            return pd.Series(dtype=float, name=symbol)
        df = pd.DataFrame(data)
        idx = pd.to_datetime(pd.to_numeric(df["fundingTime"]), unit="ms", utc=True)
        return pd.Series(pd.to_numeric(df["fundingRate"]).to_numpy(), index=idx, name=symbol).sort_index()


def normalize_klines(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz="UTC", name="open_time"))
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {c: pd.to_numeric(df[c], errors="coerce") for c in ("open", "high", "low", "close")},
    )
    q = pd.to_numeric(df.get("quoteVol"), errors="coerce")
    b = pd.to_numeric(df.get("baseVol"), errors="coerce")
    coin, usdt = split_volume(q, b, out["close"])
    out["volume"], out["turnover"] = coin, usdt
    out.index = pd.DatetimeIndex(pd.to_datetime(pd.to_numeric(df["time"]), unit="ms", utc=True), name="open_time")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out[OHLCV]


def split_volume(a: pd.Series, b: pd.Series, price: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Tài liệu Bitunix không nói rõ quoteVol/baseVol là đơn vị coin hay USDT (mẫu còn đặt ngược): suy từ dữ liệu.

    Khối lượng coin × giá ≈ giá trị USDT. Chọn cách gán nào khớp hơn trên trung vị các nến.
    """
    a, b, p = (pd.Series(x, dtype=float).reset_index(drop=True) for x in (a, b, price))
    ok = (a > 0) & (b > 0) & (p > 0)
    if not ok.any():
        return b, a  # mặc định: baseVol = coin, quoteVol = USDT
    err_ab = np.nanmedian(np.abs(np.log(a[ok] / (b[ok] * p[ok]))))  # a là USDT, b là coin
    err_ba = np.nanmedian(np.abs(np.log(b[ok] / (a[ok] * p[ok]))))  # b là USDT, a là coin
    return (b, a) if err_ab <= err_ba else (a, b)
