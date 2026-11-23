"""Dữ liệu cho kiểm định BTC + XAU (đúng nguồn ghi trong đặc tả chốt trước).

BTC: Binance spot BTCUSDT 1d (data.binance.vision). Vàng: XAUUSD giao ngay (Stooq), dự phòng GC=F (Yahoo).
Funding: Binance BTCUSDT (USDⓈ-M) theo tháng; XAUUSDT trên Bitunix (API công khai).
Tất cả chuỗi giá đánh chỉ số theo ngày (không múi giờ), giá trị = giá đóng cửa của ngày đó.
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..backtest.history import FILES, _get_bytes, list_keys
from ..data.binance_vision import _read_zip_csv, parse_funding, parse_klines

log = logging.getLogger(__name__)
UA = {"User-Agent": "Mozilla/5.0 (compatible; crypto-trading-agent/0.1)"}


def _dates(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(idx, utc=True).tz_convert(None).normalize())


def btc_daily() -> pd.Series:
    keys = [k for k in list_keys("data/spot/monthly/klines/BTCUSDT/1d/", False) if k.endswith(".zip")]
    frames = [parse_klines(_read_zip_csv(b)) for k in keys if (b := _get_bytes(f"{FILES}/{k}")) is not None]
    df = pd.concat(frames).sort_index()
    s = df["close"].astype(float)
    s.index = _dates(s.index)
    return s[~s.index.duplicated(keep="last")].rename("BTC")


def parse_stooq(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols or "close" not in cols:
        raise ValueError(f"Stooq trả dữ liệu lạ: {text[:120]!r}")
    s = pd.Series(pd.to_numeric(df[cols["close"]], errors="coerce").to_numpy(), index=pd.to_datetime(df[cols["date"]]))
    return s.dropna().sort_index().rename("XAU")


def parse_yahoo(body: dict) -> pd.Series:
    res = body["chart"]["result"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert("America/New_York").tz_localize(None)
    close = res["indicators"]["quote"][0]["close"]
    s = pd.Series(close, index=idx.normalize(), dtype=float).dropna()
    return s[~s.index.duplicated(keep="last")].sort_index().rename("XAU")


def gold_daily() -> tuple[pd.Series, str]:
    try:
        r = requests.get("https://stooq.com/q/d/l/?s=xauusd&i=d", headers=UA, timeout=60)
        r.raise_for_status()
        s = parse_stooq(r.text)
        if len(s) > 4000:
            return s, "Stooq XAUUSD giao ngay"
        log.warning("Stooq chỉ có %d dòng, chuyển sang Yahoo", len(s))
    except Exception as exc:
        log.warning("Stooq lỗi (%s), chuyển sang Yahoo GC=F", exc)
    url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F"
    r = requests.get(url, params={"period1": 946684800, "period2": int(time.time()), "interval": "1d"}, headers=UA,
                     timeout=60)  # fmt: skip
    r.raise_for_status()
    return parse_yahoo(r.json()), "Yahoo GC=F (hợp đồng vàng COMEX liên tục)"


def btc_funding_events() -> pd.Series:
    keys = [k for k in list_keys("data/futures/um/monthly/fundingRate/BTCUSDT/", False) if k.endswith(".zip")]
    parts = [parse_funding(_read_zip_csv(b)) for k in keys if (b := _get_bytes(f"{FILES}/{k}")) is not None]
    return pd.concat(parts).sort_index()


def bitunix_funding_events(symbol: str = "XAUUSDT", api=None) -> pd.Series:
    """Toàn bộ lịch sử funding có được. API trả 200 kỳ GẦN NHẤT trong khoảng, nên lật trang lùi theo endTime."""
    if api is None:
        from ..data.bitunix import BitunixPublic

        api = BitunixPublic()
    start = int(pd.Timestamp("2020-01-01", tz="UTC").timestamp() * 1000)
    end = int(time.time() * 1000)
    out: list[pd.Series] = []
    for _ in range(500):
        s = api.funding_history(symbol, start, end)
        if s.empty:
            break
        out.append(s)
        oldest = int(s.index.min().timestamp() * 1000)
        if oldest >= end or oldest <= start:
            break
        end = oldest - 1
    if not out:
        return pd.Series(dtype=float)
    s = pd.concat(out).sort_index()
    return s[~s.index.duplicated()]


def daily_funding(events: pd.Series, calendar: pd.DatetimeIndex, fill_mean: bool) -> tuple[pd.Series, float]:
    """Tổng tỷ lệ funding theo ngày lịch. Ngày chưa có dữ liệu: mức trung bình ngày (nếu fill_mean)."""
    if events.empty:
        return pd.Series(0.0, index=calendar), 0.0
    ev = events.copy()
    ev.index = _dates(ev.index)
    per_day = ev.groupby(level=0).sum()
    first, last = per_day.index.min(), per_day.index.max()
    covered = pd.date_range(first, last, freq="D")
    mean_day = float(per_day.reindex(covered, fill_value=0.0).mean())
    out = per_day.reindex(calendar)
    if fill_mean:
        out = out.fillna(mean_day)
    return out.fillna(0.0), mean_day


def load_all(cache: Path) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, fn in (
        ("btc", btc_daily),
        ("btc_funding", btc_funding_events),
        ("xau_funding_v2", bitunix_funding_events),
    ):
        p = cache / f"{name}.csv"
        if p.exists():
            s = pd.read_csv(p, index_col=0).iloc[:, 0]
            s.index = pd.to_datetime(s.index, utc=name != "btc")
            if name == "btc":
                s.index = s.index.tz_localize(None) if s.index.tz is not None else s.index
        else:
            s = fn()
            s.to_csv(p)
        out[name] = s
    p = cache / "xau.csv"
    if p.exists():
        s = pd.read_csv(p, index_col=0).iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        src = (cache / "xau_nguon.txt").read_text(encoding="utf-8")
    else:
        s, src = gold_daily()
        s.to_csv(p)
        (cache / "xau_nguon.txt").write_text(src, encoding="utf-8")
    out["xau"], out["xau_nguon"] = s, src
    return out


__all__ = ["load_all", "daily_funding", "np"]
