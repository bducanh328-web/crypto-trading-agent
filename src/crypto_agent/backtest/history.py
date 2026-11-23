"""Lịch sử cho kiểm định: mọi hợp đồng USDT-M trên data.binance.vision, kể cả mã đã hủy niêm yết.

Bước 1: liệt kê mã (S3 listing), tải nến 1D tháng của tất cả mã.
Bước 2: rổ theo thời điểm: đầu mỗi tháng lấy 30 mã có GTGD 30 ngày lớn nhất, niêm yết ≥ 180 ngày.
Bước 3: tải nến 4H và funding cho các mã từng vào rổ (và BTC).
Kết quả lưu CSV dưới data/hist/{1d,4h,funding}/SYM.csv và data/hist/ro_thang.csv.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from ..config import STABLECOINS
from ..data.binance_vision import _read_zip_csv, parse_funding, parse_klines

log = logging.getLogger(__name__)

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FILES = "https://data.binance.vision"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

# Hợp đồng không phải tiền mã hoá (kim loại, hàng hoá, cổ phiếu, chỉ số) và stablecoin: loại khỏi rổ.
NON_CRYPTO = frozenset(
    {"XAU", "XAG", "XPT", "XPD", "PAXG", "XAUT", "CL", "BZ", "NATGAS", "COPPER", "TSLA", "MSTR", "NVDA", "AAPL",
     "AMZN", "GOOGL", "META", "MSFT", "COIN", "HOOD", "QQQ", "SPY", "CRCL", "INTC", "AMD", "MU", "BTCDOM", "DEFI",
     "FOOTBALL", "BLUEBIRD"}
)  # fmt: skip
EXCLUDED = NON_CRYPTO | STABLECOINS


def _get_text(url: str) -> str:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.text


def _get_bytes(url: str) -> bytes | None:
    for _ in range(3):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException as exc:
            last = exc
    raise last  # type: ignore[possibly-undefined]


def list_keys(prefix: str, delimiter: bool, get: Callable[[str], str] = _get_text) -> list[str]:
    """Liệt kê khoá S3 (hoặc thư mục con nếu delimiter) có phân trang."""
    out, marker = [], ""
    while True:
        url = f"{S3}?prefix={prefix}" + ("&delimiter=/" if delimiter else "") + (f"&marker={marker}" if marker else "")
        root = ET.fromstring(get(url))
        if delimiter:
            items = [e.text for e in root.findall("s3:CommonPrefixes/s3:Prefix", NS)]
        else:
            items = [e.text for e in root.findall("s3:Contents/s3:Key", NS)]
        out += [i for i in items if i]
        truncated = (root.findtext("s3:IsTruncated", default="false", namespaces=NS) or "").lower() == "true"
        if not truncated or not items:
            return out
        marker = root.findtext("s3:NextMarker", default="", namespaces=NS) or items[-1]


def all_usdt_symbols(get: Callable[[str], str] = _get_text) -> list[str]:
    dirs = list_keys("data/futures/um/monthly/klines/", True, get)
    syms = [d.rstrip("/").rsplit("/", 1)[-1] for d in dirs]
    keep = []
    for s in syms:
        m = re.fullmatch(r"([A-Z0-9]+?)USDT", s)
        if m and m.group(1) not in EXCLUDED:
            keep.append(s)
    return sorted(keep)


def _month_ok(key: str, start: date, end: date) -> bool:
    m = re.search(r"(\d{4})-(\d{2})\.zip$", key)
    if not m:
        return False
    ym = (int(m.group(1)), int(m.group(2)))
    return (start.year, start.month) <= ym <= (end.year, end.month)


class History:
    def __init__(self, root: Path | str, workers: int = 16, get_text=_get_text, get_bytes=_get_bytes):
        self.root = Path(root)
        self.workers = workers
        self._text, self._bytes = get_text, get_bytes

    def path(self, kind: str, sym: str) -> Path:
        return self.root / kind / f"{sym}.csv"

    def load(self, kind: str, sym: str) -> pd.DataFrame | pd.Series | None:
        p = self.path(kind, sym)
        if not p.exists():
            return None
        df = pd.read_csv(p, index_col=0)
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True))
        return df.iloc[:, 0] if kind == "funding" else df

    def _download(self, kind: str, sym: str, start: date, end: date) -> int:
        p = self.path(kind, sym)
        if p.exists():
            return -1
        folder = {"1d": f"klines/{sym}/1d/", "4h": f"klines/{sym}/4h/", "funding": f"fundingRate/{sym}/"}[kind]
        keys = [k for k in list_keys(f"data/futures/um/monthly/{folder}", False, self._text) if k.endswith(".zip")]
        keys = [k for k in keys if _month_ok(k, start, end)]
        frames = []
        for k in keys:
            blob = self._bytes(f"{FILES}/{k}")
            if blob is None:
                continue
            raw = _read_zip_csv(blob)
            frames.append(parse_funding(raw) if kind == "funding" else parse_klines(raw))
        p.parent.mkdir(parents=True, exist_ok=True)
        if not frames:
            (pd.Series(dtype=float, name="v") if kind == "funding" else pd.DataFrame()).to_csv(p)
            return 0
        out = pd.concat(frames).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        (out.rename("funding_rate") if kind == "funding" else out).to_csv(p)
        return len(out)

    def download(self, kind: str, symbols: list[str], start: date, end: date) -> None:
        def job(sym):
            try:
                n = self._download(kind, sym, start, end)
                if n >= 0:
                    log.info("%s %s: %d dòng", kind, sym, n)
            except Exception as exc:  # một mã lỗi không dừng cả loạt
                log.warning("%s %s lỗi: %s", kind, sym, exc)

        with ThreadPoolExecutor(self.workers) as ex:
            list(ex.map(job, symbols))


def monthly_universe(daily: dict[str, pd.DataFrame], start: date, end: date, size: int = 30,
                     min_days: int = 180, window: int = 30) -> pd.DataFrame:  # fmt: skip
    """Rổ theo thời điểm: tại 00:00 UTC ngày đầu tháng, xếp theo GTGD `window` ngày trước đó."""
    rows = []
    for month in pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="MS"):
        scores = {}
        for sym, df in daily.items():
            if df is None or df.empty or df.index[0] > month - pd.Timedelta(days=min_days):
                continue
            win = df.loc[month - pd.Timedelta(days=window) : month - pd.Timedelta(seconds=1)]
            if len(win) >= window * 0.8:
                scores[sym] = float(win["turnover"].sum())
        top = sorted(scores, key=scores.get, reverse=True)[:size]
        rows += [{"thang": month.strftime("%Y-%m"), "hang": i + 1, "ma": s} for i, s in enumerate(top)]
    return pd.DataFrame(rows, columns=["thang", "hang", "ma"])
