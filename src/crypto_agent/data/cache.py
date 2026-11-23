"""Bộ đệm nến dạng CSV: data/klines/{interval}/{SYMBOL}.csv (giờ mở nến UTC). CSV để không cần thêm thư viện."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .bitunix import OHLCV


class KlineCache:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def path(self, symbol: str, interval: str) -> Path:
        return self.root / "klines" / interval / f"{symbol}.csv"

    def load(self, symbol: str, interval: str) -> pd.DataFrame:
        p = self.path(symbol, interval)
        if not p.exists():
            return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz="UTC", name="open_time"))
        df = pd.read_csv(p, index_col="open_time")
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True), name="open_time")
        return df[OHLCV]

    def merge(self, symbol: str, interval: str, new: pd.DataFrame) -> pd.DataFrame:
        old = self.load(symbol, interval)
        both = pd.concat([old, new[OHLCV]]) if len(old) else new[OHLCV]
        both = both[~both.index.duplicated(keep="last")].sort_index()
        p = self.path(symbol, interval)
        p.parent.mkdir(parents=True, exist_ok=True)
        both.to_csv(p, index_label="open_time")
        return both
