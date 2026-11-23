"""Rổ coin: N hợp đồng vĩnh cửu USDT có giá trị giao dịch 30 ngày lớn nhất trên Bitunix.

Bỏ stablecoin, coin niêm yết dưới 180 ngày, hợp đồng không ở trạng thái OPEN hoặc không cho giao dịch qua API.
Đây là rổ hiện tại (dùng chạy hằng ngày); rổ theo thời điểm cho kiểm định nằm ở giai đoạn 2.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .config import PARAMS, Params


def eligible_pairs(pairs: pd.DataFrame, now: datetime, p: Params = PARAMS) -> pd.DataFrame:
    if pairs.empty:
        return pairs
    df = pairs.copy()
    ok = df.get("quote", "USDT").astype(str).str.upper().eq("USDT")
    ok &= ~df["base"].astype(str).str.upper().isin(p.excluded_bases)
    if "symbolStatus" in df:
        ok &= df["symbolStatus"].astype(str).str.upper().eq("OPEN")
    if "isApiSupported" in df:
        ok &= df["isApiSupported"].astype(str).str.lower().isin(["true", "1"])
    now_ms = now.timestamp() * 1000
    if "launchTime" in df:
        launch = pd.to_numeric(df["launchTime"], errors="coerce")
        ok &= launch.isna() | (launch <= now_ms - p.min_listed_days * 86_400_000)
    if "delistTime" in df:
        delist = pd.to_numeric(df["delistTime"], errors="coerce").fillna(0)
        ok &= (delist <= 0) | (delist > now_ms)
    return df[ok]


def rank_by_turnover(daily: dict[str, pd.DataFrame], asof: pd.Timestamp, p: Params = PARAMS) -> list[str]:
    """Xếp hạng theo tổng giá trị giao dịch (USDT) của `turnover_days` nến ngày kết thúc tại asof."""
    scores = {}
    for sym, df in daily.items():
        win = df.loc[:asof].tail(p.turnover_days)
        if len(win) >= p.turnover_days * 0.8:
            scores[sym] = float(win["turnover"].sum())
    return [s for s, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)][: p.universe_size]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
