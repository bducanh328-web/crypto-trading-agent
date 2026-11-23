"""Lớp bối cảnh: đèn chung XANH (chỉ long) / ĐỎ (chỉ short) / VÀNG (không mở lệnh mới). CHƯA KIỂM ĐỊNH.

XANH: BTC đóng cửa trên MA200 (1D), cấu trúc swing BTC đang tăng (BOS tăng gần nhất mới hơn BOS giảm), độ rộng
(% rổ trên MA50) ≥ 55%. ĐỎ: ngược lại, độ rộng ≤ 40%. Còn lại VÀNG. Đèn chỉ đổi khi điều kiện mới giữ đủ
`confirm_days` ngày liên tiếp, để tránh nhảy đèn quanh MA200.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PARAMS, Params
from .structure import DOWN, UP, structure

GREEN, RED, YELLOW = "XANH", "ĐỎ", "VÀNG"
MIN_BREADTH_COINS = 10


def breadth(daily: dict[str, pd.DataFrame], ma: int) -> pd.Series:
    """% coin trong rổ có giá đóng cửa trên MA của chính nó, theo ngày (chỉ tính coin đã đủ dữ liệu MA)."""
    above = {}
    for sym, df in daily.items():
        c = df["close"].astype(float)
        m = c.rolling(ma).mean()
        above[sym] = (c > m).where(m.notna())
    frame = pd.DataFrame(above)
    counted = frame.notna().sum(axis=1)
    share = frame.sum(axis=1) / counted.replace(0, np.nan)
    return share.where(counted >= MIN_BREADTH_COINS)


def raw_light(btc: pd.DataFrame, br: pd.Series, p: Params = PARAMS) -> pd.DataFrame:
    c = btc["close"].astype(float)
    ma = c.rolling(p.btc_ma).mean()
    trend = structure(btc, p.swing_n)["trend"]
    br = br.reindex(btc.index)
    above = c > ma
    green = above & trend.eq(UP) & (br >= p.green_breadth)
    red = (~above) & ma.notna() & trend.eq(DOWN) & (br <= p.red_breadth)
    raw = pd.Series(np.where(green, GREEN, np.where(red, RED, YELLOW)), index=btc.index)
    valid = ma.notna() & br.notna()
    return pd.DataFrame(
        {"close": c, "ma200": ma, "btc_vs_ma": c / ma - 1, "trend": trend, "breadth": br, "raw": raw.where(valid)}
    ).dropna(subset=["raw"])


def confirm(raw: pd.Series, days: int) -> pd.DataFrame:
    """Đèn chính thức: chỉ đổi khi giá trị thô mới lặp lại `days` ngày liên tiếp. Kèm số ngày giữ đèn."""
    light, held = [], []
    cur, count, run_val, run_len = None, 0, None, 0
    for v in raw:
        run_len = run_len + 1 if v == run_val else 1
        run_val = v
        if cur is None:
            cur, count = v, 1
        elif v != cur and run_len >= days:
            cur, count = v, run_len
        else:
            count += 1
        light.append(cur)
        held.append(count)
    return pd.DataFrame({"light": light, "days": held}, index=raw.index)


def funding_per_8h(rates: pd.Series, window_hours: int, now: pd.Timestamp) -> float:
    """Funding trung bình trong cửa sổ, quy về mỗi 8 giờ (hợp đồng có kỳ 1h/4h/8h khác nhau)."""
    r = rates[rates.index > now - pd.Timedelta(hours=window_hours)]
    if r.empty:
        return float("nan")
    gaps = rates.index.to_series().diff().dt.total_seconds().dropna() / 3600
    interval = float(gaps.median()) if len(gaps) else 8.0
    interval = interval if interval > 0 else 8.0
    return float(r.mean()) * 8.0 / interval


@dataclass
class Context:
    asof: pd.Timestamp  # giờ mở nến 1D cuối cùng đã đóng (UTC)
    light: str
    days: int
    raw: str
    btc_close: float
    btc_ma200: float
    btc_vs_ma: float
    btc_trend: str | None
    breadth50: float
    breadth200: float
    funding: dict[str, float]
    history: pd.DataFrame

    def allowed(self, side: str, symbol: str, p: Params = PARAMS) -> bool:
        """Chiều lệnh có được phép theo đèn chung và bộ lọc funding của coin đó không."""
        f = self.funding.get(symbol, float("nan"))
        if side == "long":
            return self.light == GREEN and not (np.isfinite(f) and f > p.funding_hot)
        if side == "short":
            return self.light == RED and not (np.isfinite(f) and f < p.funding_cold)
        raise ValueError(side)


def build_context(
    btc: pd.DataFrame, basket: dict[str, pd.DataFrame], funding: dict[str, pd.Series], p: Params = PARAMS
) -> Context:
    br50 = breadth(basket, p.breadth_ma)
    br200 = breadth(basket, 200)
    hist = raw_light(btc, br50, p)
    if hist.empty:
        raise ValueError("Chưa đủ dữ liệu để tính đèn (cần ≥ 200 nến ngày BTC và ≥ 50 nến ngày cho rổ)")
    conf = confirm(hist["raw"], p.confirm_days)
    hist = hist.join(conf)
    last = hist.iloc[-1]
    asof = hist.index[-1]
    now = asof + pd.Timedelta(days=1)
    fund = {s: funding_per_8h(r, p.funding_window_hours, now) for s, r in funding.items() if len(r)}
    b200 = br200.reindex(hist.index).iloc[-1]
    return Context(
        asof=asof, light=last["light"], days=int(last["days"]), raw=last["raw"], btc_close=float(last["close"]),
        btc_ma200=float(last["ma200"]), btc_vs_ma=float(last["btc_vs_ma"]), btc_trend=last["trend"],
        breadth50=float(last["breadth"]), breadth200=float(b200) if pd.notna(b200) else float("nan"),
        funding=fund, history=hist,
    )  # fmt: skip
