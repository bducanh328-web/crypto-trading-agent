"""BTC + XAU theo xu hướng ngày: đúng bản "Đặc tả chốt trước kiểm định" ngày 23/11/2026. Không chỉnh thông số.

Mô hình vốn: giữ giá trị vị thế (notional) theo từng tài sản; vốn thay đổi theo lợi nhuận ngày của tài sản đó trong
ngày nó giao dịch. Funding tính mỗi ngày lịch (hợp đồng vĩnh cửu chạy cả cuối tuần). Quyết định ở giá đóng cửa ngày
giao dịch d của tài sản, thực hiện ở giá đóng cửa ngày giao dịch kế tiếp của tài sản đó (trễ 1 ngày).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LOOKBACKS = (20, 60, 120, 250)
VOL_DAYS = 60
ASSET_VOL = 0.046  # 6,5% / √2
MAX_W = 2.0
BAND = 0.20
FEE = 0.0006
SLIP = 0.0005
ANN = {"BTC": 365, "XAU": 252}


def signal(close: pd.Series, lookbacks=LOOKBACKS) -> pd.Series:
    parts = [np.sign(close / close.shift(n) - 1) for n in lookbacks]
    s = sum(parts) / len(parts)
    return s.where(close.shift(max(lookbacks)).notna())


def realized_vol(close: pd.Series, ann_days: int, n: int = VOL_DAYS) -> pd.Series:
    return np.log(close).diff().rolling(n).std() * np.sqrt(ann_days)


def target_weights(close: pd.Series, ann_days: int, mode: str, max_w: float = MAX_W) -> pd.Series:
    """mode: long_short (V1) | long_flat (V2) | always_long (M4)."""
    s = signal(close)
    if mode == "long_flat":
        s = s.clip(lower=0)
    elif mode == "always_long":
        s = s.where(s.isna(), 1.0)
    w = s * ASSET_VOL / realized_vol(close, ann_days)
    return w.clip(-max_w, max_w)


@dataclass
class SimResult:
    equity: pd.Series
    contrib: pd.DataFrame  # lãi lỗ cộng dồn theo tài sản (đơn vị vốn ban đầu)
    weights: pd.DataFrame
    trades: int
    costs: float
    funding: float


def simulate(prices: dict[str, pd.Series], targets: dict[str, pd.Series], funding_daily: dict[str, pd.Series],
             calendar: pd.DatetimeIndex, band: float = BAND, monthly: bool = False, fee: float = FEE,
             slip: float = SLIP) -> SimResult:  # fmt: skip
    """prices/targets theo ngày giao dịch của từng tài sản; funding_daily theo ngày lịch (tỷ lệ mỗi ngày)."""
    eq = 1.0
    notional = dict.fromkeys(prices, 0.0)
    last_px = dict.fromkeys(prices, np.nan)
    pending = dict.fromkeys(prices, np.nan)  # mục tiêu quyết định ở ngày giao dịch trước, chờ thực hiện
    contrib = dict.fromkeys(prices, 0.0)
    rows, wrows, cum = [], [], []
    trades, costs, fund_total = 0, 0.0, 0.0
    px_map = {a: s.to_dict() for a, s in prices.items()}
    tg_map = {a: s.to_dict() for a, s in targets.items()}
    fd_map = {a: s.to_dict() for a, s in funding_daily.items()}
    done_month = dict.fromkeys(prices, None)
    for t in calendar:
        # 1) lợi nhuận và funding trong ngày
        for a in prices:
            px = px_map[a].get(t)
            if px is not None and np.isfinite(px):
                if np.isfinite(last_px[a]) and notional[a] != 0.0:
                    pnl = notional[a] * (px / last_px[a] - 1)
                    eq += pnl
                    contrib[a] += pnl
                    notional[a] *= px / last_px[a]
                last_px[a] = px
            f = fd_map.get(a, {}).get(t, 0.0)
            if notional[a] != 0.0 and f:
                cost = notional[a] * f
                eq -= cost
                contrib[a] -= cost
                fund_total += cost
        # 2) thực hiện mục tiêu đã quyết ở ngày giao dịch trước (chỉ ngày tài sản giao dịch)
        month = (t.year, t.month)
        for a in prices:
            if px_map[a].get(t) is None:
                continue
            tgt = pending[a]
            if np.isfinite(tgt):
                cw = notional[a] / eq if eq > 0 else 0.0
                if monthly:
                    go = done_month[a] != month
                else:
                    go = (np.sign(tgt) != np.sign(cw)) or abs(tgt - cw) > band * abs(cw)
                if go and abs(tgt - cw) > 1e-12:
                    trade = abs(tgt * eq - notional[a])
                    c = trade * (fee + slip)
                    eq -= c
                    contrib[a] -= c
                    costs += c
                    trades += 1
                    notional[a] = tgt * eq
                if go:
                    done_month[a] = month
            nt = tg_map[a].get(t)
            pending[a] = nt if nt is not None else np.nan
        rows.append(eq)
        wrows.append({a: notional[a] / eq if eq > 0 else 0.0 for a in prices})
        cum.append(dict(contrib))
    return SimResult(
        pd.Series(rows, index=calendar, name="equity"), pd.DataFrame(cum, index=calendar),
        pd.DataFrame(wrows, index=calendar), trades, costs, fund_total,
    )  # fmt: skip


def stats(equity: pd.Series) -> dict:
    eq = equity.dropna()
    r = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = eq / eq.cummax() - 1
    mdd = float(dd.min())
    cagr = float(eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    sharpe = float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else np.nan
    yearly = eq.resample("YE").last()
    first = eq.resample("YE").first()
    lo, hi = eq.index[0] - pd.Timedelta(days=1), eq.index[-1]
    full = [y for y in yearly.index.year if pd.Timestamp(f"{y}-01-01") >= lo and pd.Timestamp(f"{y}-12-31") <= hi]
    yr = {}
    prev_end = None
    for y in yearly.index.year:
        start_val = prev_end if prev_end is not None else first[first.index.year == y].iloc[0]
        end_val = yearly[yearly.index.year == y].iloc[0]
        yr[int(y)] = float(end_val / start_val - 1)
        prev_end = end_val
    full_yr = {y: v for y, v in yr.items() if y in full}
    breach = dd[dd <= -0.15]
    return {
        "cagr": cagr, "sut_giam": mdd, "sharpe": sharpe,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.inf,
        "bien_dong": float(r.std() * np.sqrt(365)), "loi_nhuan": float(eq.iloc[-1] / eq.iloc[0] - 1),
        "theo_nam": yr, "nam_tron_lai": (sum(v > 0 for v in full_yr.values()) / len(full_yr)) if full_yr else np.nan,
        "so_nam_tron": len(full_yr), "cham_15": str(breach.index[0].date()) if len(breach) else "",
    }  # fmt: skip
