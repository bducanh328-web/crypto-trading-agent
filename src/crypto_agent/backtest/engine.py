"""Bộ mô phỏng theo đặc tả "Lớp bối cảnh + BOS hai chiều". Mọi ngưỡng CHƯA KIỂM ĐỊNH; đây là công cụ kiểm định.

Quy ước thời gian: nến 4H mở lúc T, đóng lúc T + 4h. Tín hiệu xét khi nến đóng; lệnh limit có hiệu lực ở nến kế.
Đèn chung và xu hướng 1D chỉ dùng nến ngày đã đóng tại thời điểm xét (không nhìn trước).
Thứ tự trong một nến (thận trọng): khớp lệnh → chạm stop → chốt 2R → quản lý cuối nến. Nến chạm cả stop và mục tiêu
tính là chạm stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from ..context import GREEN, RED, YELLOW, confirm
from ..structure import DOWN, UP, structure

H4 = pd.Timedelta(hours=4)
D1 = pd.Timedelta(days=1)
MAJORS = frozenset({"BTCUSDT", "ETHUSDT"})


@dataclass(frozen=True)
class BTParams:
    # Bối cảnh
    context: str = "full"  # full | btc_only | none
    btc_ma: int = 200
    breadth_ma: int = 50
    green_breadth: float = 0.55
    red_breadth: float = 0.40
    confirm_days: int = 2
    yellow_trade: bool = False  # biến thể: đèn VÀNG được giao dịch nửa khối lượng
    # Tín hiệu
    swing_n: int = 3
    trend_ema: int = 50
    rs_bars: int = 20
    body_min: float = 0.60
    vol_mult: float = 1.5
    max_chase_atr: float = 1.5
    funding_filter: bool = True
    funding_hot: float = 0.0005
    funding_cold: float = -0.0003
    entry: str = "breakout"  # breakout | pullback (0,7 sóng đẩy, chờ 12 nến)
    pullback_level: float = 0.7
    pullback_bars: int = 12
    skip_hours_utc: tuple[int, ...] = ()  # biến thể: bỏ tín hiệu đóng nến lúc 20:00 UTC (03:00 VN)
    majors_only: bool = False
    # Rủi ro
    risk: float = 0.005
    stop_buffer_atr: float = 0.2
    max_stop_atr: float = 3.0
    max_positions: int = 4
    max_open_risk: float = 0.03
    max_alts_same_side: int = 3
    notional_cap: float = 1.25  # giá trị vị thế ≤ 1,25 lần vốn (4 lệnh × 1,25 / đòn bẩy 5x ≤ vốn)
    cooldown_bars: int = 6
    month_stop: float = 0.06
    total_stop: float = 0.15
    # Thoát
    partial_r: float = 2.0
    time_stop_bars: int = 30
    # Chi phí
    taker: float = 0.0006
    maker: float = 0.0002
    slip_major: float = 0.0005
    slip_alt: float = 0.0015


# ------------------------------------------------------------------------------------------------------------------
# Chuẩn bị dữ liệu
# ------------------------------------------------------------------------------------------------------------------


def daily_light(btc_d: pd.DataFrame, daily: dict[str, pd.DataFrame], universe: pd.DataFrame, p: BTParams) -> pd.Series:
    """Đèn theo ngày, chỉ số = giờ ĐÓNG nến ngày (giá trị dùng được từ thời điểm đó)."""
    c = btc_d["close"].astype(float)
    ma = c.rolling(p.btc_ma).mean()
    trend = structure(btc_d, p.swing_n)["trend"]
    above = {}
    for sym, df in daily.items():
        dc = df["close"].astype(float)
        m = dc.rolling(p.breadth_ma).mean()
        above[sym] = (dc > m).where(m.notna())
    frame = pd.DataFrame(above).reindex(btc_d.index)
    months = btc_d.index.strftime("%Y-%m")
    member = pd.DataFrame(False, index=btc_d.index, columns=frame.columns)
    for month, grp in universe.groupby("thang"):
        cols = [s for s in grp["ma"] if s in member.columns]
        member.loc[months == month, cols] = True
    f = frame.where(member)
    counted = f.notna().sum(axis=1)
    br = (f.sum(axis=1) / counted.replace(0, np.nan)).where(counted >= 10)
    is_up, is_dn = trend.eq(UP), trend.eq(DOWN)
    if p.context == "btc_only":
        green, red = (c > ma) & is_up, (c < ma) & is_dn
        valid = ma.notna()
    else:
        green = (c > ma) & is_up & (br >= p.green_breadth)
        red = (c < ma) & is_dn & (br <= p.red_breadth)
        valid = ma.notna() & br.notna()
    raw = pd.Series(np.where(green, GREEN, np.where(red, RED, YELLOW)), index=btc_d.index)[valid]
    light = confirm(raw, p.confirm_days)["light"]
    light.index = light.index + D1
    return light


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def funding_8h(f: pd.Series | None, when: pd.DatetimeIndex) -> pd.Series:
    """Funding trung bình 72 giờ quy về mỗi 8 giờ, tại các thời điểm `when`."""
    if f is None or f.empty:
        return pd.Series(np.nan, index=when)
    gaps = f.index.to_series().diff().dt.total_seconds() / 3600
    interval = gaps.rolling(10, min_periods=1).median().fillna(8.0).clip(lower=1)
    norm = f * 8.0 / interval
    avg = norm.rolling("72h").mean()
    return avg.reindex(avg.index.union(when)).ffill().reindex(when)


def prepare(sym: str, h4: pd.DataFrame, d1: pd.DataFrame, btc_close_4h: pd.Series, fund: pd.Series | None,
            p: BTParams) -> pd.DataFrame:  # fmt: skip
    """Mọi cột cần cho tín hiệu, theo nến 4H (chỉ số = giờ mở nến)."""
    df = h4[["open", "high", "low", "close", "turnover"]].astype(float).copy()
    st = structure(df, p.swing_n)
    df = df.join(st)
    df["atr"] = _atr(df)
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["body"] = (df["close"] - df["open"]).abs() / rng
    df["pos"] = (df["close"] - df["low"]) / rng
    df["vol_ok"] = df["turnover"] >= p.vol_mult * df["turnover"].rolling(20).mean().shift(1)
    if sym == "BTCUSDT":
        df["rs"] = 0.0
    else:
        ratio = df["close"] / btc_close_4h.reindex(df.index)
        df["rs"] = ratio / ratio.shift(p.rs_bars) - 1
    close_t = df.index + H4
    dc = d1["close"].astype(float)
    ema = dc.ewm(span=p.trend_ema, adjust=False, min_periods=p.trend_ema).mean()
    d_up = pd.Series(np.where(ema.isna(), np.nan, (dc > ema).astype(float)), index=d1.index + D1)
    df["d_up"] = d_up.reindex(d_up.index.union(close_t)).ffill().reindex(close_t).to_numpy()
    df["fund"] = funding_8h(fund, close_t).to_numpy()
    return df


# ------------------------------------------------------------------------------------------------------------------
# Mô phỏng
# ------------------------------------------------------------------------------------------------------------------


@dataclass
class Pos:
    sym: str
    side: int  # +1 long, -1 short
    entry_t: pd.Timestamp
    entry: float
    stop: float
    stop0: float
    qty: float
    risk_amt: float
    light: str
    bars: int = 0
    partial: bool = False
    pnl: float = 0.0
    mfe_r: float = 0.0


@dataclass
class Order:
    sym: str
    side: int
    limit: float
    stop: float
    risk_amt: float
    qty: float
    atr: float
    light: str
    expires: int


@dataclass
class Result:
    trades: pd.DataFrame
    equity: pd.Series
    events: list[str] = field(default_factory=list)


def _slip(sym: str, p: BTParams) -> float:
    return p.slip_major if sym in MAJORS else p.slip_alt


def run(data: dict[str, pd.DataFrame], light: pd.Series, universe: pd.DataFrame, p: BTParams,
        start: pd.Timestamp, end: pd.Timestamp, funding: dict[str, pd.Series] | None = None) -> Result:  # fmt: skip
    funding = funding or {}
    timeline = data["BTCUSDT"].loc[start:end].index
    light_at = light.reindex(light.index.union(timeline)).ffill()
    month_members = {m: set(g["ma"]) for m, g in universe.groupby("thang")}
    cols = {s: {c: df[c].to_numpy() for c in df.columns} for s, df in data.items()}
    pos_index = {s: {t: i for i, t in enumerate(df.index)} for s, df in data.items()}
    fund_events = {s: f[(f.index > start) & (f.index <= end + H4)] for s, f in funding.items() if f is not None}

    equity = 1.0
    peak = 1.0
    month_key, month_start_eq = None, 1.0
    month_block = total_block = False
    positions: list[Pos] = []
    orders: list[Order] = []
    cooldown: dict[str, int] = {}
    trades, curve, events = [], [], []

    def close(pos: Pos, t, price: float, frac: float, reason: str, fee: float, slip: float):
        nonlocal equity
        px = price * (1 - pos.side * slip)
        q = pos.qty * frac
        pnl = pos.side * (px - pos.entry) * q - fee * px * q
        equity += pnl
        pos.pnl += pnl
        pos.qty -= q
        if pos.qty <= 1e-12:
            trades.append(
                {"sym": pos.sym, "side": "long" if pos.side > 0 else "short", "entry_t": pos.entry_t, "exit_t": t,
                 "entry": pos.entry, "stop0": pos.stop0, "exit": px, "pnl": pos.pnl, "r": pos.pnl / pos.risk_amt,
                 "risk_amt": pos.risk_amt, "reason": reason, "light": pos.light, "bars": pos.bars,
                 "partial": pos.partial, "major": pos.sym in MAJORS}
            )  # fmt: skip
            cooldown[pos.sym] = p.cooldown_bars

    for t in timeline:
        lt = light_at.get(t, np.nan)
        mk = t.strftime("%Y-%m")
        if mk != month_key:
            month_key, month_block = mk, False
            month_start_eq = _mtm(equity, positions, cols, pos_index, t, use_open=True)
        members = month_members.get(mk, set())
        for s in list(cooldown):
            cooldown[s] -= 1
            if cooldown[s] <= 0:
                del cooldown[s]

        # a) Đèn đổi sang chiều ngược: thoát ở giá mở cửa
        if p.context != "none":
            for pos in list(positions):
                i = pos_index[pos.sym].get(t)
                if i is None:
                    continue
                if (pos.side > 0 and lt == RED) or (pos.side < 0 and lt == GREEN):
                    close(pos, t, cols[pos.sym]["open"][i], 1.0, "den_doi", p.taker, _slip(pos.sym, p))
                    positions.remove(pos)

        # b) Lệnh chờ
        for od in list(orders):
            i = pos_index[od.sym].get(t)
            if i is None:
                orders.remove(od)
                continue
            o, h, lo = (cols[od.sym][k][i] for k in ("open", "high", "low"))
            gap = (o - od.limit) * od.side
            if (o - od.stop) * od.side <= 0:
                orders.remove(od)  # mở cửa đã vượt stop: bỏ lệnh
                continue
            if p.entry == "breakout" and gap > 0.5 * od.atr:
                orders.remove(od)  # giá chạy quá 0,5 ATR: không đuổi
                continue
            touched = lo <= od.limit <= h or (od.side > 0 and o <= od.limit) or (od.side < 0 and o >= od.limit)
            if touched:
                px = min(o, od.limit) if od.side > 0 else max(o, od.limit)
                fee = p.maker if p.entry == "pullback" else p.taker
                cost = fee * px * od.qty
                pos = Pos(od.sym, od.side, t, px, od.stop, od.stop, od.qty, od.risk_amt, od.light, pnl=-cost)
                equity -= cost
                positions.append(pos)
                orders.remove(od)
            else:
                od.expires -= 1
                if od.expires <= 0:
                    orders.remove(od)

        # c) Quản lý vị thế trong nến
        for pos in list(positions):
            i = pos_index[pos.sym].get(t)
            if i is None:
                last_t = data[pos.sym].index[-1]
                if t > last_t:  # hợp đồng ngừng giao dịch: đóng ở giá cuối cùng
                    close(pos, t, cols[pos.sym]["close"][-1], 1.0, "ngung_gd", p.taker, _slip(pos.sym, p))
                    positions.remove(pos)
                continue
            c_ = cols[pos.sym]
            o, h, lo, cl = c_["open"][i], c_["high"][i], c_["low"][i], c_["close"][i]
            pos.bars += 1
            r_unit = abs(pos.entry - pos.stop0)
            hit_stop = lo <= pos.stop if pos.side > 0 else h >= pos.stop
            if hit_stop:
                px = min(o, pos.stop) if pos.side > 0 else max(o, pos.stop)
                close(pos, t, px, 1.0, "stop" if not pos.partial else "stop_sau_2R", p.taker, _slip(pos.sym, p))
                positions.remove(pos)
                continue
            best = (h - pos.entry) if pos.side > 0 else (pos.entry - lo)
            pos.mfe_r = max(pos.mfe_r, best / r_unit)
            if not pos.partial and pos.mfe_r >= p.partial_r:
                target = pos.entry + pos.side * p.partial_r * r_unit
                close(pos, t, target, 0.5, "chot_2R", p.maker, 0.0)
                pos.partial, pos.stop = True, pos.entry
            # funding trong nến
            fe = fund_events.get(pos.sym)
            if fe is not None and len(fe):
                win = fe[(fe.index > t) & (fe.index <= t + H4)]
                for rate in win.to_numpy():
                    cost = pos.side * rate * cl * pos.qty
                    equity -= cost
                    pos.pnl -= cost
            # cuối nến: CHoCH, thời gian, trailing
            bos = c_["bos"][i]
            if (pos.side > 0 and bos == DOWN) or (pos.side < 0 and bos == UP):
                close(pos, t, cl, 1.0, "choch", p.taker, _slip(pos.sym, p))
                positions.remove(pos)
                continue
            if pos.bars >= p.time_stop_bars and pos.mfe_r < 1.0:
                close(pos, t, cl, 1.0, "het_gio", p.taker, _slip(pos.sym, p))
                positions.remove(pos)
                continue
            if pos.partial:
                lvl = c_["swing_low"][i] if pos.side > 0 else c_["swing_high"][i]
                if np.isfinite(lvl) and (lvl - pos.stop) * pos.side > 0 and (cl - lvl) * pos.side > 0:
                    pos.stop = lvl

        mtm = _mtm(equity, positions, cols, pos_index, t)
        curve.append((t + H4, mtm))
        peak = max(peak, mtm)
        if not month_block and mtm <= month_start_eq * (1 - p.month_stop):
            month_block = True
            events.append(f"{t:%Y-%m-%d}: ngắt mạch tháng ({mtm / month_start_eq - 1:+.1%})")
        if not total_block and mtm <= peak * (1 - p.total_stop):
            total_block = True
            events.append(f"{t:%Y-%m-%d}: chạm giới hạn sụt giảm tổng {p.total_stop:.0%} (đáng lẽ dừng hệ thống)")

        # d) Tín hiệu mới khi nến đóng
        if month_block or ((t + H4).hour in p.skip_hours_utc):
            continue
        lt_close = light_at.get(t + H4, lt)
        cands = []
        for sym in members:
            if p.majors_only and sym not in MAJORS:
                continue
            i = pos_index.get(sym, {}).get(t)
            if i is None:
                continue
            c_ = cols[sym]
            bos = c_["bos"][i]
            if bos is None or (isinstance(bos, float) and np.isnan(bos)):
                continue
            side = 1 if bos == UP else -1
            size = 1.0
            if p.context != "none":
                want = GREEN if side > 0 else RED
                if lt_close != want:
                    if p.yellow_trade and lt_close == YELLOW:
                        size = 0.5
                    else:
                        continue
            cl, atr = c_["close"][i], c_["atr"][i]
            if not np.isfinite(atr) or atr <= 0:
                continue
            d_up = c_["d_up"][i]
            if not np.isfinite(d_up) or (side > 0) != (d_up > 0.5):
                continue
            rs = c_["rs"][i]
            if sym != "BTCUSDT" and sym != "ETHUSDT" and not (np.isfinite(rs) and rs * side > 0):
                continue
            pos_c = c_["pos"][i]
            if not (c_["body"][i] >= p.body_min and ((pos_c >= 2 / 3) if side > 0 else (pos_c <= 1 / 3))):
                continue
            if not c_["vol_ok"][i]:
                continue
            broken = c_["swing_high"][i - 1] if side > 0 and i > 0 else c_["swing_low"][i - 1] if i > 0 else np.nan
            if not np.isfinite(broken) or (cl - broken) * side > p.max_chase_atr * atr:
                continue
            f = c_["fund"][i]
            hot = (side > 0 and f > p.funding_hot) or (side < 0 and f < p.funding_cold)
            if p.funding_filter and np.isfinite(f) and hot:
                continue
            base = c_["swing_low"][i] if side > 0 else c_["swing_high"][i]
            if not np.isfinite(base):
                continue
            stop = base - side * p.stop_buffer_atr * atr
            dist = (cl - stop) * side
            if dist <= 0 or dist > p.max_stop_atr * atr:
                continue
            limit = cl
            expires = 1
            if p.entry == "pullback":
                limit = cl - p.pullback_level * (cl - base)
                expires = p.pullback_bars
            cands.append((sym, side, limit, stop, atr, size, rs if np.isfinite(rs) else 0.0))
        cands.sort(key=lambda x: -x[1] * x[6])
        for sym, side, limit, stop, atr, size, _ in cands:
            busy = {x.sym for x in positions} | {x.sym for x in orders} | set(cooldown)
            if sym in busy or len(positions) + len(orders) >= p.max_positions:
                continue
            same_alts = sum(1 for x in [*positions, *orders] if x.side == side and x.sym not in MAJORS)
            if sym not in MAJORS and same_alts >= p.max_alts_same_side:
                continue
            open_risk = sum(_open_risk(x) for x in positions) + sum(x.risk_amt for x in orders)
            risk_amt = p.risk * size * equity
            if open_risk + risk_amt > p.max_open_risk * equity + 1e-12:
                continue
            qty = risk_amt / abs(limit - stop)
            cap = p.notional_cap * equity / limit
            if qty > cap:
                qty = cap
                risk_amt = qty * abs(limit - stop)
            orders.append(Order(sym, side, limit, stop, risk_amt, qty, atr, str(lt_close), expires))

    # đóng vị thế còn mở ở giá đóng cửa cuối kỳ
    for pos in list(positions):
        i = pos_index[pos.sym].get(timeline[-1])
        if i is not None:
            close(pos, timeline[-1], cols[pos.sym]["close"][i], 1.0, "het_ky", p.taker, _slip(pos.sym, p))
    eq = pd.Series(dict(curve), name="equity")
    return Result(pd.DataFrame(trades), eq, events)


def _open_risk(pos: Pos) -> float:
    return max(0.0, (pos.entry - pos.stop) * pos.side * pos.qty)


def _mtm(equity, positions, cols, pos_index, t, use_open=False) -> float:
    v = equity
    for pos in positions:
        i = pos_index[pos.sym].get(t)
        if i is None:
            continue
        px = cols[pos.sym]["open" if use_open else "close"][i]
        v += pos.side * (px - pos.entry) * pos.qty
    return v


def variant(base: BTParams, **kw) -> BTParams:
    return replace(base, **kw)
