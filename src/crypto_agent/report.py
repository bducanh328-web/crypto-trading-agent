"""Bảng trạng thái hằng ngày (Telegram, HTML)."""

from __future__ import annotations

import html
import math

import pandas as pd

from .config import PARAMS, Params
from .context import GREEN, RED, Context
from .structure import DOWN, UP

ICON = {GREEN: "🟢", RED: "🔴"}
TREND = {UP: "tăng", DOWN: "giảm", None: "chưa rõ"}
VN = pd.Timedelta(hours=7)


def _pct(x: float, sign: bool = False) -> str:
    if x is None or not math.isfinite(x):
        return "–"
    s = f"{x * 100:+.1f}%" if sign else f"{x * 100:.0f}%"
    return s.replace(".", ",")


def _rate(x: float) -> str:
    return "–" if not math.isfinite(x) else f"{x * 100:+.3f}%".replace(".", ",")


def _price(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".") if x >= 100 else f"{x:.4g}".replace(".", ",")


def daily_status(ctx: Context, universe: list[str], p: Params = PARAMS) -> str:
    day = (ctx.asof + pd.Timedelta(days=1) + VN).strftime("%d/%m/%Y")
    action = {GREEN: "chỉ mở LONG", RED: "chỉ mở SHORT"}.get(ctx.light, "không mở lệnh mới")
    lines = [
        f"{ICON.get(ctx.light, '🟡')} <b>Đèn crypto {day}: {ctx.light}</b> ({ctx.days} ngày) – {action}",
    ]
    if ctx.raw != ctx.light:
        lines.append(f"   Điều kiện hôm nay: {ctx.raw} (cần giữ {p.confirm_days} ngày mới đổi đèn)")
    lines += [
        "",
        f"BTC {_price(ctx.btc_close)} · so MA200 {_pct(ctx.btc_vs_ma, True)} · cấu trúc {TREND.get(ctx.btc_trend)}",
        f"Độ rộng rổ {len(universe)} coin: trên MA50 {_pct(ctx.breadth50)} · trên MA200 {_pct(ctx.breadth200)}",
        f"   XANH cần ≥ {_pct(p.green_breadth)}, ĐỎ cần ≤ {_pct(p.red_breadth)}",
    ]
    fund = {s: v for s, v in ctx.funding.items() if math.isfinite(v)}
    if fund:
        ranked = sorted(fund.items(), key=lambda kv: kv[1], reverse=True)
        hot = ", ".join(f"{html.escape(s.removesuffix('USDT'))} {_rate(v)}" for s, v in ranked[:5])
        cold = ", ".join(f"{html.escape(s.removesuffix('USDT'))} {_rate(v)}" for s, v in ranked[::-1][:5])
        lines += ["", f"Funding 3 ngày (mỗi 8h) cao nhất: {hot}", f"Thấp nhất: {cold}"]
        blocked_long = [s.removesuffix("USDT") for s, v in fund.items() if v > p.funding_hot]
        blocked_short = [s.removesuffix("USDT") for s, v in fund.items() if v < p.funding_cold]
        if blocked_long:
            lines.append(f"⛔ Không long (funding > {_rate(p.funding_hot)}): {', '.join(blocked_long)}")
        if blocked_short:
            lines.append(f"⛔ Không short (funding < {_rate(p.funding_cold)}): {', '.join(blocked_short)}")
    lines += ["", "<i>Giai đoạn 1: chỉ báo bối cảnh, chưa có tín hiệu vào lệnh. Thông số chưa kiểm định.</i>"]
    return "\n".join(lines)
