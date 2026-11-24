"""Chạy giấy hệ thống B1 (đặc tả 3, chốt trước 23/11/2026): BTC giao ngay, long hoặc đứng ngoài, biến động 15%/năm.

Mỗi sáng (sau nến ngày đóng 07:00 giờ VN):
1. Thực hiện (trên giấy) tỷ trọng mục tiêu đã quyết hôm trước, ở giá đóng cửa nến vừa đóng, theo ngưỡng 20%.
2. Tính tỷ trọng mục tiêu mới từ giá đóng cửa đến nến vừa đóng; chờ thực hiện sáng mai (trễ 1 ngày như kiểm định).
3. Ghi sổ so_giay/b1.csv (vốn khởi điểm = 1,0 = 100 triệu đồng) và gửi Telegram.
Giá: nến ngày BTCUSDT của Bitunix (hợp đồng vĩnh cửu, chênh rất nhỏ so với giao ngay), dùng làm giá tham chiếu.
"""

from __future__ import annotations

import html
import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import BAND, LOOKBACKS, SLIP, realized_vol, signal

log = logging.getLogger(__name__)

VOL_TARGET = 0.15
FEE_SPOT = 0.0010
ANN_BTC = 365
CAPITAL_VND = 100_000_000
START = "2026-11-22"  # nến ngày 22/11 (đóng 07:00 VN 23/11): ngày đầu chạy giấy
COLS = ["ngay", "gia", "phieu", "tin_hieu", "bien_dong", "muc_tieu_moi", "lenh", "ty_trong_truoc", "ty_trong_sau",
        "giao_dich_von", "phi", "btc", "tien", "von", "dinh", "sut_giam"]  # fmt: skip


def decide(closes: pd.Series) -> dict:
    """Tín hiệu và tỷ trọng mục tiêu tại giá đóng cửa cuối cùng của `closes`."""
    votes = {n: int(np.sign(closes.iloc[-1] / closes.iloc[-1 - n] - 1)) for n in LOOKBACKS}
    s = float(signal(closes).iloc[-1])
    vol = float(realized_vol(closes, ANN_BTC).iloc[-1])
    target = min(1.0, max(0.0, s) * VOL_TARGET / vol) if np.isfinite(vol) and vol > 0 else 0.0
    return {"phieu": votes, "tin_hieu": s, "bien_dong": vol, "muc_tieu": target}


def rebalance(btc: float, cash: float, price: float, target: float) -> tuple[float, float, str, float, float]:
    """Áp quy tắc ngưỡng 20% (như kiểm định). Trả về btc, tiền, lệnh, giá trị giao dịch, phí."""
    eq = cash + btc * price
    cw = btc * price / eq if eq > 0 else 0.0
    go = (np.sign(target) != np.sign(cw)) or abs(target - cw) > BAND * abs(cw)
    if not go or abs(target - cw) < 1e-9:
        return btc, cash, "giữ", 0.0, 0.0
    trade = target * eq - btc * price  # > 0 mua, < 0 bán (giá trị, đơn vị vốn)
    fee = abs(trade) * (FEE_SPOT + SLIP)
    new_btc = target * eq / price
    new_cash = cash - trade - fee
    return new_btc, new_cash, ("mua" if trade > 0 else "bán"), trade, fee


def load_ledger(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype={"ngay": str})
    return pd.DataFrame(columns=COLS)


def step(ledger: pd.DataFrame, closes: pd.Series) -> pd.DataFrame:
    """Xử lý mọi ngày đã đóng chưa có trong sổ (bù cả những ngày bot không chạy)."""
    closes = closes.astype(float).sort_index()
    days = [d for d in closes.index if d.strftime("%Y-%m-%d") >= START]
    done = set(ledger["ngay"]) if len(ledger) else set()
    rows = ledger.to_dict("records")
    for d in days:
        key = d.strftime("%Y-%m-%d")
        if key in done:
            continue
        hist = closes.loc[:d]
        if len(hist) < max(LOOKBACKS) + 61:
            raise ValueError(f"Cần ≥ {max(LOOKBACKS) + 61} nến ngày, mới có {len(hist)}")
        price = float(hist.iloc[-1])
        prev = rows[-1] if rows else None
        btc = float(prev["btc"]) if prev else 0.0
        cash = float(prev["tien"]) if prev else 1.0
        peak = float(prev["dinh"]) if prev else 1.0
        before = btc * price / (cash + btc * price)
        action, trade, fee = "chưa có", 0.0, 0.0
        if prev is not None:
            btc, cash, action, trade, fee = rebalance(btc, cash, price, float(prev["muc_tieu_moi"]))
        d_ = decide(hist)
        eq = cash + btc * price
        peak = max(peak, eq)
        rows.append({
            "ngay": key, "gia": price, "phieu": " ".join(f"{n}:{v:+d}" for n, v in d_["phieu"].items()),
            "tin_hieu": d_["tin_hieu"], "bien_dong": d_["bien_dong"], "muc_tieu_moi": d_["muc_tieu"],
            "lenh": action, "ty_trong_truoc": before, "ty_trong_sau": btc * price / eq, "giao_dich_von": trade,
            "phi": fee, "btc": btc, "tien": cash, "von": eq, "dinh": peak, "sut_giam": eq / peak - 1,
        })  # fmt: skip
        done.add(key)
    return pd.DataFrame(rows, columns=COLS)


def _vnd(x: float) -> str:
    return f"{x * CAPITAL_VND / 1e6:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".") + " triệu"


def _price(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")


def _pct(x: float, sign: bool = False) -> str:
    if not math.isfinite(x):
        return "–"
    return (f"{x * 100:+.1f}%" if sign else f"{x * 100:.1f}%").replace(".", ",")


def message(ledger: pd.DataFrame, usdt_vnd: float) -> str:
    r = ledger.iloc[-1]
    day = pd.Timestamp(r["ngay"]) + pd.Timedelta(days=1)
    votes = r["phieu"].replace(":+1", " ↑").replace(":-1", " ↓").replace(":+0", " =")
    lines = [f"₿ <b>B1 chạy giấy {day:%d/%m/%Y}</b> (ngày {len(ledger)}/90)", ""]
    if r["lenh"] in ("mua", "bán"):
        usdt = abs(r["giao_dich_von"]) * CAPITAL_VND / usdt_vnd
        qty = usdt / r["gia"]
        lines.append(f"🔔 <b>Lệnh hôm nay: {r['lenh'].upper()} {qty:.5f} BTC</b> (~{_price(usdt)} USDT, "
                     f"{_vnd(abs(r['giao_dich_von']))}) quanh giá {_price(r['gia'])}".replace("0.", "0,"))  # fmt: skip
        lines.append(f"   Tỷ trọng BTC {_pct(r['ty_trong_truoc'])} → {_pct(r['ty_trong_sau'])} vốn")
    elif r["lenh"] == "giữ" and r["ty_trong_sau"] < 1e-9:
        lines.append("Không có lệnh: tiếp tục đứng ngoài, không nắm BTC")
    elif r["lenh"] == "giữ":
        lines.append(f"Không có lệnh: giữ BTC {_pct(r['ty_trong_sau'])} vốn (lệch chưa quá 20%)")
    else:
        lines.append("Ngày đầu: chưa có lệnh, tỷ trọng mục tiêu bên dưới sẽ thực hiện sáng mai")
    lines += [
        "",
        f"Giá đóng cửa {_price(r['gia'])} · Xu hướng {votes} → tín hiệu " + f"{r['tin_hieu']:+.2f}".replace(".", ","),
        f"Biến động 60 ngày {_pct(r['bien_dong'])}/năm → <b>mục tiêu sáng mai: BTC {_pct(r['muc_tieu_moi'])} vốn</b>",
        "",
        f"Vốn giấy {_vnd(r['von'])} ({_pct(r['von'] - 1, True)}) · Sụt giảm từ đỉnh {_pct(r['sut_giam'], True)}",
        "<i>Chạy giấy, chưa dùng tiền thật. Đặc tả 3 chốt trước 23/11/2026.</i>",
    ]
    return "\n".join(lines)


def main_args(args) -> int:
    from ..data.bitunix import BitunixPublic

    api = BitunixPublic()
    df = api.klines("BTCUSDT", "1d", 400)
    closes = df["close"].copy()
    closes.index = closes.index.tz_convert(None).normalize()
    path = Path(args.so)
    old = load_ledger(path)
    ledger = step(old, closes)
    is_new = len(ledger) > len(old)
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(path, index=False)
    rate = float(os.environ.get("USDT_VND", args.usdt_vnd))
    text = message(ledger, rate)
    print(text)
    if args.summary:
        Path(args.summary).write_text(text + "\n", encoding="utf-8")
    if args.telegram and args.chi_khi_moi and not is_new:
        log.info("Không có nến ngày mới (đã gửi bản tin hôm nay): bỏ qua Telegram")
        return 0
    if args.telegram:
        from ..telegram import CHAT_ENV, TOKEN_ENV, send

        chat = os.environ.get(CHAT_ENV, "").strip()
        if chat.startswith("-"):
            # ID âm là kênh hoặc nhóm: bản tin B1 chỉ gửi tin nhắn riêng với bot
            raise RuntimeError(f"{CHAT_ENV} đang là kênh/nhóm ({chat[:5]}…); cần ID chat riêng (số dương)")
        if os.environ.get(TOKEN_ENV) and chat:
            send(text, chat_id=chat)
        else:
            log.warning("Chưa đặt %s / %s: không gửi Telegram", TOKEN_ENV, CHAT_ENV)
    return 0


__all__ = ["decide", "rebalance", "step", "message", "html"]
