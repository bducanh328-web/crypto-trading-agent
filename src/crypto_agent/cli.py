"""Dòng lệnh `cta`.

cta den [--telegram]          đèn chung hằng ngày (sau khi nến 1D đóng lúc 07:00 giờ Việt Nam)
cta lich-su BTCUSDT ETHUSDT   tải lịch sử dài từ data.binance.vision vào bộ đệm (cho kiểm định giai đoạn 2)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from .config import PARAMS
from .context import build_context
from .data.binance_vision import BinanceVision
from .data.bitunix import BitunixPublic, split_volume
from .data.cache import KlineCache
from .report import daily_status
from .universe import eligible_pairs, rank_by_turnover, utc_now

log = logging.getLogger("crypto_agent")
PRESELECT = 60  # lọc sơ bộ theo GTGD 24h trước khi lấy nến ngày để xếp hạng 30 ngày
BASKET_BARS = 200
BTC_BARS = 400


def cmd_den(args: argparse.Namespace) -> int:
    api = BitunixPublic()
    data_dir = Path(args.data_dir)
    cache = KlineCache(data_dir / "bitunix")
    now = utc_now()

    pairs = eligible_pairs(api.trading_pairs(), now)
    tick = api.tickers()
    tick = tick[tick["symbol"].isin(set(pairs["symbol"]))].reset_index(drop=True)
    _, usdt = split_volume(tick["quoteVol"], tick["baseVol"], tick["lastPrice"])
    tick["turnover24h"] = usdt.to_numpy()
    candidates = tick.sort_values("turnover24h", ascending=False)["symbol"].head(PRESELECT).tolist()
    if PARAMS.btc_symbol not in candidates:
        candidates.insert(0, PARAMS.btc_symbol)

    daily = {}
    for sym in candidates:
        try:
            bars = BTC_BARS if sym == PARAMS.btc_symbol else BASKET_BARS
            daily[sym] = cache.merge(sym, "1d", api.klines(sym, "1d", bars))
        except Exception as exc:  # một mã lỗi không làm hỏng cả bản tin
            log.warning("%s: không lấy được nến ngày (%s)", sym, exc)
    asof = max(df.index[-1] for df in daily.values() if len(df))
    universe = rank_by_turnover(daily, asof)
    basket = {s: daily[s] for s in universe}

    funding = {}
    start_ms = int((now - pd.Timedelta(days=4)).timestamp() * 1000)
    for sym in universe:
        try:
            funding[sym] = api.funding_history(sym, start_ms)
        except Exception as exc:
            log.warning("%s: không lấy được funding (%s)", sym, exc)

    ctx = build_context(daily[PARAMS.btc_symbol], basket, funding)
    text = daily_status(ctx, universe)
    print(text)
    _save(data_dir, ctx, universe)
    if args.summary:
        Path(args.summary).write_text(text + "\n", encoding="utf-8")
    if args.telegram:
        from .telegram import send

        send(text)
    return 0


def _save(data_dir: Path, ctx, universe: list[str]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    hist = ctx.history[["close", "ma200", "trend", "breadth", "raw", "light", "days"]].copy()
    hist.index = hist.index.strftime("%Y-%m-%d")
    hist.to_csv(data_dir / "den.csv", index_label="ngay")
    day = (ctx.asof + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    path = data_dir / "ro_coin.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["ngay", "hang", "ma"])
    new = pd.DataFrame({"ngay": day, "hang": range(1, len(universe) + 1), "ma": universe})
    pd.concat([old[old["ngay"] != day], new]).to_csv(path, index=False)


def cmd_lich_su(args: argparse.Namespace) -> int:
    vision = BinanceVision()
    cache = KlineCache(Path(args.data_dir) / "binance")
    start = date.fromisoformat(args.tu)
    end = date.fromisoformat(args.den) if args.den else date.today()
    for sym in args.ma:
        for interval in args.khung.split(","):
            df = vision.klines(sym, interval, start, end)
            merged = cache.merge(sym, interval, df) if len(df) else df
            span = f"{merged.index.min()} → {merged.index.max()}" if len(merged) else "-"
            log.info("%s %s: %d nến (%s)", sym, interval, len(merged), span)
        if args.funding:
            f = vision.funding(sym, start, end)
            out = Path(args.data_dir) / "binance" / "funding" / f"{sym}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            f.rename("funding_rate").to_csv(out, index_label="time")
            log.info("%s funding: %d kỳ", sym, len(f))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="cta", description="Hệ thống crypto: Lớp bối cảnh + BOS hai chiều")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("den", help="Đèn chung hằng ngày + bảng trạng thái")
    d.add_argument("--data-dir", default="data")
    d.add_argument("--telegram", action="store_true", help="Gửi Telegram (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)")
    d.add_argument("--summary", default="", help="Ghi bản tin ra tệp (vd. $GITHUB_STEP_SUMMARY)")
    d.set_defaults(func=cmd_den)

    h = sub.add_parser("lich-su", help="Tải lịch sử từ data.binance.vision")
    h.add_argument("ma", nargs="+", help="Mã hợp đồng, vd. BTCUSDT ETHUSDT")
    h.add_argument("--khung", default="1d,4h")
    h.add_argument("--tu", default="2020-01-01")
    h.add_argument("--den", default="")
    h.add_argument("--funding", action="store_true")
    h.add_argument("--data-dir", default="data")
    h.set_defaults(func=cmd_lich_su)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
