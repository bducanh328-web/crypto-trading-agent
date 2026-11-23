import io
import zipfile
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from crypto_agent.context import GREEN, RED, YELLOW, breadth, build_context, confirm, funding_per_8h
from crypto_agent.data.binance_vision import BinanceVision, parse_klines
from crypto_agent.data.bitunix import BitunixPublic, normalize_klines, split_volume
from crypto_agent.data.cache import KlineCache
from crypto_agent.report import daily_status
from crypto_agent.structure import DOWN, UP, structure
from crypto_agent.universe import eligible_pairs, rank_by_turnover

from conftest import make_daily


# --- Cấu trúc swing / BOS ----------------------------------------------------------------------------------------
def test_structure_detects_up_and_down_trends():
    assert structure(make_daily(120, 0.01))["trend"].iloc[-1] == UP
    assert structure(make_daily(120, -0.01))["trend"].iloc[-1] == DOWN


def test_structure_has_no_lookahead():
    df = make_daily(150, 0.005, seed=3)
    full = structure(df)
    for cut in (60, 90, 120):
        part = structure(df.iloc[:cut])
        pd.testing.assert_frame_equal(part, full.iloc[:cut])


def test_bos_needs_close_beyond_swing():
    # Đỉnh swing 110 ở nến 5; nến sau chỉ có râu vượt 110 (đóng cửa 109) -> chưa BOS; đóng 111 -> BOS tăng
    closes = [100, 102, 104, 106, 108, 109, 107, 105, 104, 103, 106, 109, 111]
    highs = [c + 1 for c in closes]
    highs[11] = 112
    df = pd.DataFrame({"high": highs, "low": [c - 1 for c in closes], "close": closes})
    s = structure(df, n=3)
    assert s["swing_high"].iloc[8] == 110
    assert pd.isna(s["bos"].iloc[11]) and s["bos"].iloc[12] == UP


# --- Nguồn dữ liệu -----------------------------------------------------------------------------------------------
def test_split_volume_detects_field_meaning():
    price = pd.Series([100.0, 200.0])
    coin, usdt = pd.Series([5.0, 4.0]), pd.Series([500.0, 800.0])
    c1, u1 = split_volume(usdt, coin, price)  # quoteVol = USDT (như tên gọi)
    c2, u2 = split_volume(coin, usdt, price)  # mẫu tài liệu: đặt ngược
    assert list(u1) == list(u2) == [500.0, 800.0] and list(c1) == list(c2) == [5.0, 4.0]


def test_bitunix_klines_pages_backwards():
    day = 86_400_000
    all_rows = [
        {"time": i * day, "open": "1", "high": "2", "low": "0.5", "close": "1.5", "quoteVol": "15", "baseVol": "10"}
        for i in range(450)
    ]
    calls = []

    def fake_get(url, params):
        calls.append(dict(params))
        end = params.get("endTime", 450 * day)
        rows = [r for r in all_rows if r["time"] < end][-params["limit"] :]
        return {"code": 0, "data": list(reversed(rows)), "msg": "Success"}

    df = BitunixPublic(get=fake_get, pause=0).klines("BTCUSDT", "1d", 400)
    assert len(df) == 400 and df.index.is_monotonic_increasing and df.index[-1].value // 10**6 == 449 * day
    assert len(calls) == 2 and calls[1]["endTime"] == 250 * day
    assert df["turnover"].iloc[0] == 15.0  # 10 coin × 1,5 = 15 USDT


def test_bitunix_error_code_raises():
    api = BitunixPublic(get=lambda url, params: {"code": 10001, "msg": "bad"}, pause=0)
    with pytest.raises(RuntimeError, match="code=10001"):
        api.trading_pairs()


def _zip_csv(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("x.csv", text)
    return buf.getvalue()


def test_binance_vision_months_days_and_formats():
    header = (
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
        "taker_buy_volume,taker_buy_quote_volume,ignore\n"
    )
    jan = header + "1704067200000,1,2,0.5,1.5,10,0,15,1,1,1,0\n"  # có tiêu đề, mili giây
    feb_day = "1706745600000000,2,3,1,2.5,10,0,25,1,1,1,0\n"  # không tiêu đề, micro giây
    seen = []

    def fetch(url):
        seen.append(url)
        if url.endswith("BTCUSDT-1d-2024-01.zip"):
            return _zip_csv(jan)
        if url.endswith("BTCUSDT-1d-2024-02-01.zip"):
            return _zip_csv(feb_day)
        return None

    df = BinanceVision(fetch=fetch).klines("BTCUSDT", "1d", date(2024, 1, 1), date(2024, 2, 2))
    assert list(df.index.strftime("%Y-%m-%d")) == ["2024-01-01", "2024-02-01"]
    assert df["turnover"].tolist() == [15.0, 25.0]
    assert any("/monthly/" in u for u in seen) and any("/daily/" in u and "2024-02-02" in u for u in seen)


def test_cache_merge_dedups(tmp_path):
    cache = KlineCache(tmp_path)
    a, b = make_daily(10, 0.0), make_daily(15, 0.0)
    cache.merge("X", "1d", a)
    out = cache.merge("X", "1d", b.iloc[5:])
    assert len(out) == 15 and out.index.is_unique
    assert len(cache.load("X", "1d")) == 15


def test_parse_klines_empty():
    assert parse_klines(pd.DataFrame()).empty and normalize_klines([]).empty


# --- Rổ coin -----------------------------------------------------------------------------------------------------
def test_eligible_pairs_filters():
    now = datetime(2026, 11, 23, tzinfo=timezone.utc)
    ms = now.timestamp() * 1000
    pairs = pd.DataFrame(
        [
            {"symbol": "BTCUSDT", "base": "BTC", "quote": "USDT", "symbolStatus": "OPEN", "isApiSupported": True,
             "launchTime": ms - 900 * 86_400_000, "delistTime": 0},
            {"symbol": "NEWUSDT", "base": "NEW", "quote": "USDT", "symbolStatus": "OPEN", "isApiSupported": True,
             "launchTime": ms - 30 * 86_400_000, "delistTime": 0},
            {"symbol": "USDCUSDT", "base": "USDC", "quote": "USDT", "symbolStatus": "OPEN", "isApiSupported": True,
             "launchTime": ms - 900 * 86_400_000, "delistTime": 0},
            {"symbol": "OLDUSDT", "base": "OLD", "quote": "USDT", "symbolStatus": "CANCEL_ONLY", "isApiSupported": True,
             "launchTime": ms - 900 * 86_400_000, "delistTime": 0},
            {"symbol": "APIUSDT", "base": "API", "quote": "USDT", "symbolStatus": "OPEN", "isApiSupported": False,
             "launchTime": ms - 900 * 86_400_000, "delistTime": 0},
        ]
    )  # fmt: skip
    assert eligible_pairs(pairs, now)["symbol"].tolist() == ["BTCUSDT"]


def test_rank_by_turnover():
    d = {f"C{i}": make_daily(40, 0.0, price=10.0 * (i + 1)) for i in range(5)}
    ranked = rank_by_turnover(d, d["C0"].index[-1])
    assert ranked[:2] == ["C4", "C3"]


# --- Lớp bối cảnh ------------------------------------------------------------------------------------------------
def test_confirm_needs_two_days():
    raw = pd.Series([YELLOW, GREEN, YELLOW, GREEN, GREEN, GREEN, RED])
    c = confirm(raw, 2)
    assert c["light"].tolist() == [YELLOW, YELLOW, YELLOW, YELLOW, GREEN, GREEN, GREEN]
    assert c["days"].tolist() == [1, 2, 3, 4, 2, 3, 4]


def test_breadth_requires_enough_coins():
    few = {f"C{i}": make_daily(80, 0.01, seed=i) for i in range(5)}
    assert breadth(few, 50).dropna().empty
    many = {f"C{i}": make_daily(80, 0.01, seed=i) for i in range(12)}
    assert breadth(many, 50).dropna().iloc[-1] > 0.9


def test_funding_normalized_to_8h():
    idx = pd.date_range("2026-11-20", periods=24, freq="4h", tz="UTC")
    rates = pd.Series(0.0001, index=idx)  # mỗi 4 giờ 0,01% = 0,02% / 8h
    now = idx[-1] + pd.Timedelta(hours=1)
    assert funding_per_8h(rates, 72, now) == pytest.approx(0.0002)


@pytest.mark.parametrize("drift,expected", [(0.004, GREEN), (-0.004, RED)])
def test_build_context_lights(drift, expected):
    btc = make_daily(400, drift, price=50_000)
    basket = {f"C{i}USDT": make_daily(200, drift, seed=i, start="2025-06-06") for i in range(15)}
    idx = pd.date_range("2026-02-01", periods=12, freq="8h", tz="UTC")
    funding = {"C0USDT": pd.Series(0.001, index=idx)}  # 0,1% / 8h: quá nóng
    ctx = build_context(btc, basket, funding)
    assert ctx.light == expected
    assert ctx.allowed("long", "C1USDT") == (expected == GREEN)
    assert ctx.allowed("short", "C1USDT") == (expected == RED)
    assert not ctx.allowed("long", "C0USDT")
    text = daily_status(ctx, list(basket))
    assert "Đèn crypto" in text and expected in text and "Không long" in text


def test_build_context_needs_history():
    with pytest.raises(ValueError, match="Chưa đủ dữ liệu"):
        build_context(make_daily(100, 0.01), {"A": make_daily(100, 0.01)}, {})


def test_yellow_when_breadth_disagrees():
    btc = make_daily(400, 0.004, price=50_000)
    basket = {f"C{i}USDT": make_daily(200, -0.004, seed=i, start="2025-06-06") for i in range(15)}
    ctx = build_context(btc, basket, {})
    assert ctx.light == YELLOW and np.isfinite(ctx.breadth50)
