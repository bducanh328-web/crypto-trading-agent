import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from crypto_agent.backtest.engine import BTParams, daily_light, prepare, run, variant
from crypto_agent.backtest.history import all_usdt_symbols, list_keys, monthly_universe
from crypto_agent.backtest.metrics import check, summarize

START, END = pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2022-12-31 20:00", tz="UTC")


def _h4(seed: int, n: int, drift: float, price: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    ret = drift * np.sign(np.sin(t / 900)) + 0.02 * np.sin(t / 25) / 25 + rng.normal(0, 0.012, n)
    close = price * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    wig = np.abs(rng.normal(0, 0.006, n))
    high, low = np.maximum(open_, close) * (1 + wig), np.minimum(open_, close) * (1 - wig)
    vol = rng.lognormal(10, 0.6, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "turnover": vol * close}, index=idx)


def _daily(h4: pd.DataFrame) -> pd.DataFrame:
    d = h4.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last", "turnover": "sum"})
    return d.dropna()


@pytest.fixture(scope="module")
def market():
    n = 6 * 365 * 3
    syms = ["BTCUSDT", "ETHUSDT"] + [f"C{i}USDT" for i in range(13)]
    h4 = {s: _h4(i, n, 0.0012, 100.0 * (i + 1)) for i, s in enumerate(syms)}
    d1 = {s: _daily(df) for s, df in h4.items()}
    uni = monthly_universe(d1, pd.Timestamp("2020-08-01").date(), pd.Timestamp("2022-12-01").date(), size=30)
    idx = pd.date_range("2020-01-01", periods=n // 2, freq="8h", tz="UTC")
    fund = {s: pd.Series(0.0001, index=idx) for s in syms}
    return h4, d1, uni, fund


def _go(market, p):
    h4, d1, uni, fund = market
    prep = {s: prepare(s, h4[s], d1[s], h4["BTCUSDT"]["close"], fund[s], p) for s in h4}
    light = daily_light(d1["BTCUSDT"], d1, uni, p)
    return run(prep, light, uni, p, START, END, fund), d1


def test_universe_point_in_time(market):
    _, d1, uni, _ = market
    first = uni.groupby("thang")["ma"].count()
    assert first.index[0] == "2020-08" and first.iloc[0] == 15  # chỉ đủ 180 ngày từ tháng 7/2020
    assert monthly_universe(d1, pd.Timestamp("2020-03-01").date(), pd.Timestamp("2020-03-01").date()).empty


@pytest.mark.parametrize(
    "kw",
    [{}, {"context": "none"}, {"entry": "pullback"}, {"yellow_trade": True}, {"majors_only": True},
     {"skip_hours_utc": (20,)}],
)  # fmt: skip
def test_engine_runs_and_accounts(market, kw):
    p = variant(BTParams(), **kw)
    res, d1 = _go(market, p)
    t = res.trades
    assert np.isfinite(res.equity).all() and len(res.equity) > 1000
    if len(t):
        # tổng lãi lỗ các lệnh = thay đổi vốn (không tính lệnh còn mở: đã đóng cuối kỳ)
        assert t["pnl"].sum() == pytest.approx(res.equity.iloc[-1] - 1.0, abs=1e-9)
        plain_stops = t[(t["reason"] == "stop") & ~t["partial"]]
        if len(plain_stops):
            assert plain_stops["r"].max() < -0.9 and plain_stops["r"].median() > -1.6
        assert (t["exit_t"] >= t["entry_t"]).all()
        # rủi ro mỗi lệnh ≤ 0,5% vốn tại thời điểm vào (vốn ≤ đỉnh)
        assert (t["risk_amt"] <= 0.005 * res.equity.max() * 1.0001).all()
    s = summarize(t, res.equity, START, END, d1["BTCUSDT"]["close"])
    assert set(check(s)) >= {"ky_vong_r", "pf", "so_lenh", "sut_giam", "nam_lai"}


def test_no_context_trades_more(market):
    base, _ = _go(market, BTParams())
    free, _ = _go(market, variant(BTParams(), context="none"))
    assert len(free.trades) >= len(base.trades)


def test_s3_listing_pagination():
    page1 = """<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>true</IsTruncated>
      <NextMarker>data/futures/um/monthly/klines/ETHUSDT/</NextMarker>
      <CommonPrefixes><Prefix>data/futures/um/monthly/klines/BTCUSDT/</Prefix></CommonPrefixes>
      <CommonPrefixes><Prefix>data/futures/um/monthly/klines/ETHUSDT/</Prefix></CommonPrefixes></ListBucketResult>"""
    page2 = """<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>false</IsTruncated>
      <CommonPrefixes><Prefix>data/futures/um/monthly/klines/USDCUSDT/</Prefix></CommonPrefixes>
      <CommonPrefixes><Prefix>data/futures/um/monthly/klines/XAUUSDT/</Prefix></CommonPrefixes>
      <CommonPrefixes><Prefix>data/futures/um/monthly/klines/BTCBUSD/</Prefix></CommonPrefixes>
      <CommonPrefixes><Prefix>data/futures/um/monthly/klines/SOLUSDT/</Prefix></CommonPrefixes></ListBucketResult>"""
    calls = []

    def get(url):
        calls.append(url)
        return page2 if "marker=" in url else page1

    assert len(list_keys("data/futures/um/monthly/klines/", True, get)) == 6
    assert all_usdt_symbols(get) == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert "marker=data/futures/um/monthly/klines/ETHUSDT/" in calls[1]


def test_history_download_parses_zip(tmp_path):
    from crypto_agent.backtest.history import History

    def zbytes(text):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("a.csv", text)
        return buf.getvalue()

    listing = """<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>false</IsTruncated>
      <Contents><Key>data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip</Key></Contents>
      <Contents><Key>data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip.CHECKSUM</Key></Contents>
      <Contents><Key>data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2027-01.zip</Key></Contents>
      </ListBucketResult>"""
    row = "1704067200000,1,2,0.5,1.5,10,0,15,1,1,1,0\n"
    h = History(tmp_path, workers=2, get_text=lambda url: listing, get_bytes=lambda url: zbytes(row))
    h.download("1d", ["BTCUSDT"], pd.Timestamp("2020-01-01").date(), pd.Timestamp("2026-10-31").date())
    df = h.load("1d", "BTCUSDT")
    assert len(df) == 1 and df["turnover"].iloc[0] == 15.0
