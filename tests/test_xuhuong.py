import numpy as np
import pandas as pd
import pytest

from crypto_agent.xuhuong.data import daily_funding, parse_stooq, parse_yahoo
from crypto_agent.xuhuong.engine import ASSET_VOL, MAX_W, signal, simulate, stats, target_weights
from crypto_agent.xuhuong.runner import evaluate, report


def _walk(n, seed, drift, vol, start="2016-01-01", freq="D"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.Series(100 * np.exp(np.cumsum(drift + vol * rng.standard_normal(n))), index=idx)


def test_signal_levels():
    up = pd.Series(np.arange(1, 400, dtype=float), index=pd.date_range("2020-01-01", periods=399))
    s = signal(up)
    assert s.iloc[:250].isna().all() and s.iloc[-1] == 1.0
    mixed = up.copy()
    mixed.iloc[-30:] = mixed.iloc[-31] - 2 * np.arange(1, 31)  # giảm mạnh 30 ngày: khung 20, 60 âm; 120, 250 dương
    assert signal(mixed).iloc[-1] == 0.0
    mixed.iloc[-30:] = mixed.iloc[-31] + 0.0  # đi ngang 30 ngày: khung 20 bằng 0, còn lại dương
    assert signal(mixed).iloc[-1] == 0.75


def test_target_weights_vol_scaled_and_capped():
    p = _walk(800, 1, 0.002, 0.03)
    w = target_weights(p, 365, "long_short").dropna()
    assert w.abs().max() <= MAX_W + 1e-12
    calm = _walk(800, 1, 0.002, 0.001)
    assert target_weights(calm, 365, "long_short").dropna().abs().max() == pytest.approx(MAX_W)
    lf = target_weights(_walk(800, 2, -0.002, 0.03), 365, "long_flat").dropna()
    assert (lf >= 0).all()
    al = target_weights(p, 365, "always_long").dropna()
    assert (al > 0).all()


def test_simulate_one_day_lag_and_costs():
    cal = pd.date_range("2024-01-01", periods=5)
    px = pd.Series([100.0, 110, 121, 121, 121], index=cal)
    tg = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=cal)
    res = simulate({"A": px}, {"A": tg}, {"A": pd.Series(0.0, index=cal)}, cal)
    # quyết định ngày 1, thực hiện ở giá đóng cửa ngày 2 -> chỉ hưởng lợi nhuận từ ngày 3 (+10%)
    fee = 0.0011
    assert res.equity.iloc[1] == pytest.approx(1 - fee)
    assert res.equity.iloc[2] == pytest.approx((1 - fee) * 1.1, rel=1e-3)
    assert res.trades >= 1


def test_simulate_band_limits_trades():
    p = _walk(1500, 3, 0.0005, 0.02)
    cal = p.index
    tg = target_weights(p, 365, "long_short")
    a = simulate({"A": p}, {"A": tg}, {"A": pd.Series(0.0, index=cal)}, cal)
    b = simulate({"A": p}, {"A": tg}, {"A": pd.Series(0.0, index=cal)}, cal, band=0.0)
    assert a.trades < b.trades / 2


def test_funding_charged_by_side():
    cal = pd.date_range("2024-01-01", periods=10)
    px = pd.Series(100.0, index=cal)
    f = pd.Series(0.001, index=cal)
    long_ = simulate({"A": px}, {"A": pd.Series(1.0, index=cal)}, {"A": f}, cal)
    short = simulate({"A": px}, {"A": pd.Series(-1.0, index=cal)}, {"A": f}, cal)
    assert long_.funding > 0 and short.funding < 0
    assert short.equity.iloc[-1] > long_.equity.iloc[-1]


def test_stats_basic():
    eq = pd.Series(np.r_[np.linspace(1, 1.2, 400), np.linspace(1.2, 1.0, 100), np.linspace(1.0, 1.3, 600)],
                   index=pd.date_range("2020-01-01", periods=1100))  # fmt: skip
    s = stats(eq)
    assert s["sut_giam"] == pytest.approx(1.0 / 1.2 - 1, rel=1e-6) and s["cham_15"]
    assert s["so_nam_tron"] == 3 and 0 < s["sharpe"]


def test_daily_funding_fill_mean():
    ev = pd.Series([0.0001, 0.0001, 0.0001], index=pd.to_datetime(["2024-01-02 00:00", "2024-01-02 08:00",
                                                                      "2024-01-02 16:00"], utc=True))  # fmt: skip
    cal = pd.date_range("2024-01-01", periods=3)
    d, mean_day = daily_funding(ev, cal, fill_mean=True)
    assert d.loc["2024-01-02"] == pytest.approx(0.0003) and mean_day == pytest.approx(0.0003)
    assert d.loc["2024-01-01"] == pytest.approx(0.0003)


def test_parsers():
    s = parse_stooq("Date,Open,High,Low,Close\n2024-01-02,1,1,1,2063.5\n2024-01-03,1,1,1,2042.1\n")
    assert s.iloc[-1] == 2042.1
    body = {"chart": {"result": [{"timestamp": [1704205800, 1704292200],
                                   "indicators": {"quote": [{"close": [2073.4, None]}]}}]}}  # fmt: skip
    assert len(parse_yahoo(body)) == 1


def test_evaluate_end_to_end():
    btc = _walk(3300, 5, 0.0015, 0.035, start="2017-08-17")
    gidx = pd.bdate_range("2003-01-01", "2026-10-31")
    xau = pd.Series(_walk(len(gidx), 6, 0.0003, 0.01).to_numpy(), index=gidx)
    ev = pd.Series(0.0001, index=pd.date_range("2019-09-10", "2026-10-31", freq="8h", tz="UTC"))
    data = {"btc": btc, "xau": xau, "btc_funding": ev, "xau_funding_v2": ev.iloc[-300:], "xau_nguon": "giả lập"}
    o = evaluate(data)
    assert len(o["V1_long_short"]["dat"]) == 6
    assert o["giai_doan"][0] > "2018-04" and o["giai_doan"][1] == "2026-10-31"
    assert 0.03 < o["V1_long_short"]["bien_dong"] < 0.12  # quanh mục tiêu 6,5%
    assert "Tổng:" in report(o)
    assert ASSET_VOL == pytest.approx(0.046)


def test_bitunix_funding_pages_backwards():
    from crypto_agent.xuhuong.data import bitunix_funding_events

    full = pd.Series(0.0001, index=pd.date_range("2025-01-01", periods=1000, freq="4h", tz="UTC"))

    class Api:
        calls = 0

        def funding_history(self, symbol, start_ms, end_ms=None):
            Api.calls += 1
            ms = full.index.as_unit("ms").asi8
            s = full[(ms >= start_ms) & (ms <= end_ms)]
            return s.iloc[-200:]

    got = bitunix_funding_events(api=Api())
    assert len(got) == 1000 and Api.calls >= 5


def test_evaluate_spot_end_to_end():
    from crypto_agent.xuhuong.runner import evaluate_spot, report_spot

    btc = _walk(3300, 5, 0.0015, 0.035, start="2017-08-17")
    gidx = pd.bdate_range("2003-01-01", "2026-10-31")
    xau = pd.Series(_walk(len(gidx), 6, 0.0003, 0.01).to_numpy(), index=gidx)
    o = evaluate_spot({"btc": btc, "xau": xau, "xau_nguon": "giả lập"})
    s = o["S1_giao_ngay"]
    assert s["funding"] == 0 and s["ty_trong_max_tong"] <= 1.0 + 1e-9 and len(s["dat"]) == 6
    assert "Tổng:" in report_spot(o)


def test_evaluate_btc_end_to_end():
    from crypto_agent.xuhuong.runner import evaluate_btc, report_btc

    btc = _walk(3300, 5, 0.0015, 0.035, start="2017-08-17")
    o = evaluate_btc({"btc": btc})
    b = o["B1_btc_xu_huong"]
    assert len(b["dat"]) == 6 and 0 <= b["ty_trong_tb"] <= 1.0001
    assert 0.08 < b["bien_dong"] < 0.25  # quanh mục tiêu 15%
    assert "Tổng:" in report_btc(o)
