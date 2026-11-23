import numpy as np
import pandas as pd
import pytest

from crypto_agent.xuhuong import giay


def _closes(n=420, drift=0.002, seed=1, end="2026-11-30"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=end, periods=n, freq="D")
    return pd.Series(50_000 * np.exp(np.cumsum(drift + 0.03 * rng.standard_normal(n))), index=idx)


def test_decide_long_flat_and_capped():
    up = giay.decide(_closes(drift=0.004))
    assert up["tin_hieu"] > 0 and 0 < up["muc_tieu"] <= 1
    assert up["muc_tieu"] == pytest.approx(min(1, up["tin_hieu"] * 0.15 / up["bien_dong"]))
    down = giay.decide(_closes(drift=-0.004))
    assert down["muc_tieu"] == 0.0


def test_rebalance_band_and_costs():
    btc, cash, act, trade, fee = giay.rebalance(0.0, 1.0, 100.0, 0.3)
    assert act == "mua" and btc * 100 == pytest.approx(0.3) and cash == pytest.approx(0.7 - fee)
    _, _, act2, _, _ = giay.rebalance(btc, cash, 100.0, 0.33)  # lệch 10% < 20%: giữ
    assert act2 == "giữ"
    btc3, _, act3, _, _ = giay.rebalance(btc, cash, 100.0, 0.0)
    assert act3 == "bán" and btc3 == 0


def test_step_lags_one_day_and_is_idempotent():
    closes = _closes(drift=0.004)
    led = giay.step(giay.load_ledger(pd.io.common.Path("/nonexistent.csv")), closes.loc[:"2026-11-22"])
    assert len(led) == 1 and led.iloc[0]["lenh"] == "chưa có" and led.iloc[0]["von"] == 1.0
    led2 = giay.step(led, closes.loc[:"2026-11-25"])
    assert list(led2["ngay"]) == ["2026-11-22", "2026-11-23", "2026-11-24", "2026-11-25"]
    first = led2.iloc[1]
    # sáng 23/11 thực hiện mục tiêu quyết ngày 22/11, ở giá đóng cửa 23/11
    assert first["ty_trong_sau"] == pytest.approx(led2.iloc[0]["muc_tieu_moi"], rel=1e-2)  # sau phí
    again = giay.step(led2, closes.loc[:"2026-11-25"])
    assert len(again) == 4  # chạy lại không ghi trùng
    text = giay.message(led2, 26_000)
    assert "B1 chạy giấy" in text and "mục tiêu sáng mai" in text


def test_step_needs_history():
    with pytest.raises(ValueError, match="Cần"):
        giay.step(pd.DataFrame(columns=giay.COLS), _closes(n=200))
