import numpy as np
import pandas as pd
import pytest


def make_daily(n: int, drift: float, seed: int = 0, start: str = "2025-01-01", price: float = 100.0) -> pd.DataFrame:
    """Nến ngày giả: xu hướng `drift` mỗi ngày + sóng hình sin để có đỉnh/đáy swing rõ."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    close = price * np.exp(drift * t + 0.04 * np.sin(t / 3.0) + rng.normal(0, 0.003, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC", name="open_time")
    vol = np.full(n, 1000.0)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol, "turnover": vol * close}, index=idx
    )


@pytest.fixture
def daily():
    return make_daily
