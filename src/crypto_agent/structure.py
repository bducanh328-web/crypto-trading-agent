"""Cấu trúc swing và BOS (dùng chung cho đèn BTC khung 1D và tín hiệu khung 4H).

Đỉnh swing tại nến p: đỉnh cao hơn n nến bên trái và không thấp hơn n nến bên phải. Swing chỉ được biết sau khi đủ
n nến bên phải (tại nến p + n), nên không nhìn trước tương lai. BOS tăng: giá đóng cửa vượt đỉnh swing gần nhất
chưa bị phá; BOS giảm: đóng cửa thủng đáy swing gần nhất chưa bị phá.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UP, DOWN = "tang", "giam"


def structure(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Theo từng nến: đỉnh/đáy swing đã xác nhận gần nhất, BOS phát sinh ở nến đó, xu hướng cấu trúc sau nến đó."""
    h, lo, c = (df[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    size = len(df)
    sh_lvl = np.full(size, np.nan)
    sl_lvl = np.full(size, np.nan)
    bos = np.full(size, None, dtype=object)
    trend = np.full(size, None, dtype=object)
    last_sh = last_sl = np.nan
    sh_live = sl_live = False
    state = None
    for i in range(size):
        p = i - n
        if p >= n:
            left_h, right_h = h[p - n : p], h[p + 1 : i + 1]
            if h[p] > left_h.max() and h[p] >= right_h.max():
                last_sh, sh_live = h[p], True
            left_l, right_l = lo[p - n : p], lo[p + 1 : i + 1]
            if lo[p] < left_l.min() and lo[p] <= right_l.min():
                last_sl, sl_live = lo[p], True
        if sh_live and c[i] > last_sh:
            bos[i], state, sh_live = UP, UP, False
        elif sl_live and c[i] < last_sl:
            bos[i], state, sl_live = DOWN, DOWN, False
        sh_lvl[i], sl_lvl[i], trend[i] = last_sh, last_sl, state
    return pd.DataFrame({"swing_high": sh_lvl, "swing_low": sl_lvl, "bos": bos, "trend": trend}, index=df.index)
