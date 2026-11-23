"""Chỉ số kết quả và đối chiếu tiêu chí đạt (đặc tả, phần Kế hoạch kiểm định)."""

from __future__ import annotations

import numpy as np
import pandas as pd

CRITERIA = {
    "ky_vong_r": 0.15,  # lợi nhuận kỳ vọng mỗi lệnh ≥ +0,15R
    "pf": 1.3,  # profit factor ≥ 1,3
    "so_lenh": 100,  # ≥ 100 lệnh
    "sut_giam": 0.10,  # sụt giảm tối đa ≤ 10%
    "nam_lai": 2 / 3,  # ≥ 2/3 số năm có lãi
}


def summarize(trades: pd.DataFrame, equity: pd.Series, start: pd.Timestamp, end: pd.Timestamp,
              btc_close: pd.Series) -> dict:  # fmt: skip
    eq = pd.concat([pd.Series([1.0], index=[start]), equity]).sort_index()
    years = max((end - start).days / 365.25, 1e-9)
    dd = float((eq / eq.cummax() - 1).min())
    # điểm vốn ghi ở giờ ĐÓNG nến: lùi 1 giây để nến đóng lúc 00:00 ngày 1/1 thuộc năm trước
    yearly = eq.set_axis(eq.index - pd.Timedelta(seconds=1)).resample("YE").last()
    prev = pd.concat([pd.Series([1.0]), yearly.iloc[:-1].reset_index(drop=True)])
    yr = pd.Series(yearly.to_numpy() / prev.to_numpy() - 1, index=yearly.index.year)
    b = btc_close.loc[start:end]
    out = {
        "so_lenh": int(len(trades)),
        "ty_le_thang": float((trades["r"] > 0).mean()) if len(trades) else np.nan,
        "ky_vong_r": float(trades["r"].mean()) if len(trades) else np.nan,
        "pf": _pf(trades),
        "loi_nhuan": float(eq.iloc[-1] - 1),
        "cagr": float(eq.iloc[-1] ** (1 / years) - 1) if eq.iloc[-1] > 0 else -1.0,
        "sut_giam": dd,
        "theo_nam": {int(k): float(v) for k, v in yr.items()},
        "nam_lai": float((yr > 0).mean()) if len(yr) else np.nan,
        "btc_giu": float(b.iloc[-1] / b.iloc[0] - 1) if len(b) > 1 else np.nan,
        "chuoi_thua": _losing_streak(trades),
    }
    for name, mask in _groups(trades).items():
        sub = trades[mask]
        out[f"{name}_n"] = int(len(sub))
        out[f"{name}_r"] = float(sub["r"].mean()) if len(sub) else np.nan
    out["ly_do"] = trades["reason"].value_counts().to_dict() if len(trades) else {}
    out["r_theo_ly_do"] = trades.groupby("reason")["r"].mean().round(2).to_dict() if len(trades) else {}
    if len(trades):
        dist = (trades["entry"] - trades["stop0"]).abs() / trades["entry"]
        out["stop_trung_vi"] = float(dist.median())
    return out


def _groups(t: pd.DataFrame) -> dict[str, pd.Series]:
    if t.empty:
        return {}
    return {
        "long": t["side"].eq("long"), "short": t["side"].eq("short"),
        "btc_eth": t["major"], "altcoin": ~t["major"],
    }  # fmt: skip


def _pf(t: pd.DataFrame) -> float:
    if t.empty:
        return np.nan
    win, loss = t.loc[t["pnl"] > 0, "pnl"].sum(), -t.loc[t["pnl"] < 0, "pnl"].sum()
    return float(win / loss) if loss > 0 else np.inf


def _losing_streak(t: pd.DataFrame) -> int:
    best = cur = 0
    for r in t.sort_values("exit_t")["r"] if len(t) else []:
        cur = cur + 1 if r <= 0 else 0
        best = max(best, cur)
    return best


def check(s: dict) -> dict[str, bool]:
    return {
        "ky_vong_r": bool(s["ky_vong_r"] >= CRITERIA["ky_vong_r"]),
        "pf": bool(s["pf"] >= CRITERIA["pf"]),
        "so_lenh": bool(s["so_lenh"] >= CRITERIA["so_lenh"]),
        "sut_giam": bool(-s["sut_giam"] <= CRITERIA["sut_giam"]),
        "nam_lai": bool(s["nam_lai"] >= CRITERIA["nam_lai"]),
        "long_short": bool(all(not (s.get(f"{k}_n", 0) and s.get(f"{k}_r", 0) < 0) for k in ("long", "short"))),
    }
