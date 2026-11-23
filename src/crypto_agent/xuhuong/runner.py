"""cta xu-huong: chạy đúng bản đặc tả chốt trước (V1, V2, M1–M4), chấm 6 tiêu chí, ghi báo cáo."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .data import daily_funding, load_all
from .engine import ANN, simulate, stats, target_weights

log = logging.getLogger(__name__)
END = pd.Timestamp("2026-10-31")
SPLIT = pd.Timestamp("2022-07-01")
GOLD_LONG_START = pd.Timestamp("2005-01-01")


def _sharpe(eq: pd.Series) -> float:
    r = eq.pct_change().dropna()
    return float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else np.nan


def evaluate(data: dict) -> dict:
    btc = data["btc"].loc[:END].astype(float)
    xau = data["xau"].loc[:END].astype(float)
    cal = pd.date_range(btc.index.min(), END, freq="D")
    f_btc, btc_mean = daily_funding(data["btc_funding"], cal, fill_mean=True)
    _, xau_mean = daily_funding(data["xau_funding"], cal, fill_mean=True)
    f_xau = pd.Series(xau_mean, index=cal)  # đặc tả: mức trung bình thực tế áp cho cả giai đoạn
    prices = {"BTC": btc, "XAU": xau}
    funding = {"BTC": f_btc, "XAU": f_xau}

    tg = {
        m: {a: target_weights(prices[a], ANN[a], m) for a in prices} for m in ("long_short", "long_flat", "always_long")
    }
    start = max(tg["long_short"][a].first_valid_index() for a in prices) + pd.Timedelta(days=1)
    period = cal[cal >= start]
    out: dict = {"giai_doan": [str(period[0].date()), str(period[-1].date())], "nguon_vang": data["xau_nguon"],
                 "funding_ngay": {"BTC_tb": btc_mean, "XAU_tb": xau_mean,
                                  "XAU_so_ky": int(len(data["xau_funding"]))}}  # fmt: skip
    runs = {}
    for name, mode in (
        ("V1_long_short", "long_short"),
        ("V2_long_dung_ngoai", "long_flat"),
        ("M4_luon_long", "always_long"),
    ):
        res = simulate(prices, {a: tg[mode][a].loc[start:] for a in prices}, funding, period)
        runs[name] = res
        s = stats(res.equity)
        s["so_giao_dich"], s["chi_phi"], s["funding"] = res.trades, res.costs, res.funding
        s["dong_gop"] = {a: float(res.contrib[a].iloc[-1]) for a in prices}
        s["ty_trong_tb"] = {a: float(res.weights[a].abs().mean()) for a in prices}
        s["sharpe_truoc"] = _sharpe(res.equity.loc[:SPLIT])
        s["sharpe_sau"] = _sharpe(res.equity.loc[SPLIT:])
        out[name] = s
    hold = {a: (prices[a].reindex(period).ffill() / prices[a].reindex(period).ffill().dropna().iloc[0]) for a in prices}
    out["M1_giu_BTC"] = stats(hold["BTC"].dropna())
    out["M2_giu_vang"] = stats(hold["XAU"].dropna())
    const = {a: pd.Series(0.5, index=prices[a].loc[start:].index) for a in prices}
    zero = {a: pd.Series(0.0, index=period) for a in prices}
    out["M3_giu_50_50"] = stats(simulate(prices, const, zero, period, monthly=True).equity)

    # Vàng chạy riêng 2005–10/2026 (tiêu chí 6), V1
    gcal = pd.date_range(GOLD_LONG_START, END, freq="D")
    gt = target_weights(xau, ANN["XAU"], "long_short")
    gstart = max(gt.first_valid_index() + pd.Timedelta(days=1), GOLD_LONG_START)
    g = simulate({"XAU": xau}, {"XAU": gt.loc[gstart:]}, {"XAU": pd.Series(xau_mean, index=gcal)},
                 gcal[gcal >= gstart])  # fmt: skip
    out["vang_rieng_2005"] = stats(g.equity)

    wk = pd.DataFrame({a: prices[a].loc[start:].resample("W-FRI").last() for a in prices}).pct_change().dropna()
    out["tuong_quan_tuan"] = float(wk.corr().iloc[0, 1])

    for name in ("V1_long_short", "V2_long_dung_ngoai"):
        s, m4 = out[name], out["M4_luon_long"]
        out[name]["dat"] = {
            "1_sut_giam_<=15%": bool(-s["sut_giam"] <= 0.15),
            "2_sharpe_>=0.6": bool(s["sharpe"] >= 0.6),
            "3_calmar_>=M4": bool(s["calmar"] >= m4["calmar"]),
            "4_nam_lai_>=2/3": bool(s["nam_tron_lai"] >= 2 / 3),
            "5_sharpe_truoc_sau_>0": bool(s["sharpe_truoc"] > 0 and s["sharpe_sau"] > 0),
            "6_vang_2005_sharpe_>0": bool(out["vang_rieng_2005"]["sharpe"] > 0),
        }
        out[name]["DAT_TAT_CA"] = all(out[name]["dat"].values())
    return out


def report(o: dict) -> str:
    def pct(x):
        return "–" if x is None or not np.isfinite(x) else f"{100 * x:+.1f}%"

    def num(x):
        return "–" if x is None or not np.isfinite(x) else f"{x:.2f}"

    lines = [
        "# Kết quả: BTC + XAU theo xu hướng ngày (đặc tả chốt trước 23/11/2026)", "",
        f"Giai đoạn giao dịch: {o['giai_doan'][0]} → {o['giai_doan'][1]}. Nguồn vàng: {o['nguon_vang']}.",
        f"Funding TB mỗi ngày: BTC {o['funding_ngay']['BTC_tb']:.5%}, XAU {o['funding_ngay']['XAU_tb']:.5%} "
        f"({o['funding_ngay']['XAU_so_ky']} kỳ). Tương quan tuần BTC–vàng: {o['tuong_quan_tuan']:.2f}", "",
        "| Chiến lược | CAGR | Biến động | Sharpe | Sụt giảm | Calmar | Năm lãi | Chạm −15% |",
        "|---|---|---|---|---|---|---|---|",
    ]  # fmt: skip
    for k in ("V1_long_short", "V2_long_dung_ngoai", "M4_luon_long", "M3_giu_50_50", "M1_giu_BTC", "M2_giu_vang",
              "vang_rieng_2005"):  # fmt: skip
        s = o[k]
        lines.append(
            f"| {k} | {pct(s['cagr'])} | {pct(s['bien_dong'])} | {num(s['sharpe'])} | {pct(s['sut_giam'])} "
            f"| {num(s['calmar'])} | {pct(s['nam_tron_lai'])} ({s['so_nam_tron']} năm) | {s['cham_15'] or 'không'} |"
        )
    for k in ("V1_long_short", "V2_long_dung_ngoai", "M4_luon_long"):
        s = o[k]
        lines += ["", f"## {k}", f"Theo năm: { {y: round(v, 4) for y, v in s['theo_nam'].items()} }",
                  f"Đóng góp: {s['dong_gop']} · Tỷ trọng TB: {s['ty_trong_tb']}",
                  f"Giao dịch: {s['so_giao_dich']} · Chi phí: {s['chi_phi']:.4f} · Funding: {s['funding']:.4f}",
                  f"Sharpe trước/sau 07/2022: {num(s['sharpe_truoc'])} / {num(s['sharpe_sau'])}"]  # fmt: skip
        if "dat" in s:
            lines += [f"- {c}: {'ĐẠT' if ok else 'KHÔNG ĐẠT'}" for c, ok in s["dat"].items()]
            lines.append(f"**Tổng: {'ĐẠT' if s['DAT_TAT_CA'] else 'KHÔNG ĐẠT'}**")
    lines += [
        "",
        f"Vàng riêng 2005: theo năm { {y: round(v, 4) for y, v in o['vang_rieng_2005']['theo_nam'].items()} }",
    ]
    return "\n".join(lines) + "\n"


def main_args(args) -> int:
    data = load_all(Path(args.data_dir) / "xuhuong")
    o = evaluate(data)
    out = Path(args.ket_qua)
    out.mkdir(parents=True, exist_ok=True)
    (out / "xu_huong.json").write_text(json.dumps(o, ensure_ascii=False, indent=1, default=str), "utf-8")
    md = report(o)
    (out / "xu_huong.md").write_text(md, "utf-8")
    print(md)
    return 0
