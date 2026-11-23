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
    _, xau_mean = daily_funding(data["xau_funding_v2"], cal, fill_mean=True)
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
                                  "XAU_so_ky": int(len(data["xau_funding_v2"]))}}  # fmt: skip
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


SPOT_FEE = 0.0010  # taker giao ngay Bitunix VIP0


def _cap_total(targets: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Tổng tỷ trọng các tài sản ≤ 1 (đặc tả 2): nếu vượt thì thu nhỏ cùng tỷ lệ."""
    frame = pd.DataFrame(targets).ffill().fillna(0.0)
    scale = (1.0 / frame.abs().sum(axis=1)).clip(upper=1.0)
    return {a: (s * scale.reindex(s.index)).where(s.notna()) for a, s in targets.items()}


def evaluate_spot(data: dict) -> dict:
    """Đặc tả 2: giao ngay, chỉ long hoặc đứng ngoài, không funding, không đòn bẩy."""
    btc = data["btc"].loc[:END].astype(float)
    xau = data["xau"].loc[:END].astype(float)
    cal = pd.date_range(btc.index.min(), END, freq="D")
    prices = {"BTC": btc, "XAU": xau}
    zero_cal = {a: pd.Series(0.0, index=cal) for a in prices}
    out: dict = {"nguon_vang": data["xau_nguon"]}
    tg = {}
    for name, mode in (("S1_giao_ngay", "long_flat"), ("M4g_luon_long", "always_long")):
        raw = {a: target_weights(prices[a], ANN[a], mode, max_w=1.0) for a in prices}
        tg[name] = _cap_total(raw)
    start = max(tg["S1_giao_ngay"][a].first_valid_index() for a in prices) + pd.Timedelta(days=1)
    period = cal[cal >= start]
    out["giai_doan"] = [str(period[0].date()), str(period[-1].date())]
    for name in ("S1_giao_ngay", "M4g_luon_long"):
        res = simulate(prices, {a: tg[name][a].loc[start:] for a in prices}, zero_cal, period, fee=SPOT_FEE)
        s = stats(res.equity)
        s["so_giao_dich"], s["chi_phi"], s["funding"] = res.trades, res.costs, res.funding
        s["dong_gop"] = {a: float(res.contrib[a].iloc[-1]) for a in prices}
        s["ty_trong_tb"] = {a: float(res.weights[a].abs().mean()) for a in prices}
        s["ty_trong_max_tong"] = float(res.weights.abs().sum(axis=1).max())
        s["sharpe_truoc"] = _sharpe(res.equity.loc[:SPLIT])
        s["sharpe_sau"] = _sharpe(res.equity.loc[SPLIT:])
        out[name] = s
    hold = {a: (prices[a].reindex(period).ffill() / prices[a].reindex(period).ffill().dropna().iloc[0]) for a in prices}
    out["M1_giu_BTC"] = stats(hold["BTC"].dropna())
    out["M2_giu_vang"] = stats(hold["XAU"].dropna())
    const = {a: pd.Series(0.5, index=prices[a].loc[start:].index) for a in prices}
    out["M3_giu_50_50"] = stats(simulate(prices, const, zero_cal, period, monthly=True, fee=SPOT_FEE).equity)
    gcal = pd.date_range(GOLD_LONG_START, END, freq="D")
    gt = target_weights(xau, ANN["XAU"], "long_flat", max_w=1.0)
    gstart = max(gt.first_valid_index() + pd.Timedelta(days=1), GOLD_LONG_START)
    g = simulate({"XAU": xau}, {"XAU": gt.loc[gstart:]}, {"XAU": pd.Series(0.0, index=gcal)}, gcal[gcal >= gstart],
                 fee=SPOT_FEE)  # fmt: skip
    out["vang_rieng_2005"] = stats(g.equity)
    s, m4 = out["S1_giao_ngay"], out["M4g_luon_long"]
    s["dat"] = {
        "1_sut_giam_<=15%": bool(-s["sut_giam"] <= 0.15),
        "2_sharpe_>=0.6": bool(s["sharpe"] >= 0.6),
        "3_calmar_>=M4g": bool(s["calmar"] >= m4["calmar"]),
        "4_nam_lai_>=2/3": bool(s["nam_tron_lai"] >= 2 / 3),
        "5_sharpe_truoc_sau_>0": bool(s["sharpe_truoc"] > 0 and s["sharpe_sau"] > 0),
        "6_vang_2005_sharpe_>0": bool(out["vang_rieng_2005"]["sharpe"] > 0),
    }
    s["DAT_TAT_CA"] = all(s["dat"].values())
    return out


def report_spot(o: dict) -> str:
    def pct(x):
        return "–" if x is None or not np.isfinite(x) else f"{100 * x:+.1f}%"

    def num(x):
        return "–" if x is None or not np.isfinite(x) else f"{x:.2f}"

    lines = ["# Kết quả đặc tả 2: BTC + vàng giao ngay, chỉ long hoặc đứng ngoài", "",
             f"Giai đoạn: {o['giai_doan'][0]} → {o['giai_doan'][1]}. Nguồn vàng: {o['nguon_vang']}.", "",
             "| Chiến lược | CAGR | Biến động | Sharpe | Sụt giảm | Calmar | Năm lãi | Chạm −15% |",
             "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for k in ("S1_giao_ngay", "M4g_luon_long", "M3_giu_50_50", "M1_giu_BTC", "M2_giu_vang", "vang_rieng_2005"):
        s = o[k]
        lines.append(
            f"| {k} | {pct(s['cagr'])} | {pct(s['bien_dong'])} | {num(s['sharpe'])} | {pct(s['sut_giam'])} "
            f"| {num(s['calmar'])} | {pct(s['nam_tron_lai'])} ({s['so_nam_tron']} năm) | {s['cham_15'] or 'không'} |"
        )
    for k in ("S1_giao_ngay", "M4g_luon_long"):
        s = o[k]
        lines += ["", f"## {k}", f"Theo năm: { {y: round(v, 4) for y, v in s['theo_nam'].items()} }",
                  f"Đóng góp: {s['dong_gop']} · Tỷ trọng TB: {s['ty_trong_tb']} · Tổng tỷ trọng max: "
                  f"{s['ty_trong_max_tong']:.2f}",
                  f"Giao dịch: {s['so_giao_dich']} · Chi phí: {s['chi_phi']:.4f}",
                  f"Sharpe trước/sau 07/2022: {num(s['sharpe_truoc'])} / {num(s['sharpe_sau'])}"]  # fmt: skip
        if "dat" in s:
            lines += [f"- {c}: {'ĐẠT' if ok else 'KHÔNG ĐẠT'}" for c, ok in s["dat"].items()]
            lines.append(f"**Tổng: {'ĐẠT' if s['DAT_TAT_CA'] else 'KHÔNG ĐẠT'}**")
    gy = {y: round(v, 4) for y, v in o["vang_rieng_2005"]["theo_nam"].items()}
    lines += ["", f"Vàng riêng 2005: theo năm {gy}"]
    return "\n".join(lines) + "\n"


def evaluate_btc(data: dict, vol_target: float = 0.15) -> dict:
    """Đặc tả 3: chỉ BTC giao ngay, long hoặc đứng ngoài, biến động mục tiêu 15%/năm, tỷ trọng ≤ 1."""
    btc = data["btc"].loc[:END].astype(float)
    cal = pd.date_range(btc.index.min(), END, freq="D")
    zero = {"BTC": pd.Series(0.0, index=cal)}
    tg = {m: target_weights(btc, ANN["BTC"], m, max_w=1.0, vol_target=vol_target) for m in ("long_flat", "always_long")}
    start = tg["long_flat"].first_valid_index() + pd.Timedelta(days=1)
    period = cal[cal >= start]
    out: dict = {"giai_doan": [str(period[0].date()), str(period[-1].date())]}
    for name, mode in (("B1_btc_xu_huong", "long_flat"), ("M4b_btc_luon_long", "always_long")):
        res = simulate({"BTC": btc}, {"BTC": tg[mode].loc[start:]}, zero, period, fee=SPOT_FEE)
        s = stats(res.equity)
        s["so_giao_dich"], s["chi_phi"] = res.trades, res.costs
        s["ty_trong_tb"] = float(res.weights["BTC"].mean())
        s["ty_le_ngay_dung_ngoai"] = float((res.weights["BTC"] < 0.01).mean())
        s["sharpe_truoc"] = _sharpe(res.equity.loc[:SPLIT])
        s["sharpe_sau"] = _sharpe(res.equity.loc[SPLIT:])
        out[name] = s
    hold = btc.reindex(period).ffill()
    out["M1_giu_BTC"] = stats((hold / hold.iloc[0]).dropna())
    s, m4, m1 = out["B1_btc_xu_huong"], out["M4b_btc_luon_long"], out["M1_giu_BTC"]
    s["dat"] = {
        "1_sut_giam_<=40%": bool(-s["sut_giam"] <= 0.40),
        "2_sharpe_>=0.6": bool(s["sharpe"] >= 0.6),
        "3_calmar_>=M4b": bool(s["calmar"] >= m4["calmar"]),
        "4_calmar_>=giu_BTC": bool(s["calmar"] >= m1["calmar"]),
        "5_nam_lai_>=2/3": bool(s["nam_tron_lai"] >= 2 / 3),
        "6_sharpe_truoc_sau_>0": bool(s["sharpe_truoc"] > 0 and s["sharpe_sau"] > 0),
    }
    s["DAT_TAT_CA"] = all(s["dat"].values())
    return out


def report_btc(o: dict) -> str:
    def pct(x):
        return "–" if x is None or not np.isfinite(x) else f"{100 * x:+.1f}%"

    def num(x):
        return "–" if x is None or not np.isfinite(x) else f"{x:.2f}"

    lines = ["# Kết quả đặc tả 3: chỉ BTC giao ngay, biến động mục tiêu 15%/năm", "",
             f"Giai đoạn: {o['giai_doan'][0]} → {o['giai_doan'][1]}.", "",
             "| Chiến lược | CAGR | Biến động | Sharpe | Sụt giảm | Calmar | Năm lãi | Chạm −15% |",
             "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for k in ("B1_btc_xu_huong", "M4b_btc_luon_long", "M1_giu_BTC"):
        s = o[k]
        lines.append(
            f"| {k} | {pct(s['cagr'])} | {pct(s['bien_dong'])} | {num(s['sharpe'])} | {pct(s['sut_giam'])} "
            f"| {num(s['calmar'])} | {pct(s['nam_tron_lai'])} ({s['so_nam_tron']} năm) | {s['cham_15'] or 'không'} |"
        )
    for k in ("B1_btc_xu_huong", "M4b_btc_luon_long"):
        s = o[k]
        lines += ["", f"## {k}", f"Theo năm: { {y: round(v, 4) for y, v in s['theo_nam'].items()} }",
                  f"Tỷ trọng TB: {s['ty_trong_tb']:.3f} · Ngày đứng ngoài: {s['ty_le_ngay_dung_ngoai']:.1%}",
                  f"Giao dịch: {s['so_giao_dich']} · Chi phí: {s['chi_phi']:.4f}",
                  f"Sharpe trước/sau 07/2022: {num(s['sharpe_truoc'])} / {num(s['sharpe_sau'])}"]  # fmt: skip
        if "dat" in s:
            lines += [f"- {c}: {'ĐẠT' if ok else 'KHÔNG ĐẠT'}" for c, ok in s["dat"].items()]
            lines.append(f"**Tổng: {'ĐẠT' if s['DAT_TAT_CA'] else 'KHÔNG ĐẠT'}**")
    return "\n".join(lines) + "\n"


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
    if getattr(args, "dac_ta", 1) == 3:
        o = evaluate_btc(data)
        print(report_btc(o))
        return 0
    if getattr(args, "dac_ta", 1) == 2:
        o = evaluate_spot(data)
        out = Path(args.ket_qua)
        out.mkdir(parents=True, exist_ok=True)
        (out / "giao_ngay.json").write_text(json.dumps(o, ensure_ascii=False, indent=1, default=str), "utf-8")
        md = report_spot(o)
        print(md)
        return 0
    o = evaluate(data)
    out = Path(args.ket_qua)
    out.mkdir(parents=True, exist_ok=True)
    (out / "xu_huong.json").write_text(json.dumps(o, ensure_ascii=False, indent=1, default=str), "utf-8")
    md = report(o)
    (out / "xu_huong.md").write_text(md, "utf-8")
    print(md)
    return 0
