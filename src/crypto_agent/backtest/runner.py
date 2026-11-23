"""cta kiem-dinh: tải lịch sử, chạy bản cơ sở + biến thể + độ nhạy, ghi báo cáo."""

from __future__ import annotations

import json
import logging
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import BTParams, daily_light, prepare, run, variant
from .history import History, all_usdt_symbols, monthly_universe
from .metrics import check, summarize

log = logging.getLogger(__name__)

DATA_START, DATA_END = date(2019, 9, 1), date(2026, 10, 31)
PERIODS = {
    "trong_mau": (pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2023-12-31 20:00", tz="UTC")),
    "ngoai_mau": (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-10-31 20:00", tz="UTC")),
}
BASE = BTParams()


def variants() -> dict[str, BTParams]:
    b = BASE
    v = {
        "co_so": b,
        "A_khong_boi_canh": variant(b, context="none"),
        "B_den_chi_btc": variant(b, context="btc_only"),
        "C_vang_nua_khoi_luong": variant(b, yellow_trade=True),
        "D_tat_loc_funding": variant(b, funding_filter=False),
        "E_swing_2": variant(b, swing_n=2),
        "E_swing_5": variant(b, swing_n=5),
        "F_cho_hoi_0_7": variant(b, entry="pullback"),
        "G_chi_btc_eth": variant(b, majors_only=True),
        "H_bo_tin_hieu_dem": variant(b, skip_hours_utc=(20,)),
        # Chẩn đoán (không phải ứng viên): bỏ hết phí, trượt giá để xem lợi thế trước chi phí
        "Z_chan_doan_khong_phi": variant(b, taker=0.0, maker=0.0, slip_major=0.0, slip_alt=0.0),
    }
    sens = {
        "do_rong": [("green_breadth", 0.44, 0.66), ("red_breadth", 0.32, 0.48)],
        "khoi_luong": [("vol_mult", 1.2, 1.8)],
        "than_nen": [("body_min", 0.48, 0.72)],
        "khong_duoi": [("max_chase_atr", 1.2, 1.8)],
        "stop_toi_da": [("max_stop_atr", 2.4, 3.6)],
        "chot_R": [("partial_r", 1.6, 2.4)],
        "het_gio": [("time_stop_bars", 24, 36)],
    }
    for name, fields in sens.items():
        v[f"S_{name}_-20%"] = variant(b, **{f: lo for f, lo, _ in fields})
        v[f"S_{name}_+20%"] = variant(b, **{f: hi for f, _, hi in fields})
    return v


def download(root: Path, workers: int) -> pd.DataFrame:
    hist = History(root, workers=workers)
    t0 = time.time()
    syms = all_usdt_symbols()
    log.info("Có %d hợp đồng USDT-M (kể cả đã hủy niêm yết)", len(syms))
    hist.download("1d", syms, DATA_START, DATA_END)
    daily = {s: hist.load("1d", s) for s in syms}
    daily = {s: d for s, d in daily.items() if d is not None and not d.empty}
    uni = monthly_universe(daily, date(2020, 1, 1), DATA_END)
    uni.to_csv(root / "ro_thang.csv", index=False)
    need = sorted(set(uni["ma"]) | {"BTCUSDT", "ETHUSDT"})
    log.info("Rổ theo tháng: %d mã từng vào rổ; tải 4H + funding", len(need))
    hist.download("4h", need, DATA_START, DATA_END)
    hist.download("funding", need, DATA_START, DATA_END)
    log.info("Tải xong sau %.0f giây", time.time() - t0)
    return uni


def load(root: Path, uni: pd.DataFrame):
    hist = History(root)
    need = sorted(set(uni["ma"]) | {"BTCUSDT", "ETHUSDT"})
    h4, d1, fund = {}, {}, {}
    for s in need:
        a, b = hist.load("4h", s), hist.load("1d", s)
        if a is None or a.empty or b is None or b.empty:
            log.warning("%s: thiếu dữ liệu, bỏ", s)
            continue
        h4[s], d1[s] = a, b
        f = hist.load("funding", s)
        fund[s] = f if f is not None and len(f) else None
    return h4, d1, fund


def evaluate(root: Path, out: Path, only: list[str] | None = None) -> dict:
    uni = pd.read_csv(root / "ro_thang.csv")
    h4, d1, fund = load(root, uni)
    btc4 = h4["BTCUSDT"]["close"]
    prepared: dict[tuple, dict] = {}
    lights: dict[tuple, pd.Series] = {}
    results = {}
    out.mkdir(parents=True, exist_ok=True)
    todo = variants()
    if only:
        todo = {k: v for k, v in todo.items() if k in only}
    for name, p in todo.items():
        t0 = time.time()
        key = (p.swing_n, p.vol_mult, p.trend_ema, p.rs_bars)
        if key not in prepared:
            prepared[key] = {s: prepare(s, h4[s], d1[s], btc4, fund.get(s), p) for s in h4}
        lkey = (p.context, p.swing_n, p.green_breadth, p.red_breadth, p.confirm_days, p.btc_ma, p.breadth_ma)
        if lkey not in lights:
            lights[lkey] = daily_light(d1["BTCUSDT"], d1, uni, p)
        results[name] = {}
        for period, (start, end) in PERIODS.items():
            res = run(prepared[key], lights[lkey], uni, p, start, end, fund)
            s = summarize(res.trades, res.equity, start, end, d1["BTCUSDT"]["close"])
            s["dat"] = check(s)
            s["su_kien"] = res.events
            results[name][period] = s
            if name == "co_so":
                res.trades.to_csv(out / f"giao_dich_co_so_{period}.csv", index=False)
                res.equity.to_csv(out / f"von_co_so_{period}.csv", index_label="time")
        o = results[name]["ngoai_mau"]
        log.info("%s: ngoài mẫu %d lệnh, %.2fR, PF %.2f, sụt giảm %.1f%% (%.0f giây)", name, o["so_lenh"],
                 o["ky_vong_r"], o["pf"], 100 * o["sut_giam"], time.time() - t0)  # fmt: skip
    lights_base = lights.get((BASE.context, BASE.swing_n, BASE.green_breadth, BASE.red_breadth, BASE.confirm_days,
                              BASE.btc_ma, BASE.breadth_ma))  # fmt: skip
    if lights_base is not None:
        lights_base.rename("den").to_csv(out / "den_lich_su.csv", index_label="ngay_dong")
    (out / "tong_hop.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=_json), "utf-8")
    md = report(results, uni)
    (out / "tong_hop.md").write_text(md, "utf-8")
    return results


def _json(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    return str(x)


def _f(x, kind="r"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    if kind == "r":
        return f"{x:+.2f}R"
    if kind == "pct":
        return f"{100 * x:+.1f}%"
    if kind == "x":
        return f"{x:.2f}"
    return str(x)


def report(results: dict, uni: pd.DataFrame) -> str:
    lines = ["# Kết quả kiểm định: Lớp bối cảnh + BOS hai chiều", ""]
    lines += [
        "Trong mẫu 2020–2023, ngoài mẫu 01/2024–10/2026. Vốn mỗi kỳ = 1. Phí Bitunix, trượt giá, funding.",
        f"Rổ theo tháng: {uni['ma'].nunique()} mã từng vào rổ (kể cả mã đã hủy niêm yết).",
        "",
    ]
    head = (
        "| Biến thể | Kỳ | Lệnh | Thắng | Kỳ vọng | PF | Lợi nhuận | CAGR | Sụt giảm | Năm lãi | BTC giữ "
        "| Long | Short |"
    )
    lines += [head, "|" + "---|" * 13]
    for name, per in results.items():
        for period, s in per.items():
            lines.append(
                f"| {name} | {period} | {s['so_lenh']} | {_f(s['ty_le_thang'], 'pct')} | {_f(s['ky_vong_r'])} "
                f"| {_f(s['pf'], 'x')} | {_f(s['loi_nhuan'], 'pct')} | {_f(s['cagr'], 'pct')} "
                f"| {_f(s['sut_giam'], 'pct')} | {_f(s['nam_lai'], 'pct')} | {_f(s['btc_giu'], 'pct')} "
                f"| {s.get('long_n', 0)} / {_f(s.get('long_r'))} | {s.get('short_n', 0)} / {_f(s.get('short_r'))} |"
            )
    lines += ["", "## Bản cơ sở: đối chiếu tiêu chí (ngoài mẫu)", ""]
    o = results.get("co_so", {}).get("ngoai_mau")
    if o:
        for k, ok in o["dat"].items():
            lines.append(f"- {k}: {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
        sens = [n for n in results if n.startswith("S_")]
        robust = all(results[n]["ngoai_mau"]["ky_vong_r"] > 0 for n in sens) if sens else False
        lines.append(f"- độ bền ±20%: {'ĐẠT' if robust else 'KHÔNG ĐẠT'}")
        lines += [
            "",
            f"Theo năm: {o['theo_nam']}",
            f"Lý do thoát: {o['ly_do']}",
            f"Chuỗi thua dài nhất: {o['chuoi_thua']}",
        ]
        lines += [f"Sự kiện: {e}" for e in o["su_kien"]]
        i = results["co_so"]["trong_mau"]
        lines += ["", f"Trong mẫu theo năm: {i['theo_nam']}", f"Trong mẫu lý do thoát: {i['ly_do']}"]
        lines += [f"Trong mẫu R theo lý do: {i['r_theo_ly_do']}"]
        lines += [f"Trong mẫu sự kiện: {e}" for e in i["su_kien"]]
    return "\n".join(lines) + "\n"


def main_args(args) -> int:
    root, out = Path(args.data_dir) / "hist", Path(args.ket_qua)
    if args.buoc in ("tai", "tat_ca"):
        download(root, args.luong)
    if args.buoc in ("chay", "tat_ca"):
        evaluate(root, out, args.chi or None)
    return 0
