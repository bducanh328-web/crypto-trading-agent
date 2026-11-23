"""Thông số khởi điểm theo đặc tả (CHƯA KIỂM ĐỊNH). Đổi thông số = phải kiểm định lại."""

from __future__ import annotations

from dataclasses import dataclass, field

STABLECOINS = frozenset(
    {"USDT", "USDC", "BUSD", "TUSD", "FDUSD", "DAI", "USDP", "USDD", "PYUSD", "USDE", "EUR", "EURI", "AEUR", "XUSD"}
)


@dataclass(frozen=True)
class Params:
    # Rổ coin
    universe_size: int = 30
    min_listed_days: int = 180
    turnover_days: int = 30

    # Lớp bối cảnh (khung 1D)
    btc_symbol: str = "BTCUSDT"
    btc_ma: int = 200
    breadth_ma: int = 50
    green_breadth: float = 0.55  # XANH: độ rộng ≥ 55%
    red_breadth: float = 0.40  # ĐỎ: độ rộng ≤ 40%
    confirm_days: int = 2  # đèn chỉ đổi khi điều kiện mới giữ 2 ngày liên tiếp

    # Swing / BOS
    swing_n: int = 3  # đỉnh/đáy so với 3 nến mỗi bên

    # Bộ lọc funding (quy về mỗi 8 giờ)
    funding_window_hours: int = 72
    funding_hot: float = 0.0005  # không long khi > 0,05% / 8h
    funding_cold: float = -0.0003  # không short khi < −0,03% / 8h

    excluded_bases: frozenset[str] = field(default_factory=lambda: STABLECOINS)


PARAMS = Params()
