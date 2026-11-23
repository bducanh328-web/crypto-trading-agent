import pandas as pd

from crypto_agent import cli
from crypto_agent.context import GREEN

from conftest import make_daily


class FakeBitunix:
    """Giả lập API Bitunix: 20 coin tăng giá, BTC tăng giá."""

    symbols = ["BTCUSDT"] + [f"C{i}USDT" for i in range(19)] + ["USDCUSDT"]

    def trading_pairs(self):
        now_ms = pd.Timestamp.now(tz="UTC").value // 10**6
        return pd.DataFrame(
            [
                {"symbol": s, "base": s.removesuffix("USDT"), "quote": "USDT", "symbolStatus": "OPEN",
                 "isApiSupported": True, "launchTime": now_ms - 900 * 86_400_000, "delistTime": 0}
                for s in self.symbols
            ]
        )  # fmt: skip

    def tickers(self):
        return pd.DataFrame(
            {"symbol": self.symbols, "lastPrice": 10.0, "quoteVol": range(100, 121), "baseVol": range(10, 31)}
        )

    def klines(self, symbol, interval, bars):
        end = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)
        start = (end - pd.Timedelta(days=bars - 1)).strftime("%Y-%m-%d")
        seed = self.symbols.index(symbol)
        price = 50_000 if symbol == "BTCUSDT" else 10.0 * (seed + 1)
        return make_daily(bars, 0.004, seed=seed, start=start, price=price)

    def funding_history(self, symbol, start_ms):
        idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=9, freq="8h")
        return pd.Series(0.0001, index=idx)


def test_cmd_den_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "BitunixPublic", FakeBitunix)
    summary = tmp_path / "summary.md"
    code = cli.main(["den", "--data-dir", str(tmp_path), "--summary", str(summary)])
    out = capsys.readouterr().out
    assert code == 0 and GREEN in out and "Độ rộng rổ 20 coin" in out  # USDC bị loại
    den = pd.read_csv(tmp_path / "den.csv")
    assert den["light"].iloc[-1] == GREEN
    ro = pd.read_csv(tmp_path / "ro_coin.csv")
    assert len(ro) == 20 and "USDCUSDT" not in set(ro["ma"])
    assert (tmp_path / "bitunix" / "klines" / "1d" / "BTCUSDT.csv").exists()
    assert summary.read_text(encoding="utf-8").startswith("🟢")
