"""Minimal live NG futures dashboard using rich and ib-insync."""

import asyncio
import os
import signal
from dataclasses import dataclass
from datetime import datetime
from typing import Dict

import pandas as pd
from dotenv import load_dotenv
from ib_insync import IB, Contract, util
from rich.live import Live
from rich.table import Table

# ----------------------------------------------------------------------
# Load environment settings
# ----------------------------------------------------------------------
load_dotenv()
TWS_HOST = os.getenv("TWS_HOST", "127.0.0.1")
TWS_PORT = int(os.getenv("TWS_PORT", "7497"))
TWS_CLIENT_ID = int(os.getenv("TWS_CLIENT_ID", "123"))
BAR_SIZE = "5 secs"          # use "1 min" or larger if you prefer
HIST_DURATION = "2 D"         # bootstrap recent bars for indicators


@dataclass
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class IBFeed:
    """Subscribe to real-time bars and maintain a rolling DataFrame."""

    def __init__(self, queue: asyncio.Queue):
        self.q = queue
        self.ib = IB()
        self.df = pd.DataFrame()

    def create_mes_contract(self) -> Contract:
        contract = Contract()
        contract.symbol = "NG"
        contract.secType = "FUT"
        contract.exchange = "NYMEX"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = "20250626"
        contract.localSymbol = "NGN25"
        contract.multiplier = "10000"
        return contract

    async def start(self):
        await self.ib.connectAsync(TWS_HOST, TWS_PORT, clientId=TWS_CLIENT_ID)
        cont = self.create_mes_contract()
        await self.ib.qualifyContractsAsync(cont)

        # recent history for indicator warm-up
        self.df = util.df(
            self.ib.reqHistoricalData(
                cont,
                endDateTime="",
                durationStr=HIST_DURATION,
                barSizeSetting=BAR_SIZE,
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1,
            )
        )
        self.df.set_index("date", inplace=True)

        # subscribe to live bars
        self.ib.reqRealTimeBars(cont, 5, "TRADES", False)
        self.ib.realTimeBarEvent += self._on_bar

        while True:
            await asyncio.sleep(1)

    def _on_bar(self, bar):
        ts = pd.Timestamp(bar.time, unit="s", tz="UTC")
        self.df.loc[ts] = [bar.open, bar.high, bar.low, bar.close, bar.volume]
        self.df = self.df.tail(1000)
        asyncio.create_task(self.q.put(ts))


class Indicators:
    """Compute basic indicators on the latest bars."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def compute(self) -> Dict[str, float]:
        df = self.df.tail(200).copy()
        df["MA_20"] = df["close"].rolling(20).mean()
        df["MA_50"] = df["close"].rolling(50).mean()
        df["rsi_14"] = self._rsi(df["close"], 14)
        df["atr_14"] = self._atr(df, 14)
        df["macd"] = self._macd(df["close"])
        last = df.iloc[-1]
        piv = self._pivots(last.high, last.low, last.close)
        return {
            "price": last.close,
            "ma20": last.MA_20,
            "ma50": last.MA_50,
            "rsi": last.rsi_14,
            "macd": last.macd,
            "atr": last.atr_14,
            "r1": piv[1],
            "s1": piv[4],
        }

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.ewm(com=period - 1, adjust=False).mean()
        ma_down = down.ewm(com=period - 1, adjust=False).mean()
        rs = ma_up / ma_down
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(period).mean()

    @staticmethod
    def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line - signal_line

    @staticmethod
    def _pivots(high: float, low: float, close: float):
        pp = (high + low + close) / 3
        r1 = 2 * pp - low
        r2 = pp + (high - low)
        r3 = high + 2 * (pp - low)
        s1 = 2 * pp - high
        s2 = pp - (high - low)
        s3 = low - 2 * (high - pp)
        return pp, r1, r2, r3, s1, s2, s3


class Dashboard:
    def __init__(self, feed: IBFeed, queue: asyncio.Queue):
        self.feed = feed
        self.q = queue
        self.metrics = {}

    async def run(self):
        async with Live(self._render(), refresh_per_second=4) as live:
            while True:
                await self.q.get()
                self.metrics = Indicators(self.feed.df).compute()
                live.update(self._render(), refresh=True)

    def _render(self) -> Table:
        table = Table(title="NG FUT LIVE DASH", expand=True)
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        if self.metrics:
            table.add_row("UTC", datetime.utcnow().strftime("%H:%M:%S"))
            table.add_row("Last Px", f"{self.metrics['price']:.3f}")
            table.add_row("MA 20", f"{self.metrics['ma20']:.3f}")
            table.add_row("MA 50", f"{self.metrics['ma50']:.3f}")
            table.add_row("RSI 14", f"{self.metrics['rsi']:.1f}")
            table.add_row("MACD", f"{self.metrics['macd']:.3f}")
            table.add_row("ATR 14", f"{self.metrics['atr']:.3f}")
            table.add_row("Pivot R1", f"{self.metrics['r1']:.3f}")
            table.add_row("Pivot S1", f"{self.metrics['s1']:.3f}")
        return table


async def shutdown(feed: IBFeed):
    print("Disconnecting …")
    feed.ib.disconnect()
    await asyncio.sleep(0.2)
    raise SystemExit(0)


async def main():
    q = asyncio.Queue()
    feed = IBFeed(q)
    dash = Dashboard(feed, q)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(feed)))

    await asyncio.gather(feed.start(), dash.run())


if __name__ == "__main__":
    asyncio.run(main())
