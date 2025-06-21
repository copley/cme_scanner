"""Real-Time NG Trend Confidence Dashboard using ib_insync and rich."""

import asyncio
import os
import signal
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd
from dotenv import load_dotenv
from ib_insync import IB, Contract, Ticker
from rich.live import Live
from rich.table import Table

load_dotenv()

TWS_HOST = os.getenv("TWS_HOST", "127.0.0.1")
TWS_PORT = int(os.getenv("TWS_PORT", "7497"))
TWS_CLIENT_ID = int(os.getenv("TWS_CLIENT_ID", "124"))
DEPTH_ROWS = 5
REFRESH_SECS = 10
DB_FILE = "ng_trend.db"

@dataclass
class DepthLevel:
    price: float
    size: float
    mmid: str = ""

@dataclass
class OrderBook:
    bids: List[DepthLevel] = field(default_factory=list)
    asks: List[DepthLevel] = field(default_factory=list)

class L2Feed:
    def __init__(self, queue: asyncio.Queue):
        self.q = queue
        self.ib = IB()
        self.book = OrderBook()
        self.ticker: Optional[Ticker] = None

    def create_contract(self) -> Contract:
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
        contract = self.create_contract()
        await self.ib.qualifyContractsAsync(contract)
        self.ticker = self.ib.reqMktDepth(contract, numRows=DEPTH_ROWS)
        self.ib.updateMktDepthEvent += self._on_depth
        while True:
            await asyncio.sleep(1)

    def _on_depth(self, ticker: Ticker, row: int, operation: int, side: int, price: float, size: float):
        levels = self.book.bids if side == 0 else self.book.asks
        while len(levels) <= row:
            levels.append(DepthLevel(0.0, 0.0))
        levels[row].price = price
        levels[row].size = size
        asyncio.create_task(self.q.put(datetime.utcnow()))

class FeatureEngine:
    def __init__(self, book: OrderBook):
        self.book = book
        self.mid_prices: List[float] = []

    def compute(self) -> Dict[str, float]:
        mid = 0.0
        if self.book.bids and self.book.asks:
            mid = (self.book.bids[0].price + self.book.asks[0].price) / 2
            self.mid_prices.append(mid)
            self.mid_prices = self.mid_prices[-120:]
        vwap = sum(self.mid_prices) / len(self.mid_prices) if self.mid_prices else 0.0
        imbalance = sum(b.size for b in self.book.bids) - sum(a.size for a in self.book.asks)
        wall = max([lvl.size for lvl in (self.book.bids + self.book.asks)] or [0])
        delta = mid - vwap if self.mid_prices else 0.0
        return {"mid": mid, "vwap": vwap, "imbalance": imbalance, "wall": wall, "delta": delta}

class ConfidenceEngine:
    def __init__(self):
        self.score = 50

    def update(self, feats: Dict[str, float]) -> int:
        if feats["imbalance"] > 0 and feats["delta"] > 0:
            self.score = min(100, self.score + 5)
        elif feats["imbalance"] < 0 and feats["delta"] < 0:
            self.score = max(0, self.score - 5)
        else:
            self.score = max(0, self.score - 1)
        return self.score

class PhaseDetector:
    def __init__(self):
        self.prev_mid: Optional[float] = None
        self.trend_dir = 0

    def classify(self, mid: float, conf: int) -> str:
        if self.prev_mid is None:
            self.prev_mid = mid
            return "Initializing"
        if mid > self.prev_mid:
            new_dir = 1
        elif mid < self.prev_mid:
            new_dir = -1
        else:
            new_dir = self.trend_dir
        phase = "Choppy"
        if conf > 60 and new_dir == 1:
            phase = "Trending Up"
        elif conf > 60 and new_dir == -1:
            phase = "Trending Down"
        elif conf < 40:
            phase = "Reversing"
        self.trend_dir = new_dir
        self.prev_mid = mid
        return phase

class TraderAssist:
    def suggest(self, conf: int, phase: str) -> str:
        if conf < 40:
            return "Consider exit / reversal"
        if phase.startswith("Trending"):
            return "Hold bias with trend"
        return "Wait / watch"

class SQLiteLogger:
    def __init__(self, filename: str):
        self.conn = sqlite3.connect(filename)
        self._setup()

    def _setup(self):
        cur = self.conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS depth(
                ts TEXT, bid1 REAL, bid1sz REAL, ask1 REAL, ask1sz REAL,
                bid2 REAL, bid2sz REAL, ask2 REAL, ask2sz REAL,
                bid3 REAL, bid3sz REAL, ask3 REAL, ask3sz REAL,
                bid4 REAL, bid4sz REAL, ask4 REAL, ask4sz REAL,
                bid5 REAL, bid5sz REAL, ask5 REAL, ask5sz REAL
            )"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS analysis(
                ts TEXT, mid REAL, vwap REAL, imbalance REAL,
                wall REAL, delta REAL, confidence INTEGER, phase TEXT, advice TEXT
            )"""
        )
        self.conn.commit()

    def log_depth(self, ts: datetime, book: OrderBook):
        values = [ts.isoformat()]
        for i in range(DEPTH_ROWS):
            if i < len(book.bids):
                values += [book.bids[i].price, book.bids[i].size]
            else:
                values += [0.0, 0.0]
            if i < len(book.asks):
                values += [book.asks[i].price, book.asks[i].size]
            else:
                values += [0.0, 0.0]
        placeholders = ",".join(["?"] * (1 + DEPTH_ROWS * 4))
        self.conn.execute(f"INSERT INTO depth VALUES({placeholders})", values)
        self.conn.commit()

    def log_analysis(self, ts: datetime, feats: Dict[str, float], conf: int, phase: str, advice: str):
        self.conn.execute(
            "INSERT INTO analysis VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ts.isoformat(),
                feats["mid"],
                feats["vwap"],
                feats["imbalance"],
                feats["wall"],
                feats["delta"],
                conf,
                phase,
                advice,
            ),
        )
        self.conn.commit()

class Dashboard:
    def __init__(self, feed: L2Feed, queue: asyncio.Queue, db: SQLiteLogger):
        self.feed = feed
        self.q = queue
        self.db = db
        self.fe = FeatureEngine(feed.book)
        self.conf_engine = ConfidenceEngine()
        self.phase_det = PhaseDetector()
        self.assist = TraderAssist()
        self.metrics: Dict[str, float] = {}
        self.last_update = datetime.utcnow()

    async def run(self):
        async with Live(self._render(), refresh_per_second=1) as live:
            while True:
                await self.q.get()
                now = datetime.utcnow()
                if (now - self.last_update).total_seconds() >= REFRESH_SECS:
                    feats = self.fe.compute()
                    conf = self.conf_engine.update(feats)
                    phase = self.phase_det.classify(feats["mid"], conf)
                    advice = self.assist.suggest(conf, phase)
                    self.db.log_depth(now, self.feed.book)
                    self.db.log_analysis(now, feats, conf, phase, advice)
                    self.metrics = {**feats, "conf": conf, "phase": phase, "advice": advice}
                    live.update(self._render(), refresh=True)
                    self.last_update = now

    def _render(self) -> Table:
        table = Table(title="NG Trend Confidence", expand=True)
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        if self.metrics:
            m = self.metrics
            table.add_row("UTC", datetime.utcnow().strftime("%H:%M:%S"))
            table.add_row("Mid", f"{m['mid']:.3f}")
            table.add_row("VWAP", f"{m['vwap']:.3f}")
            table.add_row("Imbalance", f"{m['imbalance']:.0f}")
            table.add_row("Wall", f"{m['wall']:.0f}")
            table.add_row("Delta", f"{m['delta']:.3f}")
            table.add_row("Confidence", str(m['conf']))
            table.add_row("Phase", m['phase'])
            table.add_row("Advice", m['advice'])
        return table

async def shutdown(feed: L2Feed):
    feed.ib.disconnect()
    await asyncio.sleep(0.2)
    raise SystemExit(0)

async def main():
    q = asyncio.Queue()
    feed = L2Feed(q)
    db = SQLiteLogger(DB_FILE)
    dash = Dashboard(feed, q, db)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(feed)))

    await asyncio.gather(feed.start(), dash.run())

if __name__ == "__main__":
    asyncio.run(main())
