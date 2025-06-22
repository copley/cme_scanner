# Real-Time NG Edge Dashboard: Unified Implementation
# Combines: Level 2 data ingestion, feature extraction, geometry (Fibonacci), rhythm (vibration/frequency), confidence scoring, trend phase, edge scoring, trend age, exhaustion alerts.

import asyncio
import os
import math
from collections import deque, Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, List, Literal, Optional

from dotenv import load_dotenv
from ib_insync import IB, Contract
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

load_dotenv()

# --------------------------- Settings --------------------------- #
TWS_HOST = os.getenv("TWS_HOST", "127.0.0.1")
TWS_PORT = int(os.getenv("TWS_PORT", "7497"))
TWS_CLIENT_ID = int(os.getenv("TWS_CLIENT_ID", "101"))
DEPTH_LEVELS = 5
ROLL_WINDOW = 120

Side = Literal["bid", "ask"]

# ------------------------- Data Classes ------------------------- #
@dataclass
class L2Level:
    side: Side
    position: int
    price: float
    size: float
    mm_id: str
    update_type: int

@dataclass
class L2Snapshot:
    bids: List[L2Level]
    asks: List[L2Level]
    last_price: float
    ts_exchange: datetime
    ts_local: datetime

@dataclass
class FeatureVector:
    ts: datetime
    mid: float
    vwap: float
    imbalance: float
    wall_size: float
    delta: float
    fib_prox: float
    energy: float
    vibration: float
    resonance: float
    trend_age: int
    exhaustion: bool
    edge_score: float

# --------------------------- L2 Ingestor --------------------------- #
class L2Ingestor:
    def __init__(self, queue: asyncio.Queue):
        self.ib = IB()
        self.queue = queue
        self._snapshot: dict[str, list[L2Level]] = {"bid": [], "ask": []}
        self._last_price: float = 0.0

    async def start(self):
        contract = Contract(symbol="NG", secType="FUT", exchange="NYMEX", currency="USD")
        await self.ib.connectAsync(TWS_HOST, TWS_PORT, clientId=TWS_CLIENT_ID)
        await self.ib.qualifyContractsAsync(contract)
        self.ib.reqMktDepth(contract, numRows=DEPTH_LEVELS, isSmartDepth=False)
        self.ib.pendingTickersEvent += self._on_tick
        self.ib.updateMktDepthEvent += self._on_depth

    def _on_tick(self, tickers):
        for t in tickers:
            if t.last:
                self._last_price = t.last

    def _on_depth(self, req_id, pos, op, side, price, size, mm_id):
        s = "bid" if side == 1 else "ask"
        lvl = L2Level(s, pos, price, size, mm_id, op)
        book_side = self._snapshot[s]
        while len(book_side) <= pos:
            book_side.append(lvl)
        book_side[pos] = lvl
        if len(self._snapshot["bid"]) and len(self._snapshot["ask"]):
            snap = L2Snapshot(
                bids=self._snapshot["bid"][:DEPTH_LEVELS],
                asks=self._snapshot["ask"][:DEPTH_LEVELS],
                last_price=self._last_price,
                ts_exchange=datetime.now(timezone.utc),
                ts_local=datetime.utcnow().replace(tzinfo=timezone.utc)
            )
            asyncio.create_task(self.queue.put(snap))

# ----------------------- Feature Extraction ------------------------ #
class FeatureEngine:
    def __init__(self, in_q: asyncio.Queue, out_q: asyncio.Queue):
        self.in_q, self.out_q = in_q, out_q
        self.snapshots: Deque[L2Snapshot] = deque(maxlen=ROLL_WINDOW)
        self.fib_levels = []
        self.phase = "Initializing"
        self.trend_start_ts: Optional[datetime] = None

    async def run(self):
        while True:
            snap = await self.in_q.get()
            self.snapshots.append(snap)
            if len(self.snapshots) < 10:
                continue
            fv = self._compute(snap)
            await self.out_q.put(fv)

    def _compute(self, snap: L2Snapshot) -> FeatureVector:
        mid = (snap.bids[0].price + snap.asks[0].price) / 2
        vwap, sizes = 0.0, [sum(l.size for l in s.bids + s.asks) for s in self.snapshots]
        prices = [s.last_price for s in self.snapshots]
        vwap = sum(p * q for p, q in zip(prices, sizes)) / sum(sizes)

        imbalance = sum(l.size for l in snap.bids) - sum(l.size for l in snap.asks)
        wall = max([l.size for l in snap.bids + snap.asks], default=0)
        delta = mid - vwap

        hi, lo = max(prices), min(prices)
        self.fib_levels = [hi - r * (hi - lo) for r in (0.382, 0.5, 0.618)]
        fib_prox = min(abs(mid - f) for f in self.fib_levels)

        energy = mid - (self.snapshots[-10].last_price)
        diffs = [prices[i+1] - prices[i] for i in range(len(prices)-1)]
        vibration = float(pd.Series(diffs).std())
        bin_counts = Counter(round(p, 2) for p in prices)
        resonance = max(bin_counts.values()) / len(prices)

        now = snap.ts_local
        if self.phase in ["TrendingUp", "TrendingDown"]:
            if not self.trend_start_ts:
                self.trend_start_ts = now
            trend_age = int((now - self.trend_start_ts).total_seconds())
        else:
            self.trend_start_ts = None
            trend_age = 0

        exhaustion = energy < 0.05 and vibration < 0.03 and fib_prox < 0.02
        edge_score = 0.3 * min(100, max(0, imbalance)) + 0.2 * energy + 0.2 * (1 - fib_prox) + 0.3 * vibration

        return FeatureVector(
            ts=now, mid=mid, vwap=vwap, imbalance=imbalance, wall_size=wall,
            delta=delta, fib_prox=fib_prox, energy=energy, vibration=vibration,
            resonance=resonance, trend_age=trend_age, exhaustion=exhaustion,
            edge_score=round(edge_score, 2)
        )

# --------------------------- Dashboard --------------------------- #
class Dashboard:
    def __init__(self, q: asyncio.Queue):
        self.q = q
        self.last: Optional[FeatureVector] = None

    async def run(self):
        async with Live(self._render(), refresh_per_second=1) as live:
            while True:
                while not self.q.empty():
                    self.last = await self.q.get()
                    live.update(self._render(), refresh=True)
                await asyncio.sleep(1)

    def _render(self) -> Panel:
        table = Table(title="NG Edge Forecast", expand=True)
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        if self.last:
            f = self.last
            table.add_row("Time", f.ts.strftime("%H:%M:%S"))
            table.add_row("Mid", f"{f.mid:.3f}")
            table.add_row("VWAP", f"{f.vwap:.3f}")
            table.add_row("Imbalance", f"{f.imbalance:.1f}")
            table.add_row("Wall Size", f"{f.wall_size:.1f}")
            table.add_row("Delta", f"{f.delta:.3f}")
            table.add_row("Fib Prox", f"{f.fib_prox:.4f}")
            table.add_row("Energy", f"{f.energy:+.4f}")
            table.add_row("Vibration", f"{f.vibration:.4f}")
            table.add_row("Resonance", f"{f.resonance:.2f}")
            table.add_row("Trend Age (s)", str(f.trend_age))
            table.add_row("Exhaustion", "Yes" if f.exhaustion else "No")
            table.add_row("Edge Score", f"{f.edge_score:.2f}")
        return Panel(table, border_style="cyan")

# --------------------------- Main --------------------------- #
async def main():
    q_raw = asyncio.Queue()
    q_feat = asyncio.Queue()

    ingestor = L2Ingestor(queue=q_raw)
    feats = FeatureEngine(q_raw, q_feat)
    dash = Dashboard(q_feat)

    await ingestor.start()
    await asyncio.gather(feats.run(), dash.run())

if __name__ == "__main__":
    import pandas as pd
    asyncio.run(main())
