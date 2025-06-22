#!/usr/bin/env python3
"""
Real‑Time NG Trend Confidence Dashboard – Super Script (no DB)
Dependencies: ib‑insync, rich, pandas, numpy, python‑dotenv
"""

import asyncio
import math
import os
import signal
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Deque, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from ib_insync import IB, Contract, TickByTickData, MktDepthData
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

load_dotenv()

TWS_HOST        = os.getenv("TWS_HOST", "127.0.0.1")
TWS_PORT        = int(os.getenv("TWS_PORT", 7497))
TWS_CLIENT_ID   = int(os.getenv("TWS_CLIENT_ID", 124))

DEPTH_LEVELS    = int(os.getenv("DEPTH_LEVELS", 5))
ROLL_WINDOW_S   = int(os.getenv("ROLL_WINDOW_S", 120))
CONF_TAU_S      = int(os.getenv("CONF_TAU_S", 30))
REFRESH_SECS    = int(os.getenv("REFRESH_SECS", 10))

# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #

Side = Literal["bid", "ask"]

@dataclass
class BookLevel:
    price: float
    size: float
    mmid: str = ""

@dataclass
class Snapshot:
    ts_local:   datetime
    bids:       List[BookLevel]
    asks:       List[BookLevel]
    trades:     List[Tuple[float, float, str]]    # (price, size, side "B"/"S")
    last_price: float

# --------------------------------------------------------------------------- #
# Ingestors
# --------------------------------------------------------------------------- #

class Ingestor:
    """
    Collects Level‑2 market depth AND tick‑by‑tick prints for the NG front‑month.
    Emits merged Snapshot objects to an asyncio.Queue.
    """

    def __init__(self, out_q: asyncio.Queue):
        self.ib = IB()
        self.out_q = out_q
        self.depth_state: Dict[Side, Dict[int, BookLevel]] = {"bid": {}, "ask": {}}
        self.pending_trades: List[Tuple[float, float, str]] = []
        self.last_price: float = 0.0
        self._lock = asyncio.Lock()

    async def start(self):
        await self.ib.connectAsync(TWS_HOST, TWS_PORT, clientId=TWS_CLIENT_ID)

        # Discover front‑month – here hard‑coding "NG" June‑25 for demo
        contract = Contract(symbol="NG", secType="FUT", exchange="NYMEX",
                             currency="USD", lastTradeDateOrContractMonth="20250626")

        await self.ib.qualifyContractsAsync(contract)

        # Subscriptions
        self.ib.updateMktDepthEvent += self._on_depth
        self.ib.pendingTickersEvent += self._on_price      # for last_price
        self.ib.updateTickByTickEvent += self._on_print

        self.ib.reqMktDepth(contract, numRows=DEPTH_LEVELS, isSmartDepth=False)
        self.ib.reqTickByTickData(contract, "Last", numberOfTicks=0, ignoreSize=False)

        # Periodic snapshot assembler
        asyncio.create_task(self._snapshot_loop())

    # ---------------- event handlers ---------------- #

    def _on_depth(self, depth: MktDepthData, **__):
        """
        depth.side: 0 = ask, 1 = bid   (IB quirk)
        depth.position = level
        """
        side: Side = "ask" if depth.side == 0 else "bid"
        lvl = BookLevel(price=depth.price, size=depth.size, mmid=depth.mmid or "")
        self.depth_state[side][depth.position] = lvl

    def _on_price(self, tickers):
        for t in tickers:
            if t.last is not None:
                self.last_price = t.last

    def _on_print(self, tick: TickByTickData):
        # tick.tickAttribLast.pastLimit etc. available if needed
        side = "B" if tick.attributes.bidPastLow else "S"
        self.pending_trades.append((tick.price, tick.size, side))

    # ---------------- snapshot assembler ---------------- #

    async def _snapshot_loop(self):
        while True:
            await asyncio.sleep(0.1)  # 100ms cadence
            async with self._lock:
                bids = [self.depth_state["bid"].get(i, BookLevel(0, 0)) for i in range(DEPTH_LEVELS)]
                asks = [self.depth_state["ask"].get(i, BookLevel(0, 0)) for i in range(DEPTH_LEVELS)]
                trades = self.pending_trades
                self.pending_trades = []

            snap = Snapshot(
                ts_local=datetime.utcnow().replace(tzinfo=timezone.utc),
                bids=bids,
                asks=asks,
                trades=trades,
                last_price=self.last_price,
            )
            await self.out_q.put(snap)

# --------------------------------------------------------------------------- #
# Feature Engineering
# --------------------------------------------------------------------------- #

@dataclass
class FeatureVector:
    ts: datetime
    book_imb: float
    slope: float
    wall_px: Optional[float]
    wall_size: Optional[float]
    cdv: float
    vwap_delta: float
    tsi: float
    vol_sigma: float

class FeatureEngine:
    def __init__(self, in_q: asyncio.Queue, out_q: asyncio.Queue):
        self.in_q, self.out_q = in_q, out_q
        self.buf: Deque[Snapshot] = deque(maxlen=ROLL_WINDOW_S * 10)  # 100 ms * window
        self.cdv_total: float = 0.0                                   # cumulative delta

        # Rolling mid‑price & realised σ
        self.mid_prices: Deque[float] = deque(maxlen=ROLL_WINDOW_S * 10)

    async def run(self):
        while True:
            snap: Snapshot = await self.in_q.get()
            self.buf.append(snap)
            self.mid_prices.append(snap.last_price or self._mid_of(snap))

            # Update cumulative delta with latest prints
            for _, sz, side in snap.trades:
                self.cdv_total += sz if side == "B" else -sz

            if len(self.buf) < 20:
                continue
            fv = self._make_feat()
            await self.out_q.put(fv)

    # ---------------- helpers ---------------- #

    @staticmethod
    def _mid_of(s: Snapshot) -> float:
        if s.bids[0].price and s.asks[0].price:
            return (s.bids[0].price + s.asks[0].price) / 2
        return s.last_price

    def _make_feat(self) -> FeatureVector:
        latest = self.buf[-1]
        bid_qty = sum(l.size for l in latest.bids)
        ask_qty = sum(l.size for l in latest.asks)
        book_imb = bid_qty - ask_qty

        # Order‑book slope  Σ(size_i / distance_i)
        slope = 0.0
        mid = self._mid_of(latest)
        for lvl in latest.bids:
            dist = max(0.01, mid - lvl.price)
            slope += lvl.size / dist
        for lvl in latest.asks:
            dist = max(0.01, lvl.price - mid)
            slope -= lvl.size / dist    # negative contribution

        # Wall detection
        wall_px, wall_sz = None, None
        wall_threshold = max(bid_qty, ask_qty) * 0.5
        for lvl in latest.bids + latest.asks:
            if lvl.size > wall_threshold:
                wall_px, wall_sz = lvl.price, lvl.size
                break

        # VWAP over buffer
        prices = np.array([self._mid_of(s) for s in self.buf])
        sizes  = np.array([sum(l.size for l in s.bids + s.asks) for s in self.buf])
        vwap   = float(np.average(prices, weights=sizes))
        vwap_delta = latest.last_price - vwap

        # TSI (ADX‑style) on mid‑prices (simple 14‑period)
        if len(self.mid_prices) >= 16:
            plus_dm  = np.diff(np.maximum(np.diff(self.mid_prices, prepend=self.mid_prices[0]), 0))
            minus_dm = np.diff(np.maximum(np.diff(self.mid_prices, prepend=self.mid_prices[0]) * -1, 0))
            tr  = np.diff(np.maximum(self.mid_prices, 0))  # crude proxy – NG ticks are small
            atr = pd.Series(tr).ewm(span=14, adjust=False).mean().iloc[-1] or 1
            pdi = 100 * pd.Series(plus_dm / atr).ewm(span=14, adjust=False).mean().iloc[-1]
            mdi = 100 * pd.Series(minus_dm / atr).ewm(span=14, adjust=False).mean().iloc[-1]
            tsi = abs(pdi - mdi) / max(pdi + mdi, 1) * 100
        else:
            tsi = 0.0

        vol_sigma = float(np.std(self.mid_prices))

        return FeatureVector(
            ts=latest.ts_local,
            book_imb=book_imb, slope=slope,
            wall_px=wall_px, wall_size=wall_sz,
            cdv=self.cdv_total,
            vwap_delta=vwap_delta,
            tsi=tsi,
            vol_sigma=vol_sigma,
        )

# --------------------------------------------------------------------------- #
# Confidence Engine
# --------------------------------------------------------------------------- #

@dataclass
class Confidence:
    ts: datetime
    score: float       # 0‑100
    half_life: float   # seconds until 50 % reversal probability
    trajectory: str    # up / flat / down

class ConfidenceEngine:
    """
    Simple exponential‑decay confidence model with feature boosts and a
    naive hazard‑based half‑life estimator.
    """

    def __init__(self, in_q: asyncio.Queue, out_q: asyncio.Queue):
        self.in_q, self.out_q = in_q, out_q
        self.score = 50.0
        self.last_trend_flip = datetime.utcnow()

    async def run(self):
        while True:
            fv: FeatureVector = await self.in_q.get()
            prev = self.score

            # Decay toward 50
            λ = math.exp(-1 / CONF_TAU_S)
            self.score = 50 + (self.score - 50) * λ

            # Feature boosts
            self.score += 0.0004 * fv.book_imb
            self.score +=  0.003 * fv.slope
            self.score +=  0.002 * fv.cdv
            self.score += 20    * math.tanh(fv.tsi / 50)
            self.score += 10    * math.tanh(fv.vwap_delta / 0.02)

            # Clamp
            self.score = max(0, min(100, self.score))
            trend_dir = 1 if fv.vwap_delta > 0 else -1 if fv.vwap_delta < 0 else 0

            # Detect flips
            if hasattr(self, "_prev_dir") and trend_dir != self._prev_dir and trend_dir != 0:
                self.last_trend_flip = fv.ts
            self._prev_dir = trend_dir

            # Half‑life estimation – proportional to score & time since flip
            age = (fv.ts - self.last_trend_flip).total_seconds()
            hazard = max(0.01, (100 - self.score) / 200)  # crude
            half_life = math.log(2) / hazard + age*0.1

            traj = "up" if self.score > prev + 1 else "down" if self.score < prev - 1 else "flat"
            conf = Confidence(ts=fv.ts, score=self.score, half_life=half_life, trajectory=traj)
            await self.out_q.put(conf)

# --------------------------------------------------------------------------- #
# Phase Detector
# --------------------------------------------------------------------------- #

Phase = Literal["TrendingUp", "TrendingDown", "Consolidating", "Reversing", "Choppy"]

@dataclass
class PhaseSnapshot:
    ts: datetime
    phase: Phase
    rationale: str

class PhaseDetector:
    def __init__(self, feat_q: asyncio.Queue, conf_q: asyncio.Queue, out_q: asyncio.Queue):
        self.feat_q, self.conf_q, self.out_q = feat_q, conf_q, out_q
        self.latest_feat: Optional[FeatureVector] = None
        self.latest_conf: Optional[Confidence] = None

    async def run(self):
        while True:
            done, _ = await asyncio.wait(
                [self.feat_q.get(), self.conf_q.get()], return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                item = task.result()
                if isinstance(item, FeatureVector):
                    self.latest_feat = item
                else:
                    self.latest_conf = item
            if self.latest_feat and self.latest_conf:
                snap = self._classify(self.latest_feat, self.latest_conf)
                await self.out_q.put(snap)

    def _classify(self, fv: FeatureVector, conf: Confidence) -> PhaseSnapshot:
        if conf.score > 65 and fv.vwap_delta > 0:
            phase = "TrendingUp"
        elif conf.score > 65 and fv.vwap_delta < 0:
            phase = "TrendingDown"
        elif conf.score < 40 and abs(fv.vwap_delta) < 0.01:
            phase = "Consolidating"
        elif conf.trajectory == "down" and conf.score < 50:
            phase = "Reversing"
        else:
            phase = "Choppy"
        rationale = f"imb={fv.book_imb:.0f} cdv={fv.cdv:.0f} vwapΔ={fv.vwap_delta:.3f} score={conf.score:.1f}"
        return PhaseSnapshot(ts=fv.ts, phase=phase, rationale=rationale)

# --------------------------------------------------------------------------- #
# Trader Assist
# --------------------------------------------------------------------------- #

@dataclass
class Advice:
    ts: datetime
    bias: str
    text: str

class TraderAssist:
    def __init__(self, phase_q: asyncio.Queue, conf_q: asyncio.Queue, out_q: asyncio.Queue):
        self.phase_q, self.conf_q, self.out_q = phase_q, conf_q, out_q
        self.phase: Optional[PhaseSnapshot] = None
        self.conf:  Optional[Confidence] = None

    async def run(self):
        while True:
            done, _ = await asyncio.wait(
                [self.phase_q.get(), self.conf_q.get()], return_when=asyncio.FIRST_COMPLETED
            )
            for t in done:
                itm = t.result()
                if isinstance(itm, Confidence):
                    self.conf = itm
                else:
                    self.phase = itm
            if self.conf and self.phase:
                adv = self._make_advice(self.phase, self.conf)
                await self.out_q.put(adv)

    @staticmethod
    def _make_advice(phase: PhaseSnapshot, conf: Confidence) -> Advice:
        if phase.phase == "TrendingUp" and conf.score > 70:
            bias = "Hold Long"
        elif phase.phase == "TrendingDown" and conf.score > 70:
            bias = "Hold Short"
        elif phase.phase == "Reversing":
            bias = "Exit / Flip"
        elif conf.score < 40:
            bias = "Flat / Wait"
        else:
            bias = "Scalp"
        txt = (f"{bias} | {phase.phase} | conf={conf.score:.0f}% "
               f"({conf.trajectory}) hl={conf.half_life:.0f}s")
        return Advice(ts=phase.ts, bias=bias, text=txt)

# --------------------------------------------------------------------------- #
# Dashboard (rich TUI)
# --------------------------------------------------------------------------- #

class Dashboard:
    def __init__(self, conf_q: asyncio.Queue, phase_q: asyncio.Queue, adv_q: asyncio.Queue,
                 feat_q: asyncio.Queue):
        self.conf_q, self.phase_q, self.adv_q, self.feat_q = conf_q, phase_q, adv_q, feat_q
        self.conf: Optional[Confidence] = None
        self.phase: Optional[PhaseSnapshot] = None
        self.adv: Optional[Advice] = None
        self.feat: Optional[FeatureVector] = None

    async def run(self):
        async with Live(self._render(), auto_refresh=False, refresh_per_second=4) as live:
            while True:
                await self._drain_queues()
                live.update(self._render(), refresh=True)
                await asyncio.sleep(REFRESH_SECS)

    async def _drain_queues(self):
        for q in (self.conf_q, self.phase_q, self.adv_q, self.feat_q):
            while not q.empty():
                item = q.get_nowait()
                if isinstance(item, Confidence):
                    self.conf = item
                elif isinstance(item, PhaseSnapshot):
                    self.phase = item
                elif isinstance(item, Advice):
                    self.adv = item
                elif isinstance(item, FeatureVector):
                    self.feat = item

    def _render(self) -> Panel:
        table = Table(title="NG Trend Dashboard", expand=True)
        table.add_column("Metric", justify="left")
        table.add_column("Value", justify="right")

        table.add_row("UTC Time", datetime.utcnow().strftime("%H:%M:%S"))

        if self.conf:
            table.add_row("Confidence", f"{self.conf.score:.1f}% {self._bar(self.conf.score)}")
            table.add_row("Half‑life", f"{self.conf.half_life:.0f} s")
            table.add_row("Trajectory", self.conf.trajectory)

        if self.phase:
            table.add_row("Phase", self.phase.phase)
            table.add_row("Rationale", self.phase.rationale)

        if self.adv:
            table.add_row("Advice", f"[bold]{self.adv.bias}[/] – {self.adv.text}")

        # Raw micro‑metrics
        if self.feat:
            table.add_row("Imbalance", f"{self.feat.book_imb:.0f}")
            table.add_row("Slope", f"{self.feat.slope:.1f}")
            table.add_row("VWAP Δ", f"{self.feat.vwap_delta:+.3f}")
            table.add_row("CDV", f"{self.feat.cdv:+.0f}")
            table.add_row("TSI", f"{self.feat.tsi:.1f}")
            table.add_row("σ(realised)", f"{self.feat.vol_sigma:.4f}")

        return Panel(table, border_style="cyan")

    @staticmethod
    def _bar(pct: float, length=20) -> str:
        filled = int(pct / 100 * length)
        return "[" + "█" * filled + "." * (length - filled) + "]"

# --------------------------------------------------------------------------- #
# Main orchestration
# --------------------------------------------------------------------------- #

async def main():
    q_snap  = asyncio.Queue(maxsize=1000)
    q_feat  = asyncio.Queue(maxsize=1000)
    q_conf  = asyncio.Queue(maxsize=1000)
    q_phase = asyncio.Queue(maxsize=1000)
    q_adv   = asyncio.Queue(maxsize=1000)

    ingestor = Ingestor(q_snap)
    feat_eng = FeatureEngine(q_snap, q_feat)
    conf_eng = ConfidenceEngine(q_feat, q_conf)
    phase_det = PhaseDetector(q_feat, q_conf, q_phase)
    assist    = TraderAssist(q_phase, q_conf, q_adv)
    dash      = Dashboard(q_conf, q_phase, q_adv, q_feat)

    await ingestor.start()               # starts IB + subscriptions

    # ---- graceful shutdown ---- #
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(ingestor)))

    await asyncio.gather(
        feat_eng.run(),
        conf_eng.run(),
        phase_det.run(),
        assist.run(),
        dash.run(),
    )

async def shutdown(ingestor: Ingestor):
    ingestor.ib.disconnect()
    await asyncio.sleep(0.2)
    raise SystemExit

if __name__ == "__main__":
    asyncio.run(main())
