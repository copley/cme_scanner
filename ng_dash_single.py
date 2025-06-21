"""
Real-Time NG Trend Confidence Dashboard (single-file version)
Dependencies: ib-insync, rich, aiosqlite, python-dotenv
"""

import asyncio
import json
import math
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Deque, List, Literal

from dotenv import load_dotenv
from ib_insync import IB, Contract, util
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
import aiosqlite

load_dotenv()

# ----------------------------- Configuration ----------------------------- #

@dataclass(frozen=True)
class Settings:
    tws_host: str = os.getenv("TWS_HOST", "127.0.0.1")
    tws_port: int = int(os.getenv("TWS_PORT", "7497"))
    tws_client_id: int = int(os.getenv("TWS_CLIENT_ID", "42"))
    depth_levels: int = int(os.getenv("DEPTH_LEVELS", "5"))
    roll_window_s: int = int(os.getenv("ROLL_WINDOW_S", "120"))
    conf_tau_s: int = int(os.getenv("CONF_TAU_S", "30"))

settings = Settings()

Side = Literal["bid", "ask"]


# ----------------------------- Level 2 Ingestion ----------------------------- #

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


class L2Ingestor:
    def __init__(self, contract: Contract, queue: asyncio.Queue):
        self.ib = IB()
        self.contract = contract
        self.queue = queue
        self.depth_levels = settings.depth_levels
        self._snapshot: dict[str, list[L2Level]] = {"bid": [], "ask": []}

    async def start(self) -> None:
        await self.ib.connectAsync(
            settings.tws_host, settings.tws_port, clientId=settings.tws_client_id
        )
        self._register_handlers()
        self.ib.reqMktDepth(self.contract, numRows=self.depth_levels, isSmartDepth=False)

    def _register_handlers(self) -> None:
        self.ib.pendingTickersEvent += self._on_tick_price
        self.ib.updateMktDepthEvent += self._on_depth

    def _on_tick_price(self, tickers):
        for t in tickers:
            if t.contract.conId != self.contract.conId:
                continue
            self._last_price = t.last
            break

    def _on_depth(
        self,
        req_id: int,
        position: int,
        operation: int,
        side: int,
        price: float,
        size: float,
        mm_id: str,
    ):
        side_str: Side = "bid" if side == 1 else "ask"
        level = L2Level(
            side=side_str,
            position=position,
            price=price,
            size=size,
            mm_id=mm_id,
            update_type=operation,
        )
        book_side = self._snapshot[side_str]
        while len(book_side) <= position:
            book_side.append(level)
        book_side[position] = level
        if len(self._snapshot["bid"]) and len(self._snapshot["ask"]):
            snap = L2Snapshot(
                bids=self._snapshot["bid"][: self.depth_levels],
                asks=self._snapshot["ask"][: self.depth_levels],
                last_price=getattr(self, "_last_price", 0.0),
                ts_exchange=datetime.now(timezone.utc),
                ts_local=datetime.utcnow().replace(tzinfo=timezone.utc),
            )
            asyncio.create_task(self.queue.put(snap))


# ----------------------------- Feature Engineering ----------------------------- #

@dataclass
class FeatureVector:
    ts: datetime
    book_imbalance: float
    absorption_score: float
    spoof_flag: bool
    wall_px: float | None
    wall_size: float | None
    vwap_delta: float


class FeatureEngine:
    def __init__(self, in_q: asyncio.Queue, out_q: asyncio.Queue):
        self.in_q, self.out_q = in_q, out_q
        self.buffer: Deque[L2Snapshot] = deque(maxlen=settings.roll_window_s)

    async def run(self):
        while True:
            snap: L2Snapshot = await self.in_q.get()
            self.buffer.append(snap)
            if len(self.buffer) < 2:
                continue
            fv = self._compute_features()
            await self.out_q.put(fv)

    def _compute_features(self) -> FeatureVector:
        latest = self.buffer[-1]
        bid_qty = sum(l.size for l in latest.bids)
        ask_qty = sum(l.size for l in latest.asks)
        book_imbalance = bid_qty - ask_qty
        repeat_price = latest.bids[0].price == self.buffer[-2].bids[0].price
        filled_size = self.buffer[-2].bids[0].size - latest.bids[0].size
        absorption_score = max(0.0, filled_size) / max(1.0, self.buffer[-2].bids[0].size)
        spoof_flag = False
        if len(self.buffer) >= 3:
            prev2 = self.buffer[-3].bids[0].size
            prev1 = self.buffer[-2].bids[0].size
            cur = latest.bids[0].size
            if prev1 > prev2 * 2 and cur < prev1 / 2:
                spoof_flag = True
        wall_px, wall_size = None, None
        for lvl in latest.bids + latest.asks:
            if lvl.size > max(bid_qty, ask_qty) * 0.5:
                wall_px, wall_size = lvl.price, lvl.size
                break
        prices = [s.last_price for s in self.buffer]
        sizes = [sum(l.size for l in s.bids + s.asks) for s in self.buffer]
        vwap = sum(p * q for p, q in zip(prices, sizes)) / sum(sizes)
        vwap_delta = latest.last_price - vwap
        return FeatureVector(
            ts=latest.ts_local,
            book_imbalance=book_imbalance,
            absorption_score=absorption_score,
            spoof_flag=spoof_flag,
            wall_px=wall_px,
            wall_size=wall_size,
            vwap_delta=vwap_delta,
        )


# ----------------------------- Confidence Engine ----------------------------- #

@dataclass
class Confidence:
    ts: datetime
    score_pct: float
    decay_pct: float
    trajectory: str


class ConfidenceEngine:
    def __init__(self, in_q: asyncio.Queue, out_q: asyncio.Queue):
        self.in_q, self.out_q = in_q, out_q
        self.prev_score = 50.0

    async def run(self):
        while True:
            fv: FeatureVector = await self.in_q.get()
            new_score = self._calc_score(fv)
            decay = new_score - self.prev_score
            traj = "up" if decay > 1 else "down" if decay < -1 else "flat"
            conf = Confidence(ts=fv.ts, score_pct=new_score, decay_pct=decay, trajectory=traj)
            self.prev_score = new_score
            await self.out_q.put(conf)

    def _calc_score(self, fv: FeatureVector) -> float:
        β1, β2 = 0.0005, 30.0
        base = β1 * fv.book_imbalance + β2 * fv.absorption_score
        if fv.spoof_flag:
            base -= 20.0
        decay_factor = math.exp(-10 / settings.conf_tau_s)
        raw = self.prev_score * decay_factor + base
        return max(0.0, min(100.0, raw))


# ----------------------------- Phase Detector ----------------------------- #

Phase = Literal[
    "TrendingUp",
    "TrendingDown",
    "Expanding",
    "Choppy",
    "Consolidating",
    "Reversing",
]


@dataclass
class PhaseSnapshot:
    ts: datetime
    phase: Phase
    rationale: str


class PhaseDetector:
    def __init__(self, feat_q: asyncio.Queue, conf_q: asyncio.Queue, out_q: asyncio.Queue):
        self.feat_q, self.conf_q, self.out_q = feat_q, conf_q, out_q
        self._latest_feat: FeatureVector | None = None
        self._latest_conf: Confidence | None = None

    async def run(self):
        while True:
            done, _ = await asyncio.wait(
                [self.feat_q.get(), self.conf_q.get()], return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                item = task.result()
                if isinstance(item, FeatureVector):
                    self._latest_feat = item
                else:
                    self._latest_conf = item
            if self._latest_feat and self._latest_conf:
                phase = self._classify(self._latest_feat, self._latest_conf)
                await self.out_q.put(phase)

    def _classify(self, fv: FeatureVector, conf: Confidence) -> PhaseSnapshot:
        if conf.score_pct >= 65 and fv.vwap_delta > 0:
            phase = "TrendingUp"
        elif conf.score_pct >= 65 and fv.vwap_delta < 0:
            phase = "TrendingDown"
        elif abs(fv.book_imbalance) < 1 and abs(fv.vwap_delta) < 0.5:
            phase = "Consolidating"
        elif conf.decay_pct < -10:
            phase = "Reversing"
        else:
            phase = "Choppy"
        return PhaseSnapshot(
            ts=fv.ts,
            phase=phase,
            rationale=f"imb={fv.book_imbalance:.0f}, vwapΔ={fv.vwap_delta:.2f}, conf={conf.score_pct:.1f}",
        )


# ----------------------------- Trader Assist ----------------------------- #

@dataclass
class Advice:
    ts: datetime
    bias: str
    text: str


class TraderAssist:
    def __init__(self, phase_q: asyncio.Queue, conf_q: asyncio.Queue, out_q: asyncio.Queue):
        self.phase_q, self.conf_q, self.out_q = phase_q, conf_q, out_q
        self._last_conf: Confidence | None = None

    async def run(self):
        while True:
            done, _ = await asyncio.wait(
                [self.phase_q.get(), self.conf_q.get()], return_when=asyncio.FIRST_COMPLETED
            )
            for t in done:
                item = t.result()
                if isinstance(item, Confidence):
                    self._last_conf = item
                else:
                    phase = item
            if self._last_conf and isinstance(item, PhaseSnapshot):
                advice = self._advise(phase, self._last_conf)
                await self.out_q.put(advice)

    def _advise(self, phase: PhaseSnapshot, conf: Confidence) -> Advice:
        if phase.phase.startswith("TrendingUp") and conf.score_pct > 70:
            bias = "Hold Long"
        elif phase.phase.startswith("TrendingDown") and conf.score_pct > 70:
            bias = "Hold Short"
        elif phase.phase == "Reversing":
            bias = "Exit / Flip"
        elif conf.score_pct < 40:
            bias = "Flat / Wait"
        else:
            bias = "Scalp"
        txt = f"{bias} | {phase.phase} | conf={conf.score_pct:.0f}% ({conf.trajectory})"
        return Advice(ts=phase.ts, bias=bias, text=txt)


# ----------------------------- Async Logger ----------------------------- #

class LoggerDB:
    DB_PATH = Path("ng_dash.sqlite")

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()

    async def writer(self):
        async with aiosqlite.connect(self.DB_PATH) as db:
            await self._create_schema(db)
            while True:
                item = await self._queue.get()
                if isinstance(item, L2Snapshot):
                    await db.execute(
                        "INSERT OR IGNORE INTO ticks VALUES(?,?,?,?,?)",
                        (
                            int(item.ts_local.timestamp()),
                            item.last_price,
                            json.dumps([l.__dict__ for l in item.bids]),
                            json.dumps([l.__dict__ for l in item.asks]),
                            item.ts_exchange.timestamp(),
                        ),
                    )
                elif isinstance(item, FeatureVector):
                    await db.execute(
                        "INSERT OR IGNORE INTO features VALUES(?,?,?,?,?,?,?)",
                        (
                            int(item.ts.timestamp()),
                            item.book_imbalance,
                            item.absorption_score,
                            int(item.spoof_flag),
                            item.wall_px or 0,
                            item.wall_size or 0,
                            item.vwap_delta,
                        ),
                    )
                elif isinstance(item, Confidence):
                    await db.execute(
                        "INSERT OR IGNORE INTO confidence VALUES(?,?,?,?)",
                        (
                            int(item.ts.timestamp()),
                            item.score_pct,
                            item.decay_pct,
                            item.trajectory,
                        ),
                    )
                elif isinstance(item, PhaseSnapshot):
                    await db.execute(
                        "INSERT OR IGNORE INTO phases VALUES(?,?,?)",
                        (
                            int(item.ts.timestamp()),
                            item.phase,
                            item.rationale,
                        ),
                    )
                elif isinstance(item, Advice):
                    await db.execute(
                        "INSERT OR IGNORE INTO advice VALUES(?,?,?)",
                        (int(item.ts.timestamp()), item.bias, item.text),
                    )
                await db.commit()

    async def _create_schema(self, db):
        await db.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS ticks(
                ts_epoch INTEGER PRIMARY KEY,
                last_price REAL,
                bids_json TEXT,
                asks_json TEXT,
                ts_exch REAL
            );
            CREATE TABLE IF NOT EXISTS features(
                ts_epoch INTEGER PRIMARY KEY,
                book_imb REAL,
                absorption REAL,
                spoof INTEGER,
                wall_px REAL,
                wall_sz REAL,
                vwap_delta REAL
            );
            CREATE TABLE IF NOT EXISTS confidence(
                ts_epoch INTEGER PRIMARY KEY,
                score REAL,
                decay REAL,
                traj TEXT
            );
            CREATE TABLE IF NOT EXISTS phases(
                ts_epoch INTEGER PRIMARY KEY,
                phase TEXT,
                rationale TEXT
            );
            CREATE TABLE IF NOT EXISTS advice(
                ts_epoch INTEGER PRIMARY KEY,
                bias TEXT,
                txt TEXT
            );
            """
        )

    def queue(self) -> asyncio.Queue:
        return self._queue


# ----------------------------- Dashboard ----------------------------- #

class Dashboard:
    def __init__(self, conf_q: asyncio.Queue, phase_q: asyncio.Queue, advice_q: asyncio.Queue):
        self.conf_q, self.phase_q, self.advice_q = conf_q, phase_q, advice_q
        self._latest_conf: Confidence | None = None
        self._latest_phase: PhaseSnapshot | None = None
        self._latest_advice: Advice | None = None

    async def run(self):
        async with Live(self._render(), auto_refresh=False, refresh_per_second=4) as live:
            while True:
                await self._gather_updates()
                live.update(self._render(), refresh=True)
                await asyncio.sleep(10)

    async def _gather_updates(self):
        for q in (self.conf_q, self.phase_q, self.advice_q):
            while not q.empty():
                item = q.get_nowait()
                if isinstance(item, Confidence):
                    self._latest_conf = item
                elif isinstance(item, PhaseSnapshot):
                    self._latest_phase = item
                else:
                    self._latest_advice = item

    def _render(self) -> Panel:
        table = Table(title="NG Trend Confidence", expand=True)
        table.add_column("Metric", justify="left")
        table.add_column("Value", justify="right")

        ts = datetime.utcnow().strftime("%H:%M:%S")
        table.add_row("Timestamp (UTC)", ts)

        if self._latest_conf:
            bar = self._bar(self._latest_conf.score_pct)
            table.add_row("Confidence", f"{self._latest_conf.score_pct:.1f}% {bar}")
            table.add_row("Trajectory", self._latest_conf.trajectory)

        if self._latest_phase:
            table.add_row("Phase", self._latest_phase.phase)
            table.add_row("Rationale", self._latest_phase.rationale)

        if self._latest_advice:
            table.add_row("Advice", self._latest_advice.text)

        return Panel(table, border_style="cyan")

    @staticmethod
    def _bar(pct: float, length: int = 20) -> str:
        filled = int(pct / 100 * length)
        return "[" + "█" * filled + "." * (length - filled) + "]"


# ----------------------------- Main ----------------------------- #

async def main():
    q_snap = asyncio.Queue(maxsize=1000)
    q_feat = asyncio.Queue(maxsize=1000)
    q_conf = asyncio.Queue(maxsize=1000)
    q_phase = asyncio.Queue(maxsize=1000)
    q_adv = asyncio.Queue(maxsize=1000)

    logger = LoggerDB()
    asyncio.create_task(logger.writer())

    ng_contract = Contract(symbol="NG", secType="FUT", exchange="NYMEX", currency="USD")

    ingestor = L2Ingestor(contract=ng_contract, queue=q_snap)
    feat_eng = FeatureEngine(q_snap, q_feat)
    conf_eng = ConfidenceEngine(q_feat, q_conf)
    phase_det = PhaseDetector(q_feat, q_conf, q_phase)
    assist = TraderAssist(q_phase, q_conf, q_adv)
    dash = Dashboard(q_conf, q_phase, q_adv)

    await ingestor.start()

    await asyncio.gather(
        feat_eng.run(),
        conf_eng.run(),
        phase_det.run(),
        assist.run(),
        dash.run(),
    )


if __name__ == "__main__":
    asyncio.run(main())
