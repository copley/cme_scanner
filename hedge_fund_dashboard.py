#!/usr/bin/env python3
"""
NG Hedge‑Fund Dashboard  — MULTI‑HORIZON EDITION
------------------------------------------------------------
• L2 market‑depth + full tick stream (ib_insync)
• Multi‑time‑frame bar builders    (4 h, 1 h, 15 m, 1 m, 1 s)
• Micro‑structure metrics          (imbalance, wall, VWAPΔ)
• Technical indicators             (ADX/ADXR, Vortex, BB‑Width,
                                     ATR%, RSI, MACD, DonchianW,
                                     Choppiness, OBV, CMF)
• Confidence / Phase / Advice engine (1‑s stream)
• NEW  Trend‑fusion & Profit‑plan modules
• 5‑panel Rich dashboard: Edge, Health, Global Trend,
  Time‑Frame Stack, Profit Plan
------------------------------------------------------------
Dependencies: ib‑insync, rich, python‑dotenv, pandas, numpy,
              psutil
"""

from __future__ import annotations

import asyncio, math, os, signal, psutil, tracemalloc
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from ib_insync import IB, Contract, Ticker
from rich import box
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

# --------------------------------------------------------------------------- #
#  Global cancellation signal
# --------------------------------------------------------------------------- #

STOP: asyncio.Event = asyncio.Event()


def _install_signal_handlers() -> None:
    """Convert SIGINT / SIGTERM into STOP.set() without breaking Windows."""
    loop = asyncio.get_running_loop()

    def _handler() -> None:
        STOP.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handler)
        except (NotImplementedError, RuntimeError):
            signal.signal(sig, lambda *_: _handler())


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #

load_dotenv()


@dataclass(frozen=True)
class Settings:
    tws_host: str = os.getenv("TWS_HOST", "127.0.0.1")
    tws_port: int = int(os.getenv("TWS_PORT", 7497))
    tws_cid: int = int(os.getenv("TWS_CLIENT_ID", 42))
    depth_rows: int = int(os.getenv("DEPTH_LEVELS", 5))

    roll_secs: int = int(os.getenv("ROLL_WINDOW_S", 120))   # bar look‑back
    conf_tau: int = int(os.getenv("CONF_TAU_S", 30))        # confidence decay
    refresh: int = int(os.getenv("REFRESH_SECS", 10))       # dashboard redraw


S = Settings()
Side = Literal["bid", "ask"]

# --------------------------------------------------------------------------- #
#  Order‑book ingestion
# --------------------------------------------------------------------------- #

@dataclass
class L2Level:
    side: Side
    pos: int
    price: float
    size: float
    mmid: str
    op: int


@dataclass
class Snapshot:
    bids: List[L2Level]
    asks: List[L2Level]
    last: float
    ts: datetime


class L2Ingestor:
    """Grabs depth + last trade price → emits Snapshot every 100 ms."""

    def __init__(self, contract: Contract, q_out: asyncio.Queue):
        self.ib = IB()
        self.contract = contract
        self.q_out = q_out
        self.depth: Dict[Side, Dict[int, L2Level]] = {"bid": {}, "ask": {}}
        self.last = 0.0
        self._depth_sub: Optional[int] = None
        self._mkt_sub = None
        self._tbt_sub = None

    # ---------- connection helpers ---------- #
    @property
    def connected(self) -> bool:
        return self.ib.isConnected()

    async def start(self) -> None:
        await self.ib.connectAsync(S.tws_host, S.tws_port, clientId=S.tws_cid)

        self.ib.updateMktDepthEvent += self._on_depth
        self._depth_sub = self.ib.reqMktDepth(
            self.contract, numRows=S.depth_rows
        ).reqId

        self._mkt_sub = self.ib.reqMktData(self.contract, "", snapshot=False)
        self._mkt_sub.updateEvent += self._on_ticker

        self._tbt_sub = self.ib.reqTickByTickData(
            self.contract, "Last", 0, False
        )
        self._tbt_sub.updateEvent += self._on_tbtick

        asyncio.create_task(self._emit_loop())

    async def stop(self) -> None:
        if self._depth_sub is not None:
            self.ib.cancelMktDepth(self._depth_sub)
        if self._mkt_sub is not None:
            self.ib.cancelMktData(self._mkt_sub.tickerId)
        if self._tbt_sub is not None:
            self.ib.cancelTickByTickData(self._tbt_sub.reqId)
        if self.connected:
            await self.ib.disconnectAsync()

    # ---------- IB callbacks ---------- #
    def _on_depth(
        self,
        _ticker: Ticker,
        pos,
        mmid,
        op,
        side,
        price,
        size,
        isSmartDepth=None,
    ):
        s: Side = "bid" if side == 1 else "ask"
        self.depth[s][pos] = L2Level(s, pos, price, size, mmid or "", op)

    def _on_ticker(self, ticker: Ticker):
        if ticker.last is not None:
            self.last = ticker.last

    def _on_tbtick(self, tick):
        if tick.price is not None and tick.price > 0:
            self.last = tick.price

    # ---------- snapshot pump ---------- #
    async def _emit_loop(self) -> None:
        while not STOP.is_set():
            await asyncio.sleep(0.1)
            if not (self.depth["bid"] and self.depth["ask"]):
                continue
            bids = [
                self.depth["bid"].get(i, L2Level("bid", i, 0, 0, "", 0))
                for i in range(S.depth_rows)
            ]
            asks = [
                self.depth["ask"].get(i, L2Level("ask", i, 0, 0, "", 0))
                for i in range(S.depth_rows)
            ]
            snap = Snapshot(
                bids=bids,
                asks=asks,
                last=self.last,
                ts=datetime.utcnow().replace(tzinfo=timezone.utc),
            )
            try:
                self.q_out.put_nowait(snap)
            except asyncio.QueueFull:
                pass


# --------------------------------------------------------------------------- #
#  Broadcast hub
# --------------------------------------------------------------------------- #

class SnapshotBroadcaster:
    def __init__(self, q_in: asyncio.Queue, *subscribers: asyncio.Queue):
        self.q_in = q_in
        self.subs = subscribers

    async def run(self):
        while not STOP.is_set():
            snap: Snapshot = await self.q_in.get()
            for q in self.subs:
                try:
                    q.put_nowait(snap)
                except asyncio.QueueFull:
                    pass


# --------------------------------------------------------------------------- #
#  Bar builder  – horizon‑aware
# --------------------------------------------------------------------------- #

@dataclass
class Bar:
    ts: datetime
    high: float
    low: float
    close: float


class BarBuilder:
    """Aggregates snapshots into OHLC bars of arbitrary length."""

    def __init__(
        self,
        q_in: asyncio.Queue,
        q_out: asyncio.Queue,
        *,
        bar_seconds: int = 1,
    ):
        self.q_in, self.q_out = q_in, q_out
        self.bar_seconds = bar_seconds
        self.curr: Optional[Bar] = None
        self.window: Deque[Bar] = deque(maxlen=S.roll_secs)

    async def run(self):
        while not STOP.is_set():
            snap: Snapshot = await self.q_in.get()
            ts0 = snap.ts.astimezone(timezone.utc)
            epoch = int(ts0.timestamp())
            bar_start = epoch - epoch % self.bar_seconds
            ts = datetime.fromtimestamp(bar_start, tz=timezone.utc)

            high = snap.asks[0].price
            low = snap.bids[0].price
            if self.curr and ts == self.curr.ts:
                self.curr.high = max(self.curr.high, high)
                self.curr.low = min(self.curr.low, low)
                self.curr.close = (high + low) / 2
            else:
                if self.curr:
                    self.window.append(self.curr)
                    await self.q_out.put(self.curr)
                self.curr = Bar(ts, high, low, (high + low) / 2)

    def prices(self):
        return [b.close for b in self.window]


# --------------------------------------------------------------------------- #
#  Trend‑state & fusion dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class TrendState:
    ts: datetime
    tf: str
    dir: int           # +1 / -1 / 0
    strength: float    # 0‑100
    half_life: float   # seconds
    label: str


@dataclass
class GlobalTrend:
    ts: datetime
    dir: int
    strength: float
    dominant_tf: str
    agreement: float


# --------------------------------------------------------------------------- #
#  ----- TEMP stubs to avoid NameError – replace with real code -------
#  If you want to run *only* the new plumbing without indicators,
#  uncomment this block *and* comment‑out the “indicator math” section
#  further below.
# ---------------------------------------------------------------------------
"""
def adx(*a, **k): return (None, None)
def vortex(*a, **k): return (None, None)
def bb_width(*a, **k): return None
def atr_pct(*a, **k): return None
def rsi(*a, **k): return None
def macd(*a, **k): return (None, None)
def donchian_width(*a, **k): return None
def choppiness(*a, **k): return None
def obv_proxy(*a, **k): return None
def cmf_proxy(*a, **k): return None

Phase = Literal["Choppy"]
@dataclass
class PhaseSnap:
    ts: datetime
    phase: str
    rationale: str
@dataclass
class Advice:
    ts: datetime
    text: str
class PhaseDetector:
    async def run(self, *a, **k): pass
class Advisor:
    async def run(self, *a, **k): pass
"""
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#  REAL indicator math
# --------------------------------------------------------------------------- #

def _ema(arr: List[float], span: int) -> float:
    return pd.Series(arr).ewm(span=span, adjust=False).mean().iloc[-1]


def adx(high: List[float], low: List[float], close: List[float], period: int = 14):
    if len(close) < period + 1:
        return None, None
    h, l, c = np.array(high), np.array(low), np.array(close)
    up = h[1:] - h[:-1]
    down = l[:-1] - l[1:]
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    tr = np.maximum.reduce(
        [h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])]
    )
    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_val = float(dx.ewm(alpha=1 / period, adjust=False).mean().iloc[-1])
    adxr = float((adx_val + dx.iloc[-period]) / 2) if len(dx) >= period + 1 else None
    return adx_val, adxr


def vortex(high, low, close, period=14):
    if len(close) <= period:
        return None, None
    h, l, c = np.array(high), np.array(low), np.array(close)
    tr = np.sum(
        np.maximum.reduce(
            [h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])]
        )[-period:]
    )
    vip = np.sum(np.abs(h[1:] - l[:-1])[-period:]) / tr
    vin = np.sum(np.abs(l[1:] - h[:-1])[-period:]) / tr
    return vip, vin


def bb_width(close, period=20, mult=2):
    if len(close) < period:
        return None
    s = pd.Series(close)
    mid = s.rolling(period).mean().iloc[-1]
    sd = s.rolling(period).std(ddof=0).iloc[-1]
    return (2 * mult * sd) / mid * 100 if mid else None


def atr_pct(high, low, close, period=14):
    if len(close) <= period:
        return None
    h, l, c = np.array(high), np.array(low), np.array(close)
    tr = np.maximum.reduce(
        [h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])]
    )
    atr = np.mean(tr[-period:])
    return atr / close[-1] * 100


def rsi(close, period=14):
    if len(close) < period + 1:
        return None
    diff = np.diff(close)
    up = np.where(diff > 0, diff, 0)
    down = np.where(diff < 0, -diff, 0)
    rs = np.mean(up[-period:]) / max(np.mean(down[-period:]), 1e-9)
    return 100 - (100 / (1 + rs))


def macd(close, fast=12, slow=26, signal=9):
    if len(close) < slow + signal:
        return None, None
    fast_ema = _ema(close, fast)
    slow_ema = _ema(close, slow)
    macd_line = fast_ema - slow_ema
    hist_arr = pd.Series(close).ewm(span=fast, adjust=False).mean() - pd.Series(
        close
    ).ewm(span=slow, adjust=False).mean()
    sig_line = hist_arr.ewm(span=signal, adjust=False).mean().iloc[-1]
    return macd_line, sig_line


def donchian_width(high, low, period=20):
    if len(high) < period:
        return None
    return (max(high[-period:]) - min(low[-period:])) / (
        (high[-1] + low[-1]) / 2
    ) * 100


def choppiness(high, low, close, period=14):
    if len(close) < period + 1:
        return None
    h, l, c = np.array(high), np.array(low), np.array(close)
    tr = np.maximum.reduce(
        [h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])]
    )
    atr_sum = np.sum(tr[-period:])
    high_max = np.max(h[-period:])
    low_min = np.min(l[-period:])
    return 100 * math.log10(atr_sum / (high_max - low_min)) / math.log10(period)


def obv_proxy(close, book_imb_arr):
    if len(close) < 2:
        return None
    direction = np.sign(np.diff(close))
    return float(np.sum(direction * np.array(book_imb_arr[-len(direction):])))


def cmf_proxy(book_sizes_bid, book_sizes_ask):
    if not book_sizes_bid or not book_sizes_ask:
        return None
    vol_bid = sum(book_sizes_bid)
    vol_ask = sum(book_sizes_ask)
    denom = vol_bid + vol_ask
    return (vol_bid - vol_ask) / denom if denom else None


# --------------------------------------------------------------------------- #
#  Feature engine
# --------------------------------------------------------------------------- #

@dataclass
class FeatureVector:
    ts: datetime
    mid: float
    vwap_delta: float
    book_imb: float
    wall_sz: float | None
    adx: float | None
    adxr: float | None
    vip: float | None
    vin: float | None
    bb_width: float | None
    atr_pct: float | None
    rsi: float | None
    macd: float | None
    macd_sig: float | None
    don_w: float | None
    chop: float | None
    obv: float | None
    cmf: float | None
    trend_age: int
    exhaustion: float | None
    edge_score: float
    horizon: str


class FeatureEngine:
    """Produces FeatureVector **and** TrendState for a single horizon."""

    def __init__(
        self,
        snap_q: asyncio.Queue,
        bar_q: asyncio.Queue,
        out_feat_q: asyncio.Queue,
        out_trend_q: asyncio.Queue,
        *,
        horizon_name: str,
    ):
        self.snap_q, self.bar_q = snap_q, bar_q
        self.out_feat_q, self.out_trend_q = out_feat_q, out_trend_q
        self.horizon = horizon_name
        self.bars: Deque[Bar] = deque(maxlen=S.roll_secs)
        self.mid_prices: Deque[float] = deque(maxlen=300)
        self.book_hist: Deque[float] = deque(maxlen=300)
        self.trend_age = 0

    async def run(self):
        asyncio.create_task(self._bar_listener())
        while not STOP.is_set():
            snap: Snapshot = await self.snap_q.get()
            mid = (snap.bids[0].price + snap.asks[0].price) / 2
            self.mid_prices.append(mid)
            book_imb = sum(b.size for b in snap.bids) - sum(a.size for a in snap.asks)
            self.book_hist.append(book_imb)
            wall_sz = max((lvl.size for lvl in (*snap.bids, *snap.asks)), default=0)
            vwap = sum(self.mid_prices) / len(self.mid_prices)
            vwap_delta = mid - vwap

            ind_ready = len(self.bars) >= 26
            adx_val = adxr = v_ip = v_in = bbw = atrp = None
            rs = mcd = mcd_sig = donw = chp = obv = cmf = None
            if ind_ready:
                h = [b.high for b in self.bars]
                l = [b.low for b in self.bars]
                c = [b.close for b in self.bars]

                adx_val, adxr = adx(h, l, c)
                v_ip, v_in = vortex(h, l, c)
                bbw = bb_width(c)
                atrp = atr_pct(h, l, c)
                rs = rsi(c)
                mcd, mcd_sig = macd(c)
                donw = donchian_width(h, l)
                chp = choppiness(h, l, c)
                obv = obv_proxy(c, list(self.book_hist))
                cmf = cmf_proxy([b.size for b in snap.bids], [a.size for a in snap.asks])

            if adx_val and adx_val > 25:
                self.trend_age += 1
            else:
                self.trend_age = 0
            exhaustion = None if self.trend_age < 14 or not adx_val else max(0, 50 - adx_val)

            edge = min(100, abs(vwap_delta) * 4000 + abs(book_imb) / 40)

            fv = FeatureVector(
                snap.ts,
                mid,
                vwap_delta,
                book_imb,
                wall_sz,
                adx_val,
                adxr,
                v_ip,
                v_in,
                bbw,
                atrp,
                rs,
                mcd,
                mcd_sig,
                donw,
                chp,
                obv,
                cmf,
                self.trend_age,
                exhaustion,
                edge,
                self.horizon,
            )
            await self.out_feat_q.put(fv)

            if ind_ready and adx_val is not None and rs is not None and atrp:
                atr_abs = atrp * mid / 100
                dir_ = 1 if adx_val > 25 and vwap_delta > 0 and rs > 55 else \
                       -1 if adx_val > 25 and vwap_delta < 0 and rs < 45 else 0
                strength = min(
                    100, (adx_val - 20) * 2 + abs(vwap_delta) / max(atr_abs, 1e-6) * 40
                )
                half_life = max(20, 600 * (strength / 100) ** 2)
                label = (
                    "StrongUp" if dir_ == 1 and strength > 70 else
                    "WeakUp" if dir_ == 1 else
                    "StrongDn" if dir_ == -1 and strength > 70 else
                    "WeakDn" if dir_ == -1 else
                    "Sideways"
                )
                await self.out_trend_q.put(
                    TrendState(fv.ts, self.horizon, dir_, strength, half_life, label)
                )

    async def _bar_listener(self):
        while not STOP.is_set():
            bar: Bar = await self.bar_q.get()
            self.bars.append(bar)


# --------------------------------------------------------------------------- #
#  Trend‑fusion engine
# --------------------------------------------------------------------------- #

class TrendFusion:
    def __init__(self, in_queues: Dict[str, asyncio.Queue], out_q: asyncio.Queue):
        self.in_queues = in_queues
        self.out_q = out_q
        self.latest: Dict[str, TrendState] = {}

    async def run(self):
        pending = {tf: asyncio.create_task(q.get()) for tf, q in self.in_queues.items()}
        while not STOP.is_set():
            done, _ = await asyncio.wait(
                pending.values(), return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                ts: TrendState = task.result()
                self.latest[ts.tf] = ts
                pending[ts.tf] = asyncio.create_task(self.in_queues[ts.tf].get())

            if len(self.latest) < 2:
                continue

            score = sum(ts.dir * math.log(ts.half_life) for ts in self.latest.values())
            dir_master = 1 if score > 0 else -1 if score < 0 else 0
            weights = [abs(math.log(ts.half_life)) for ts in self.latest.values()]
            strength_master = float(
                np.average([ts.strength for ts in self.latest.values()], weights=weights)
            )
            agreeing = sum(
                1 for ts in self.latest.values() if ts.dir == dir_master and dir_master != 0
            )
            total_nonzero = sum(1 for ts in self.latest.values() if ts.dir != 0)
            agreement = agreeing / total_nonzero if total_nonzero else 0.0
            dominant = max(
                (ts for ts in self.latest.values() if ts.dir == dir_master),
                key=lambda t: t.half_life,
                default=list(self.latest.values())[0],
            )
            await self.out_q.put(
                GlobalTrend(datetime.utcnow(), dir_master, strength_master, dominant.tf, agreement)
            )


# --------------------------------------------------------------------------- #
#  Profit‑plan engine
# --------------------------------------------------------------------------- #

@dataclass
class ProfitPlan:
    ts: datetime
    trailing_stop: float
    next_pt: float
    rr: float
    half_life: float


class ProfitPlanEngine:
    """Uses GlobalTrend + last 1‑m FeatureVector to compute a simple R:R."""

    def __init__(
        self,
        global_q: asyncio.Queue,
        feat1m_q: asyncio.Queue,
        out_q: asyncio.Queue,
    ):
        self.gq, self.fq, self.oq = global_q, feat1m_q, out_q
        self.last_feat: Optional[FeatureVector] = None

    async def run(self):
        while not STOP.is_set():
            done, _ = await asyncio.wait(
                [self.gq.get(), self.fq.get()],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in done:
                obj = t.result()
                if isinstance(obj, GlobalTrend):
                    g = obj
                    if self.last_feat and self.last_feat.atr_pct:
                        mid = self.last_feat.mid
                        atr = self.last_feat.atr_pct * mid / 100
                        stop = mid - 1.2 * atr if g.dir == 1 else mid + 1.2 * atr
                        pt1 = mid + 0.5 * atr if g.dir == 1 else mid - 0.5 * atr
                        rr = abs(pt1 - mid) / abs(mid - stop) if stop != mid else 0
                        await self.oq.put(ProfitPlan(g.ts, stop, pt1, rr, g.strength))
                else:
                    if obj.horizon == "1m":
                        self.last_feat = obj


# --------------------------------------------------------------------------- #
#  Confidence / Phase / Advice  (original single‑horizon logic)
# --------------------------------------------------------------------------- #

@dataclass
class Confidence:
    ts: datetime
    score: float
    traj: str


class ConfidenceEngine:
    def __init__(self, in_q: asyncio.Queue, out_q: asyncio.Queue):
        self.in_q, self.out_q = in_q, out_q
        self.score = 50.0

    async def run(self):
        while not STOP.is_set():
            fv: FeatureVector = await self.in_q.get()
            prev = self.score
            self.score = 50 + (self.score - 50) * math.exp(-1 / S.conf_tau)
            self.score += 0.0004 * fv.book_imb + 25 * math.tanh(fv.vwap_delta / 0.02)
            if fv.adx:
                self.score += (fv.adx - 25) * 0.4
            self.score = max(0, min(100, self.score))
            traj = (
                "up" if self.score > prev + 1 else
                "down" if self.score < prev - 1 else
                "flat"
            )
            await self.out_q.put(Confidence(fv.ts, self.score, traj))


Phase = Literal[
    "TrendingUp",
    "TrendingDown",
    "Ranging",
    "Volatile",
    "Consolidating",
    "Breakout",
    "Reversing",
    "Choppy",
    "Expanding",
    "Contracting",
    "ProfitTaking",
    "BreakoutFailed",
]


@dataclass
class PhaseSnap:
    ts: datetime
    phase: Phase
    rationale: str


class PhaseDetector:
    def __init__(self, feat_q, conf_q, out_q):
        self.fq, self.cq, self.oq = feat_q, conf_q, out_q
        self.fv: FeatureVector | None = None
        self.cf: Confidence | None = None
        self.prev_phase: Optional[Phase] = None

    async def run(self):
        while not STOP.is_set():
            done, _ = await asyncio.wait(
                [self.fq.get(), self.cq.get()],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in done:
                obj = t.result()
                if isinstance(obj, FeatureVector):
                    self.fv = obj
                else:
                    self.cf = obj
            if self.fv and self.cf:
                snap = self._classify()
                self.prev_phase = snap.phase
                await self.oq.put(snap)

    def _classify(self) -> PhaseSnap:
        fv, cf = self.fv, self.cf
        if fv.atr_pct and fv.atr_pct > 3.0:
            phase: Phase = "Volatile"
        elif fv.bb_width and fv.bb_width < 4 and 40 <= cf.score <= 60:
            phase = "Ranging"
        elif fv.don_w and fv.don_w > 8 and cf.score > 65:
            phase = "Expanding"
        elif fv.bb_width and fv.bb_width > 8 and cf.score > 65:
            phase = "Breakout"
        elif self.prev_phase == "Breakout" and cf.traj == "down" and cf.score < 60:
            phase = "BreakoutFailed"
        elif cf.score > 70 and fv.vwap_delta > 0:
            phase = "TrendingUp"
        elif cf.score > 70 and fv.vwap_delta < 0:
            phase = "TrendingDown"
        elif cf.score < 40 and (fv.bb_width or 0) < 5:
            phase = "Consolidating"
        elif cf.traj == "down" and cf.score < 50:
            phase = "Reversing"
        elif fv.chop and fv.chop > 60:
            phase = "Contracting"
        else:
            phase = "Choppy"

        if phase in ("TrendingUp", "TrendingDown") and fv.exhaustion and fv.exhaustion > 10:
            phase = "ProfitTaking"

        rat = " ".join(
            f"{k}={v:.1f}"
            for k, v in {
                "adx": fv.adx,
                "bb%": fv.bb_width,
                "atr%": fv.atr_pct,
                "rsi": fv.rsi,
            }.items()
            if v is not None
        ) or "—"
        return PhaseSnap(fv.ts, phase, rat)


@dataclass
class Advice:
    ts: datetime
    text: str


class Advisor:
    def __init__(self, phase_q, conf_q, out_q):
        self.pq, self.cq, self.oq = phase_q, conf_q, out_q
        self.ph: PhaseSnap | None = None
        self.cf: Confidence | None = None

    async def run(self):
        while not STOP.is_set():
            done, _ = await asyncio.wait(
                [self.pq.get(), self.cq.get()],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in done:
                obj = t.result()
                if isinstance(obj, PhaseSnap):
                    self.ph = obj
                else:
                    self.cf = obj
            if self.ph and self.cf:
                await self.oq.put(self._make())

    def _make(self) -> Advice:
        ph, sc = self.ph.phase, self.cf.score
        if ph == "TrendingUp" and sc > 70:
            bias = "🚀 Hold Long"
        elif ph == "TrendingDown" and sc > 70:
            bias = "🔻 Hold Short"
        elif ph in ("Reversing", "ProfitTaking", "BreakoutFailed"):
            bias = "⚠ Exit / Flip"
        elif sc < 40:
            bias = "⏸ Wait"
        else:
            bias = "🔄 Scalp"
        return Advice(self.ph.ts, f"{bias} | conf={sc:.0f}% ({self.cf.traj})")


# --------------------------------------------------------------------------- #
#  Dashboard (five panels)
# --------------------------------------------------------------------------- #

class HFDashboard:
    def __init__(
        self,
        feat_q,
        conf_q,
        phase_q,
        adv_q,
        ingest_q,
        ingestor: L2Ingestor,
        trend_stack_qs: Dict[str, asyncio.Queue],
        global_q: asyncio.Queue,
        profit_q: asyncio.Queue,
    ):
        self.fq, self.cq, self.pq, self.aq, self.iq = feat_q, conf_q, phase_q, adv_q, ingest_q
        self.ingestor = ingestor
        self.stack_qs = trend_stack_qs
        self.gq = global_q
        self.ppq = profit_q

        self.fv: FeatureVector | None = None
        self.cf: Confidence | None = None
        self.ph: PhaseSnap | None = None
        self.ad: Advice | None = None
        self.stack: Dict[str, TrendState] = {}
        self.gtrend: Optional[GlobalTrend] = None
        self.pp: Optional[ProfitPlan] = None
        self.last = datetime.utcnow()

    async def run(self):
        lay = self._layout()
        async with Live(lay, screen=False, refresh_per_second=4) as live:
            while not STOP.is_set():
                self._drain()
                lay["edge"].update(self._edge_panel())
                lay["health"].update(self._health_panel())
                lay["trend"].update(self._trend_panel())
                lay["stack"].update(self._stack_panel())
                lay["profit"].update(self._profit_panel())
                live.refresh()
                await asyncio.sleep(S.refresh)

    # ---------- layout ---------- #
    def _layout(self):
        root = Layout()
        root.split_row(Layout(name="edge", ratio=3), Layout(name="health", ratio=2))
        root.split(
            Layout(name="trend", ratio=2),
            Layout(name="stack", ratio=2),
            Layout(name="profit", ratio=2),
            direction="vertical",
        )
        return root

    # ---------- panels ---------- #
    def _edge_panel(self) -> Panel:
        if not self.fv:
            return Panel("Waiting …", title="⚡ Edge Forecast", box=box.ROUNDED, border_style="grey50")
        f = self.fv
        T = Table.grid(expand=True)
        T.add_column(); T.add_column(justify="right")
        T.add_row("🕒 Time", f.ts.strftime("%H:%M:%S"))
        T.add_row("📈 Mid", f"{f.mid:.4f}")
        T.add_row("🧮 VWAP Δ", f"{f.vwap_delta:+.4f}")
        T.add_row("🧲 Imbalance", f"{f.book_imb:+.0f}")
        T.add_row("🧱 Wall", f"{f.wall_sz or 0:.0f}")
        if f.adx:        T.add_row("⚡ ADX", f"{f.adx:.1f}")
        if f.bb_width:   T.add_row("📏 BB‑W", f"{f.bb_width:.1f}%")
        if f.rsi:        T.add_row("🌀 RSI", f"{f.rsi:.1f}")
        if f.don_w:      T.add_row("📣 DonW", f"{f.don_w:.1f}%")
        T.add_row("💡 Edge", self._edge_str(f.edge_score))
        return Panel(T, title="⚡ NG Edge Forecast", border_style="white", box=box.ROUNDED)

    def _health_panel(self) -> Panel:
        lag = (datetime.utcnow() - self.last).total_seconds()
        backlog = self.iq.qsize()
        mem = psutil.Process().memory_info().rss / (1024 * 1024)
        conn = "[green]LIVE[/]" if self.ingestor.connected else "[red]DOWN[/]"
        T = Table.grid(expand=True)
        T.add_column(); T.add_column(justify="right")
        T.add_row("IB Conn", conn)
        T.add_row("Lag", f"{lag:.1f}s")
        T.add_row("Backlog", str(backlog))
        T.add_row("RSS", f"{mem:.1f} MB")
        status = "OK" if lag < S.refresh * 2 and self.ingestor.connected else "[red]STALE[/]"
        T.add_row("Status", status)
        return Panel(T, title="🛠 System Health", border_style="cyan", box=box.ROUNDED)

    def _trend_panel(self) -> Panel:
        if not self.gtrend:
            return Panel("…", title="Trend", border_style="grey50", box=box.ROUNDED)
        dir_sym = "▲" if self.gtrend.dir == 1 else "▼" if self.gtrend.dir == -1 else "→"
        bar = self._bar(self.gtrend.strength)
        T = Table.grid(expand=True)
        T.add_column(); T.add_column(justify="right")
        T.add_row("Dir / Str", f"{dir_sym} {self.gtrend.strength:.1f}% {bar}")
        T.add_row("Dominant TF", self.gtrend.dominant_tf)
        T.add_row("Agreement", f"{self.gtrend.agreement*100:.0f}%")
        color = "green" if self.gtrend.dir == 1 else "red" if self.gtrend.dir == -1 else "yellow"
        return Panel(T, title="📊 Global Trend", border_style=color, box=box.ROUNDED)

    def _stack_panel(self) -> Panel:
        if not self.stack:
            return Panel("…", title="TF Stack", border_style="grey50", box=box.ROUNDED)
        order = ["4h", "1h", "15m", "1m", "1s"]
        T = Table(show_header=True, header_style="bold")
        T.add_column("TF"); T.add_column("Dir"); T.add_column("Str"); T.add_column("½‑life")
        for tf in order:
            ts = self.stack.get(tf)
            if not ts:
                continue
            arrow = "▲" if ts.dir == 1 else "▼" if ts.dir == -1 else "→"
            bar = self._bar(ts.strength, l=10)
            T.add_row(tf, arrow, bar, f"{ts.half_life/60:.1f} m")
        return Panel(T, title="🪜 Time‑Frame Stack", border_style="white", box=box.ROUNDED)

    def _profit_panel(self) -> Panel:
        if not self.pp:
            return Panel("…", title="Profit Plan", border_style="grey50", box=box.ROUNDED)
        T = Table.grid(expand=True)
        T.add_column(); T.add_column(justify="right")
        T.add_row("Trailing Stop", f"{self.pp.trailing_stop:.3f}")
        T.add_row("Next PT", f"{self.pp.next_pt:.3f}")
        T.add_row("R‑R", f"{self.pp.rr:.2f}")
        T.add_row("Trend ½‑life", f"{self.pp.half_life/60:.1f} m")
        color = "yellow" if self.pp.rr < 1 else "green"
        return Panel(T, title="💰 Profit Plan", border_style=color, box=box.ROUNDED)

    # ---------- drain ---------- #
    def _drain(self) -> None:
        for q in (self.fq, self.cq, self.pq, self.aq):
            while not q.empty():
                o = q.get_nowait()
                if isinstance(o, FeatureVector):
                    self.fv = o
                elif isinstance(o, Confidence):
                    self.cf = o
                elif isinstance(o, PhaseSnap):
                    self.ph = o
                elif isinstance(o, Advice):
                    self.ad = o
        for tf, q in self.stack_qs.items():
            while not q.empty():
                self.stack[tf] = q.get_nowait()
        while not self.gq.empty():
            self.gtrend = self.gq.get_nowait()
        while not self.ppq.empty():
            self.pp = self.ppq.get_nowait()
        self.last = datetime.utcnow()

    # ---------- helpers ---------- #
    @staticmethod
    def _bar(pct: float, l: int = 14) -> str:
        filled = int(pct / 100 * l)
        return "[" + "█" * filled + "." * (l - filled) + "]"

    @staticmethod
    def _edge_str(score: float) -> str:
        color = "green" if score > 85 else "yellow" if score > 60 else "white"
        tag = " ⚑HIGH" if score > 85 else ""
        return f"[{color}]{score:5.2f}[/{color}]{tag}"


# --------------------------------------------------------------------------- #
#  Contract helper
# --------------------------------------------------------------------------- #

def front_ng_contract() -> Contract:
    return Contract(
        symbol="NG",
        secType="FUT",
        exchange="NYMEX",
        currency="USD",
        lastTradeDateOrContractMonth="20250626",
    )


# --------------------------------------------------------------------------- #
#  Main orchestration
# --------------------------------------------------------------------------- #

async def main():
    _install_signal_handlers()

    # snapshot queues
    q_ingest = asyncio.Queue(3000)
    snap_qs = {tf: asyncio.Queue(2000) for tf in ("1s", "1m", "15m", "1h", "4h")}

    bar_qs   = {tf: asyncio.Queue(1000) for tf in snap_qs}
    feat_qs  = {tf: asyncio.Queue(1000) for tf in snap_qs}
    ts_qs    = {tf: asyncio.Queue(200)  for tf in snap_qs}

    q_conf   = asyncio.Queue(1000)
    q_phase  = asyncio.Queue(1000)
    q_adv    = asyncio.Queue(1000)
    q_global = asyncio.Queue(200)
    q_profit = asyncio.Queue(200)

    ingestor = L2Ingestor(front_ng_contract(), q_ingest)
    fanout = SnapshotBroadcaster(q_ingest, *snap_qs.values())

    builders = {
        '1s':  BarBuilder(snap_qs['1s'],  bar_qs['1s'],  bar_seconds=1),
        '1m':  BarBuilder(snap_qs['1m'],  bar_qs['1m'],  bar_seconds=60),
        '15m': BarBuilder(snap_qs['15m'], bar_qs['15m'], bar_seconds=900),
        '1h':  BarBuilder(snap_qs['1h'],  bar_qs['1h'],  bar_seconds=3600),
        '4h':  BarBuilder(snap_qs['4h'],  bar_qs['4h'],  bar_seconds=14_400),
    }

    feats = {
        tf: FeatureEngine(snap_qs[tf], bar_qs[tf], feat_qs[tf], ts_qs[tf], horizon_name=tf)
        for tf in snap_qs
    }

    trend_fusion = TrendFusion(ts_qs, q_global)
    profit_engine = ProfitPlanEngine(q_global, feat_qs['1m'], q_profit)

    conf_eng  = ConfidenceEngine(feat_qs['1s'], q_conf)
    phase_det = PhaseDetector(feat_qs['1s'], q_conf, q_phase)
    advisor   = Advisor(q_phase, q_conf, q_adv)

    dash = HFDashboard(
        feat_qs['1s'], q_conf, q_phase, q_adv, q_ingest, ingestor,
        ts_qs, q_global, q_profit,
    )

    await ingestor.start()

    tasks = [asyncio.create_task(fanout.run())]
    tasks += [asyncio.create_task(b.run()) for b in builders.values()]
    tasks += [asyncio.create_task(f.run()) for f in feats.values()]
    tasks += [
        asyncio.create_task(conf_eng.run()),
        asyncio.create_task(phase_det.run()),
        asyncio.create_task(advisor.run()),
        asyncio.create_task(trend_fusion.run()),
        asyncio.create_task(profit_engine.run()),
        asyncio.create_task(dash.run()),
    ]

    await STOP.wait()
    print("\nShutting down …")
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await ingestor.stop()
    print("Bye 👋")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        STOP.set()
        print("\nInterrupted – exiting.")
