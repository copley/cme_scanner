#!/usr/bin/env python3
"""
NG Hedge‑Fund Dashboard
------------------------------------------------------------
• L2 market‑depth + full tick stream (ib_insync)
• 1‑second OHLC bar builder
• Micro‑structure metrics  (imbalance, wall, VWAPΔ)
• Technical indicators     (ADX/ADXR, Vortex, BB‑Width, ATR%,
                             RSI, MACD, DonchianW, Choppiness,
                             OBV, CMF)
• Confidence / Phase / Advice engine
• 3‑panel rich Layout dashboard: Edge Forecast, System Health,
  Trend Confidence
------------------------------------------------------------
Dependencies: ib‑insync, rich, python‑dotenv, pandas, numpy,
              psutil
"""

from __future__ import annotations

import asyncio
import math
import os
import signal
import sys
import tracemalloc
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, List, Literal, Optional

import numpy as np
import pandas as pd
import psutil
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


def _install_signal_handlers():
    """Convert SIGINT / SIGTERM into STOP.set() without breaking Windows."""
    loop = asyncio.get_running_loop()

    def _handler():
        STOP.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handler)
        except (NotImplementedError, RuntimeError):
            # Windows / uvloop fallback
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
    """
    Grabs depth + last trade price → emits Snapshot every 100 ms.
    Provides .connected property for health pane and clean stop().
    """

    def __init__(self, c: Contract, q_out: asyncio.Queue):
        self.ib = IB()
        self.contract = c
        self.q_out = q_out
        self.depth = {"bid": {}, "ask": {}}
        self.last = 0.0
        self._depth_sub: Optional[int] = None
        self._mkt_sub = None
        self._tbt_sub = None

    # ---------- connection helpers ---------- #
    @property
    def connected(self) -> bool:
        return self.ib.isConnected()

    async def start(self):
        await self.ib.connectAsync(S.tws_host, S.tws_port, clientId=S.tws_cid)

        # Depth
        self.ib.updateMktDepthEvent += self._on_depth
        self._depth_sub = self.ib.reqMktDepth(
            self.contract, numRows=S.depth_rows
        ).reqId

        # Last‑price streams
        self._mkt_sub = self.ib.reqMktData(self.contract, "", snapshot=False)
        self._mkt_sub.updateEvent += self._on_ticker

        self._tbt_sub = self.ib.reqTickByTickData(
            self.contract, "Last", 0, False
        )
        self._tbt_sub.updateEvent += self._on_tbtick

        asyncio.create_task(self._emit_loop())

    async def stop(self):
        if self._depth_sub is not None:
            self.ib.cancelMktDepth(self._depth_sub)
        if self._mkt_sub is not None:
            self.ib.cancelMktData(self._mkt_sub.tickerId)
        if self._tbt_sub is not None:
            self.ib.cancelTickByTickData(self._tbt_sub.reqId)
        if self.connected:
            await self.ib.disconnectAsync()

    # ---------- IB callbacks ---------- #
    # depth signature future‑proof: accepts isSmartDepth (new in API v10.25)
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
    async def _emit_loop(self):
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
                bids,
                asks,
                self.last,
                datetime.utcnow().replace(tzinfo=timezone.utc),
            )
            try:
                self.q_out.put_nowait(snap)
            except asyncio.QueueFull:
                # drop if downstream can’t keep up – we do not block
                pass


# --------------------------------------------------------------------------- #
#  Broadcast hub (back‑pressure safe)
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
                    # slow consumer – skip this tick for that queue
                    pass


# --------------------------------------------------------------------------- #
#  Bar builder + indicator helpers
# --------------------------------------------------------------------------- #


@dataclass
class Bar:
    ts: datetime
    high: float
    low: float
    close: float


class BarBuilder:
    """Aggregates tick snapshots into strictly UTC‑aware 1‑s OHLC bars."""

    def __init__(self, q_in: asyncio.Queue, q_out: asyncio.Queue):
        self.q_in, self.q_out = q_in, q_out
        self.curr: Optional[Bar] = None
        self.window: Deque[Bar] = deque(maxlen=S.roll_secs)

    async def run(self):
        while not STOP.is_set():
            snap: Snapshot = await self.q_in.get()
            ts = snap.ts.astimezone(timezone.utc).replace(microsecond=0)
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

    # --------------- window access helper --------------- #
    def prices(self):
        return [b.close for b in self.window]


# ---------- indicator math ---------- #
def _ema(arr, span):
    return pd.Series(arr).ewm(span=span, adjust=False).mean().iloc[-1]


def adx(high, low, close, period=14):
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
    plus_di = (
        100
        * pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean()
        / atr
    )
    minus_di = (
        100
        * pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean()
        / atr
    )
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_val = float(
        dx.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    )
    adxr = (
        float((adx_val + dx.iloc[-period]) / 2)
        if len(dx) >= period + 1
        else None
    )
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
    return 100 * math.log10(atr_sum / (high_max - low_min)) / math.log10(
        period
    )


# OBV & CMF need volume – we approximate using book imbalance delta as proxy
def obv_proxy(close, book_imb_arr):
    if len(close) < 2:
        return None
    direction = np.sign(np.diff(close))
    return float(np.sum(direction * np.array(book_imb_arr[-(len(direction)) :])))


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
    # --- indicator core ---
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
    # --- lifecycle ---
    trend_age: int
    exhaustion: float | None
    # --- derived ---
    edge_score: float


class FeatureEngine:
    def __init__(
        self,
        snap_q: asyncio.Queue,
        bar_q: asyncio.Queue,
        out_q: asyncio.Queue,
    ):
        self.snap_q, self.bar_q, self.out_q = snap_q, bar_q, out_q
        self.bars: Deque[Bar] = deque(maxlen=S.roll_secs)
        self.mid_prices: Deque[float] = deque(maxlen=300)
        self.book_hist: Deque[float] = deque(maxlen=300)
        self.trend_age = 0
        self.prev_conf: Optional[float] = None

    async def run(self):
        asyncio.create_task(self._bar_listener())
        while not STOP.is_set():
            snap: Snapshot = await self.snap_q.get()
            mid = (snap.bids[0].price + snap.asks[0].price) / 2
            self.mid_prices.append(mid)
            book_imb = sum(b.size for b in snap.bids) - sum(
                a.size for a in snap.asks
            )
            self.book_hist.append(book_imb)
            wall_sz = max(
                (lvl.size for lvl in (*snap.bids, *snap.asks)), default=0
            )
            vwap = sum(self.mid_prices) / len(self.mid_prices)
            vwap_delta = mid - vwap

            adx_val = adxr = v_ip = v_in = bbw = atrp = None
            rs = mcd = mcd_sig = donw = chp = obv = cmf = None
            if len(self.bars) >= 26:
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

            # --- trend age / exhaustion rough heuristics ---
            if adx_val and adx_val > 25:
                self.trend_age += 1
            else:
                self.trend_age = 0
            exhaustion = (
                None
                if self.trend_age < 14 or not adx_val
                else max(0, 50 - adx_val)
            )

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
            )
            await self.out_q.put(fv)

    async def _bar_listener(self):
        while not STOP.is_set():
            bar: Bar = await self.bar_q.get()
            self.bars.append(bar)


# --------------------------------------------------------------------------- #
#  Confidence / Phase / Advice
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
            self.score += (
                0.0004 * fv.book_imb + 25 * math.tanh(fv.vwap_delta / 0.02)
            )
            if fv.adx:
                self.score += (fv.adx - 25) * 0.4
            self.score = max(0, min(100, self.score))
            traj = (
                "up"
                if self.score > prev + 1
                else "down"
                if self.score < prev - 1
                else "flat"
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

        # ---------- phase logic ----------
        if fv.atr_pct and fv.atr_pct > 3.0:
            phase: Phase = "Volatile"
        elif fv.bb_width and fv.bb_width < 4 and 40 <= cf.score <= 60:
            phase = "Ranging"
        elif fv.don_w and fv.don_w > 8 and cf.score > 65:
            phase = "Expanding"
        elif fv.bb_width and fv.bb_width > 8 and cf.score > 65:
            phase = "Breakout"
        elif (
            self.prev_phase == "Breakout"
            and cf.traj == "down"
            and cf.score < 60
        ):
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

        # profit‑taking flag
        if (
            phase in ("TrendingUp", "TrendingDown")
            and fv.exhaustion
            and fv.exhaustion > 10
        ):
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
        )
        return PhaseSnap(fv.ts, phase, rat or "—")


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
                o = t.result()
                if isinstance(o, PhaseSnap):
                    self.ph = o
                else:
                    self.cf = o
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
#  Hedge‑Fund Dashboard
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
    ):
        self.fq, self.cq, self.pq, self.aq, self.iq = (
            feat_q,
            conf_q,
            phase_q,
            adv_q,
            ingest_q,
        )
        self.ingestor = ingestor
        self.fv: FeatureVector | None = None
        self.cf: Confidence | None = None
        self.ph: PhaseSnap | None = None
        self.ad: Advice | None = None
        self.last = datetime.utcnow()

    async def run(self):
        lay = self._layout()
        async with Live(lay, screen=False, refresh_per_second=4) as live:
            while not STOP.is_set():
                self._drain()
                lay["edge"].update(self._edge_panel())
                lay["health"].update(self._health_panel())
                lay["trend"].update(self._trend_panel())
                live.refresh()
                await asyncio.sleep(S.refresh)

    def _layout(self):
        root = Layout()
        root.split_row(Layout(name="edge", ratio=3), Layout(name="health", ratio=2))
        root.split(
            Layout(name="spacer", size=1), Layout(name="trend"), direction="vertical"
        )
        return root

    # ---------- panels ---------- #
    def _edge_panel(self) -> Panel:
        if not self.fv:
            return Panel(
                "Waiting …",
                title="⚡ Edge Forecast",
                border_style="grey50",
                box=box.ROUNDED,
            )
        f = self.fv
        T = Table.grid(expand=True)
        T.add_column()
        T.add_column(justify="right")
        T.add_row("🕒 Time", f.ts.strftime("%H:%M:%S"))
        T.add_row("📈 Mid", f"{f.mid:.3f}")
        T.add_row("🧮 VWAP Δ", f"{f.vwap_delta:+.4f}")
        T.add_row("🧲 Imbalance", f"{f.book_imb:+.0f}")
        T.add_row("🧱 Wall Size", f"{f.wall_sz or 0:.0f}")
        if f.adx:
            T.add_row("⚡ ADX", f"{f.adx:.1f}")
        if f.bb_width:
            T.add_row("📏 BB‑Width", f"{f.bb_width:.1f}%")
        if f.rsi:
            T.add_row("🌀 RSI", f"{f.rsi:.1f}")
        if f.don_w:
            T.add_row("📣 DonW", f"{f.don_w:.1f}%")
        T.add_row("💡 Edge", self._edge_str(f.edge_score))
        return Panel(T, title="⚡ NG Edge Forecast", border_style="white", box=box.ROUNDED)

    def _health_panel(self) -> Panel:
        lag = (datetime.utcnow() - self.last).total_seconds()
        backlog = self.iq.qsize()
        mem = psutil.Process().memory_info().rss / (1024 * 1024)
        conn = "[green]LIVE[/]" if self.ingestor.connected else "[red]DOWN[/]"
        T = Table.grid(expand=True)
        T.add_column()
        T.add_column(justify="right")
        T.add_row("IB Conn", conn)
        T.add_row("Lag", f"{lag:.1f}s")
        T.add_row("Backlog", str(backlog))
        T.add_row("RSS", f"{mem:.1f} MB")
        T.add_row("Status", "OK" if lag < S.refresh * 2 and conn else "[red]STALE[/]")
        return Panel(T, title="🛠 System Health", border_style="cyan", box=box.ROUNDED)

    def _trend_panel(self) -> Panel:
        if not (self.cf and self.ph and self.ad):
            return Panel("…", title="Trend", border_style="grey50", box=box.ROUNDED)
        bar = self._bar(self.cf.score)
        T = Table.grid(expand=True)
        T.add_column()
        T.add_column(justify="right")
        T.add_row("Confidence", f"{self.cf.score:.1f}% {bar} ({self.cf.traj})")
        T.add_row("Phase", self.ph.phase)
        T.add_row("Rationale", self.ph.rationale or "—")
        T.add_row("Advice", self.ad.text)
        color_map = {
            "TrendingUp": "green",
            "TrendingDown": "red",
            "Volatile": "magenta",
            "Breakout": "bright_green",
            "BreakoutFailed": "bright_red",
            "ProfitTaking": "yellow",
        }
        color = color_map.get(self.ph.phase, "yellow")
        return Panel(T, title="📊 Trend Confidence", border_style=color, box=box.ROUNDED)

    # ---------- helpers ---------- #
    def _drain(self):
        for q in (self.fq, self.cq, self.pq, self.aq):
            while not q.empty():
                o = q.get_nowait()
                if isinstance(o, FeatureVector):
                    self.fv = o
                elif isinstance(o, Confidence):
                    self.cf = o
                elif isinstance(o, PhaseSnap):
                    self.ph = o
                else:
                    self.ad = o
                self.last = datetime.utcnow()

    @staticmethod
    def _bar(pct, l=14):
        f = int(pct / 100 * l)
        return "[" + "█" * f + "." * (l - f) + "]"

    @staticmethod
    def _edge_str(score):
        color = "green" if score > 85 else "yellow" if score > 60 else "white"
        tag = " ⚑HIGH" if score > 85 else ""
        return f"[{color}]{score:5.2f}[/{color}]{tag}"


# --------------------------------------------------------------------------- #
#  Contract helper
# --------------------------------------------------------------------------- #


def front_ng_contract():
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

    q_ingest = asyncio.Queue(2000)
    q_snap_feat = asyncio.Queue(2000)
    q_snap_bar = asyncio.Queue(2000)
    q_bar = asyncio.Queue(2000)
    q_feat = asyncio.Queue(2000)
    q_conf = asyncio.Queue(2000)
    q_phase = asyncio.Queue(2000)
    q_adv = asyncio.Queue(2000)

    ingestor = L2Ingestor(front_ng_contract(), q_ingest)
    fanout = SnapshotBroadcaster(q_ingest, q_snap_feat, q_snap_bar)
    bar_build = BarBuilder(q_snap_bar, q_bar)
    feat_eng = FeatureEngine(q_snap_feat, q_bar, q_feat)
    conf_eng = ConfidenceEngine(q_feat, q_conf)
    phase_det = PhaseDetector(q_feat, q_conf, q_phase)
    advisor = Advisor(q_phase, q_conf, q_adv)
    dash = HFDashboard(q_feat, q_conf, q_phase, q_adv, q_ingest, ingestor)

    await ingestor.start()

    tasks = [
        asyncio.create_task(fn.run())
        for fn in (
            fanout,
            bar_build,
            feat_eng,
            conf_eng,
            phase_det,
            advisor,
            dash,
        )
    ]

    # ---------- wait for STOP ---------- #
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
        # Fallback path if signals failed to set STOP
        STOP.set()
        print("\nInterrupted – exiting.")
