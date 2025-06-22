#!/usr/bin/env python3
"""
Real-Time NG Trend Confidence Dashboard
(No database – everything in RAM)

Dependencies
------------
pip install ib-insync rich python-dotenv pandas
"""

import asyncio
import math
import os
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Dict, List, Literal, Optional, Tuple

from dotenv import load_dotenv
from ib_insync import IB, Contract, Ticker
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

load_dotenv()

@dataclass(frozen=True)
class Settings:
    tws_host:     str = os.getenv("TWS_HOST", "127.0.0.1")
    tws_port:     int = int(os.getenv("TWS_PORT", 7497))
    tws_client_id:int = int(os.getenv("TWS_CLIENT_ID", 42))
    depth_levels: int = int(os.getenv("DEPTH_LEVELS", 5))
    roll_window_s:int = int(os.getenv("ROLL_WINDOW_S", 120))
    conf_tau_s:   int = int(os.getenv("CONF_TAU_S", 30))
    refresh_secs: int = int(os.getenv("REFRESH_SECS", 10))

S = Settings()

Side = Literal["bid", "ask"]

# --------------------------------------------------------------------------- #
# Order-book ingestion
# --------------------------------------------------------------------------- #

@dataclass
class L2Level:
    side: Side
    position: int
    price: float
    size: float
    mmid: str
    op: int        # operation code

@dataclass
class L2Snapshot:
    bids: List[L2Level]
    asks: List[L2Level]
    last_price: float
    ts_local: datetime

class L2Ingestor:
    """Consumes L2 book + last trade price and emits Snapshot objects."""

    def __init__(self, contract: Contract, out_q: asyncio.Queue):
        self.ib = IB()
        self.contract = contract
        self.out_q  = out_q
        self.depth  = {"bid": {}, "ask": {}}
        self.last   = 0.0

    async def start(self):
        await self.ib.connectAsync(S.tws_host, S.tws_port, clientId=S.tws_client_id)
        self._reg_handlers()
        self.ib.reqMktDepth(self.contract, numRows=S.depth_levels)
        # periodic snapshot builder
        asyncio.create_task(self._snapshot_loop())

    # ---------- IB event handlers ---------- #
    def _reg_handlers(self):
        self.ib.updateMktDepthEvent += self._on_depth
        self.ib.pendingTickersEvent += self._on_tick

    def _on_depth(self, _:Ticker, pos:int, op:int, side:int, price:float,size:float):
        side_str : Side = "bid" if side==1 else "ask"
        self.depth[side_str][pos] = L2Level(side_str,pos,price,size,"",op)

    def _on_tick(self, tickers):
        for t in tickers:
            if t.last is not None:
                self.last = t.last

    # ---------- assemble & push ---------- #
    async def _snapshot_loop(self):
        while True:
            await asyncio.sleep(0.1)
            if not (self.depth["bid"] and self.depth["ask"]):
                continue
            bids = [self.depth["bid"].get(i, L2Level("bid",i,0,0,"",0))
                    for i in range(S.depth_levels)]
            asks = [self.depth["ask"].get(i, L2Level("ask",i,0,0,"",0))
                    for i in range(S.depth_levels)]
            snap = L2Snapshot(
                bids=bids,
                asks=asks,
                last_price=self.last,
                ts_local=datetime.utcnow().replace(tzinfo=timezone.utc)
            )
            await self.out_q.put(snap)

# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #

@dataclass
class FeatureVector:
    ts: datetime
    book_imb: float
    wall_px: Optional[float]
    wall_sz: Optional[float]
    vwap_delta: float
    mid: float

class FeatureEngine:
    def __init__(self, in_q: asyncio.Queue, out_q: asyncio.Queue):
        self.in_q, self.out_q = in_q, out_q
        self.buf : Deque[L2Snapshot] = deque(maxlen=S.roll_window_s*10)

    async def run(self):
        while True:
            snap:L2Snapshot = await self.in_q.get()
            self.buf.append(snap)
            if len(self.buf) < 2:         # need history
                continue
            fv = self._compute()
            await self.out_q.put(fv)

    def _compute(self)->FeatureVector:
        latest = self.buf[-1]
        bid_qty = sum(l.size for l in latest.bids)
        ask_qty = sum(l.size for l in latest.asks)
        book_imb = bid_qty - ask_qty

        # wall detection
        wall_px = wall_sz = None
        threshold = max(bid_qty,ask_qty)*0.5
        for lvl in (*latest.bids, *latest.asks):
            if lvl.size > threshold:
                wall_px, wall_sz = lvl.price, lvl.size
                break

        # rolling VWAP
        prices = [s.last_price or (s.bids[0].price+s.asks[0].price)/2 for s in self.buf]
        sizes  = [sum(l.size for l in (*s.bids,*s.asks)) for s in self.buf]
        vwap   = sum(p*q for p,q in zip(prices,sizes))/max(1,sum(sizes))
        vwap_delta = (latest.last_price or prices[-1]) - vwap

        mid = (latest.bids[0].price + latest.asks[0].price)/2

        return FeatureVector(latest.ts_local,book_imb,wall_px,wall_sz,vwap_delta,mid)

# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #

@dataclass
class Confidence:
    ts: datetime
    score: float
    trajectory:str

class ConfidenceEngine:
    def __init__(self,in_q:asyncio.Queue,out_q:asyncio.Queue):
        self.in_q,self.out_q=in_q,out_q
        self.score=50.0

    async def run(self):
        while True:
            fv:FeatureVector=await self.in_q.get()
            prev=self.score
            # exponential decay
            self.score = 50 + (self.score-50)*math.exp(-1/S.conf_tau_s)
            # boosts
            self.score += 0.0004*fv.book_imb + 20*math.tanh(fv.vwap_delta/0.02)
            self.score = max(0,min(100,self.score))
            traj = "up" if self.score>prev+1 else "down" if self.score<prev-1 else "flat"
            await self.out_q.put(Confidence(fv.ts,self.score,traj))

# --------------------------------------------------------------------------- #
# Phase + Advice
# --------------------------------------------------------------------------- #

Phase = Literal["TrendingUp","TrendingDown","Consolidating","Reversing","Choppy"]

@dataclass
class PhaseSnapshot:
    ts: datetime
    phase: Phase
    rationale:str

class PhaseDetector:
    def __init__(self,feat_q:asyncio.Queue,conf_q:asyncio.Queue,out_q:asyncio.Queue):
        self.fq,self.cq,self.oq=feat_q,conf_q,out_q
        self.fv:Optional[FeatureVector]=None
        self.conf:Optional[Confidence]=None

    async def run(self):
        while True:
            done,_=await asyncio.wait(
                [self.fq.get(),self.cq.get()],return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                o=t.result()
                if isinstance(o,FeatureVector):self.fv=o
                else:self.conf=o
            if self.fv and self.conf:
                snap=self._classify(self.fv,self.conf)
                await self.oq.put(snap)

    def _classify(self,fv:FeatureVector,conf:Confidence)->PhaseSnapshot:
        if conf.score>65 and fv.vwap_delta>0: ph="TrendingUp"
        elif conf.score>65 and fv.vwap_delta<0: ph="TrendingDown"
        elif conf.score<40 and abs(fv.vwap_delta)<0.01: ph="Consolidating"
        elif conf.trajectory=="down" and conf.score<50: ph="Reversing"
        else: ph="Choppy"
        rat=f"imb={fv.book_imb:.0f}, vwapΔ={fv.vwap_delta:.3f}, conf={conf.score:.1f}"
        return PhaseSnapshot(fv.ts,ph,rat)

@dataclass
class Advice:
    ts: datetime
    text:str

class TraderAssist:
    def __init__(self,phase_q:asyncio.Queue,conf_q:asyncio.Queue,out_q:asyncio.Queue):
        self.pq,self.cq,self.oq=phase_q,conf_q,out_q
        self.phase:Optional[PhaseSnapshot]=None
        self.conf:Optional[Confidence]=None

    async def run(self):
        while True:
            done,_=await asyncio.wait([self.pq.get(),self.cq.get()],
                                      return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                o=t.result()
                if isinstance(o,PhaseSnapshot):self.phase=o
                else:self.conf=o
            if self.phase and self.conf:
                await self.oq.put(self._advise())

    def _advise(self)->Advice:
        ph,sc=self.phase.phase,self.conf.score
        if ph=="TrendingUp" and sc>70: bias="Hold Long"
        elif ph=="TrendingDown" and sc>70: bias="Hold Short"
        elif ph=="Reversing": bias="Exit / Flip"
        elif sc<40: bias="Flat / Wait"
        else: bias="Scalp"
        txt=f"{bias} | {ph} | conf={sc:.0f}% ({self.conf.trajectory})"
        return Advice(self.phase.ts,txt)

# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #

class Dashboard:
    def __init__(self,conf_q,phase_q,adv_q):
        self.conf_q,self.phase_q,self.adv_q=conf_q,phase_q,adv_q
        self.conf:Optional[Confidence]=None
        self.phase:Optional[PhaseSnapshot]=None
        self.adv:Optional[Advice]=None

    async def run(self):
        async with Live(self._render(),auto_refresh=False,refresh_per_second=4) as live:
            while True:
                await self._drain()
                live.update(self._render(),refresh=True)
                await asyncio.sleep(S.refresh_secs)

    async def _drain(self):
        for q in (self.conf_q,self.phase_q,self.adv_q):
            while not q.empty():
                o=q.get_nowait()
                if   isinstance(o,Confidence):     self.conf=o
                elif isinstance(o,PhaseSnapshot):  self.phase=o
                else:                              self.adv=o

    def _render(self)->Panel:
        t=Table(title="NG Trend Dashboard",expand=True)
        t.add_column("Metric"); t.add_column("Value",justify="right")
        t.add_row("UTC",datetime.utcnow().strftime("%H:%M:%S"))
        if self.conf:
            t.add_row("Confidence",f"{self.conf.score:.1f}% {self._bar(self.conf.score)}")
            t.add_row("Trajectory",self.conf.trajectory)
        if self.phase:
            t.add_row("Phase",self.phase.phase)
            t.add_row("Rationale",self.phase.rationale)
        if self.adv:
            t.add_row("Advice",self.adv.text)
        return Panel(t,border_style="cyan")

    @staticmethod
    def _bar(pct:float,l=20)->str:
        filled=int(pct/100*l)
        return "["+"█"*filled+"."*(l-filled)+"]"

# --------------------------------------------------------------------------- #
# Helper – contract builder
# --------------------------------------------------------------------------- #

def front_ng_contract()->Contract:
    c=Contract(symbol="NG",secType="FUT",exchange="NYMEX",
               currency="USD",lastTradeDateOrContractMonth="20250626")
    return c

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

async def main():
    q_snap = asyncio.Queue(maxsize=1000)
    q_feat = asyncio.Queue(maxsize=1000)
    q_conf = asyncio.Queue(maxsize=1000)
    q_phase= asyncio.Queue(maxsize=1000)
    q_adv  = asyncio.Queue(maxsize=1000)

    ingestor = L2Ingestor(front_ng_contract(), q_snap)
    feat_eng = FeatureEngine(q_snap, q_feat)
    conf_eng = ConfidenceEngine(q_feat, q_conf)
    phase_det= PhaseDetector(q_feat, q_conf, q_phase)
    assist   = TraderAssist(q_phase, q_conf, q_adv)
    dash     = Dashboard(q_conf, q_phase, q_adv)

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
