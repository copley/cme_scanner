from __future__ import annotations

from datetime import datetime
from typing import Dict

# ANSI color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def color_probability(p: float, thresholds: Dict[str, float]) -> str:
    if p >= thresholds['continue']:
        return f"{GREEN}{p:.2f}{RESET}"
    if p >= thresholds['watch']:
        return f"{YELLOW}{p:.2f}{RESET}"
    return f"{RED}{p:.2f}{RESET}"


def print_report(results: list[dict], thresholds: Dict[str, float]) -> None:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"{now_str} ─────────────────────────────────")
    print("Symbol   TF     ADX     %>20MA   R2     P(cont)   Signal")
    for row in results:
        prob_str = color_probability(row['probability'], thresholds)
        print(
            f"{row['symbol']:<7} {row['timeframe']:<5} "
            f"{row['ADX']:<7.2f} {row['%Above20MA']:<8.2f} {row['R2']:<6.2f} {prob_str:<10} "
            f"{row['signal']}"
        )
