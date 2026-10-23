"""
screener.py – FinPilot AI QuantScreener Agent
=============================================
Responsibilities
----------------
1. Receive a `ScreeningRequest` from the Orchestrator.
2. Fetch fundamental and price data for the candidate universe via yfinance.
3. Apply deterministic quantitative filters (P/E cap, market cap floor, etc.).
4. Score and rank the survivors.
5. Send a `RankedShortlistResponse` back to the Orchestrator.

Run this agent FIRST (before the orchestrator) in its own terminal:
    python screener.py

Copy the printed address and paste it into orchestrator.py's
SCREENER_AGENT_ADDRESS (or set SCREENER_AGENT_ADDRESS in .env).
"""

from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from uagents import Agent, Context

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from schemas import (
    RankedShortlistResponse,
    ScreenedStock,
    ScreeningRequest,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Environment & logging
# ═══════════════════════════════════════════════════════════════════════════════

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("finpilot.screener")

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

SCREENER_SEED: str = os.getenv(
    "SCREENER_SEED",
    "screener_default_seed_finpilot_ai_v1_replace_in_prod",
)
SCREENER_PORT: int = int(os.getenv("SCREENER_PORT", "8002"))

SCREENER_VERSION: str = "1.0.0"
INR_CRORE: float = 1e7  # 1 Crore = 10,000,000

# ── Phase 1 candidate universe (5 blue-chip NSE stocks) ───────────────────────
# Phase 2 will replace this with the full Nifty 50 list read from nifty50.csv.
CANDIDATE_UNIVERSE: List[str] = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ITC.NS",
]

# ── Filter thresholds (all configurable via env vars) ─────────────────────────
MAX_PE_RATIO: float = float(os.getenv("MAX_PE_RATIO", "50.0"))
MIN_MARKET_CAP_CR: float = float(os.getenv("MIN_MARKET_CAP_CR", "10000.0"))  # ₹10k Cr
MAX_PCT_FROM_52W_HIGH: float = float(
    os.getenv("MAX_PCT_FROM_52W_HIGH", "0.0")
)  # 0 = no restriction; negative values mean stock must be below its 52w high

# ═══════════════════════════════════════════════════════════════════════════════
# Data layer – raw fetching from yfinance
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RawStockData:
    """
    Intermediate container for data fetched from yfinance before filtering.
    Using a dataclass (not a Pydantic model) because this never crosses the
    agent wire – it lives only in-process.
    """

    ticker: str
    company_name: str
    current_price: float
    market_cap: float          # raw, in INR (not Crores)
    pe_ratio: Optional[float]
    week_52_high: float
    week_52_low: float
    fetch_ok: bool = True
    fetch_error: str = ""


def _safe_float(value: object, fallback: float = math.nan) -> float:
    """Cast yfinance info values to float, returning fallback on failure."""
    try:
        if value is None:
            return fallback
        f = float(value)
        return f if math.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


def _fetch_stock_data(ticker: str) -> RawStockData:
    """
    Fetch fundamental + price data for a single NSE ticker.

    Uses yfinance Ticker.info dict.  Key fields extracted:
      - shortName / longName : company name
      - currentPrice / regularMarketPrice / previousClose : CMP
      - marketCap : market capitalisation in INR
      - trailingPE : trailing 12-month P/E ratio
      - fiftyTwoWeekHigh / fiftyTwoWeekLow : 52-week price range

    Returns a RawStockData; sets fetch_ok=False on any error.
    """
    logger.info("Fetching data for %s…", ticker)
    try:
        tkr = yf.Ticker(ticker)
        info: Dict = tkr.info  # network call

        # ── Company name ──────────────────────────────────────────────────────
        company_name: str = (
            info.get("shortName")
            or info.get("longName")
            or ticker.replace(".NS", "").replace(".BO", "")
        )

        # ── Current price ─────────────────────────────────────────────────────
        # Prefer currentPrice; fall back through a chain of alternatives.
        price_keys: Tuple[str, ...] = (
            "currentPrice",
            "regularMarketPrice",
            "previousClose",
            "ask",
        )
        current_price: float = math.nan
        for key in price_keys:
            val = _safe_float(info.get(key))
            if not math.isnan(val) and val > 0:
                current_price = val
                break

        if math.isnan(current_price):
            # Last resort: download 5-day OHLCV
            hist: pd.DataFrame = yf.download(
                ticker, period="5d", interval="1d", progress=False, auto_adjust=True
            )
            if not hist.empty:
                current_price = float(hist["Close"].dropna().iloc[-1])

        # ── Market cap ────────────────────────────────────────────────────────
        market_cap: float = _safe_float(info.get("marketCap"), math.nan)

        # ── Trailing P/E ──────────────────────────────────────────────────────
        trailing_pe_raw: float = _safe_float(info.get("trailingPE"), math.nan)
        pe_ratio: Optional[float] = (
            None if math.isnan(trailing_pe_raw) else trailing_pe_raw
        )

        # ── 52-week range ─────────────────────────────────────────────────────
        week_52_high: float = _safe_float(info.get("fiftyTwoWeekHigh"), math.nan)
        week_52_low: float = _safe_float(info.get("fiftyTwoWeekLow"), math.nan)

        # ── Validate that we at minimum have price + market cap ───────────────
        missing: List[str] = []
        if math.isnan(current_price):
            missing.append("currentPrice")
        if math.isnan(market_cap):
            missing.append("marketCap")
        if math.isnan(week_52_high):
            missing.append("fiftyTwoWeekHigh")

        if missing:
            logger.warning(
                "%s: missing critical fields %s – will be dropped", ticker, missing
            )
            return RawStockData(
                ticker=ticker,
                company_name=company_name,
                current_price=0.0,
                market_cap=0.0,
                pe_ratio=None,
                week_52_high=0.0,
                week_52_low=0.0,
                fetch_ok=False,
                fetch_error=f"Missing fields: {missing}",
            )

        logger.info(
            "  %s → CMP=₹%.2f  MCap=₹%.0fCr  P/E=%s  52wH=₹%.2f",
            ticker,
            current_price,
            market_cap / INR_CRORE,
            f"{pe_ratio:.2f}" if pe_ratio is not None else "N/A",
            week_52_high,
        )

        return RawStockData(
            ticker=ticker,
            company_name=company_name,
            current_price=current_price,
            market_cap=market_cap,
            pe_ratio=pe_ratio,
            week_52_high=week_52_high,
            week_52_low=week_52_low,
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("Unhandled error fetching %s: %s", ticker, exc, exc_info=True)
        return RawStockData(
            ticker=ticker,
            company_name=ticker,
            current_price=0.0,
            market_cap=0.0,
            pe_ratio=None,
            week_52_high=0.0,
            week_52_low=0.0,
            fetch_ok=False,
            fetch_error=str(exc),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Screening engine – deterministic, pure functions
# ═══════════════════════════════════════════════════════════════════════════════


def _pct_from_52w_high(current_price: float, week_52_high: float) -> float:
    """
    Return the percentage deviation of the current price from the 52-week high.

    Negative values indicate the stock is trading below its 52-week high.
    E.g.: current=900, 52wH=1000 → -10.0  (10% below peak)
          current=1050, 52wH=1000 → +5.0   (new high territory)
    """
    if week_52_high <= 0:
        return math.nan
    return ((current_price - week_52_high) / week_52_high) * 100.0


def _apply_filters(
    candidates: List[RawStockData],
    max_pe: float,
    min_market_cap_cr: float,
) -> Tuple[List[RawStockData], str]:
    """
    Apply deterministic quantitative filters to the raw candidate list.

    Filters applied (in order):
      F1 – Exclude stocks with fetch errors.
      F2 – Exclude stocks with P/E > max_pe (stocks with None P/E are kept with
           a warning; you can tighten this in production).
      F3 – Exclude stocks with market cap < min_market_cap_cr Crores.

    Returns (survivors, human-readable filter summary string).
    """
    total_in: int = len(candidates)
    summary_parts: List[str] = [
        f"Universe: {total_in} stocks. "
        f"Filters: P/E ≤ {max_pe:.0f}, MCap ≥ ₹{min_market_cap_cr:,.0f}Cr."
    ]

    # F1 – Fetch failures
    ok_data: List[RawStockData] = [c for c in candidates if c.fetch_ok]
    dropped_f1: int = total_in - len(ok_data)
    if dropped_f1:
        summary_parts.append(f"F1 (fetch error): dropped {dropped_f1}.")

    # F2 – P/E filter
    after_pe: List[RawStockData] = []
    dropped_pe_names: List[str] = []
    for c in ok_data:
        if c.pe_ratio is None:
            # No P/E data – keep but flag
            logger.warning("%s has no P/E data; keeping with caveat.", c.ticker)
            after_pe.append(c)
        elif c.pe_ratio > max_pe:
            dropped_pe_names.append(f"{c.ticker}(P/E={c.pe_ratio:.1f})")
        else:
            after_pe.append(c)

    if dropped_pe_names:
        summary_parts.append(
            f"F2 (P/E>{max_pe:.0f}): dropped {', '.join(dropped_pe_names)}."
        )
    else:
        summary_parts.append(f"F2 (P/E>{max_pe:.0f}): all passed.")

    # F3 – Market cap filter
    min_market_cap_inr: float = min_market_cap_cr * INR_CRORE
    after_mcap: List[RawStockData] = []
    dropped_mcap_names: List[str] = []
    for c in after_pe:
        if c.market_cap < min_market_cap_inr:
            dropped_mcap_names.append(
                f"{c.ticker}(MCap=₹{c.market_cap/INR_CRORE:,.0f}Cr)"
            )
        else:
            after_mcap.append(c)

    if dropped_mcap_names:
        summary_parts.append(
            f"F3 (MCap<₹{min_market_cap_cr:,.0f}Cr): dropped {', '.join(dropped_mcap_names)}."
        )
    else:
        summary_parts.append(f"F3 (MCap<₹{min_market_cap_cr:,.0f}Cr): all passed.")

    summary_parts.append(f"Survivors: {len(after_mcap)} of {total_in}.")
    summary: str = "  ".join(summary_parts)
    return after_mcap, summary


def _score_and_rank(survivors: List[RawStockData]) -> List[RawStockData]:
    """
    Rank survivors using a simple composite score.

    Scoring dimensions (all normalised 0–1 within the candidate set):
      - Lower P/E is better          : weight 40%
      - Larger market cap is better  : weight 30%
      - Closer to 52-week high is better (momentum): weight 30%

    Stocks with no P/E are scored 0 on that dimension.
    """
    if not survivors:
        return survivors

    # Collect raw values
    pe_values: List[float] = [
        s.pe_ratio if s.pe_ratio is not None else MAX_PE_RATIO
        for s in survivors
    ]
    mcap_values: List[float] = [s.market_cap for s in survivors]
    pct_high_values: List[float] = [
        _pct_from_52w_high(s.current_price, s.week_52_high) for s in survivors
    ]

    def _min_max_norm(values: List[float], invert: bool = False) -> List[float]:
        """Normalise a list of floats to [0, 1]; invert=True flips the scale."""
        min_v: float = min(values)
        max_v: float = max(values)
        rng: float = max_v - min_v
        if rng == 0.0:
            return [0.5] * len(values)
        normed: List[float] = [(v - min_v) / rng for v in values]
        if invert:
            normed = [1.0 - n for n in normed]
        return normed

    # Lower P/E → higher score (invert=True)
    pe_scores: List[float] = _min_max_norm(pe_values, invert=True)
    # Higher market cap → higher score
    mcap_scores: List[float] = _min_max_norm(mcap_values, invert=False)
    # Closer to 52w high (less negative) → higher score
    pct_high_scores: List[float] = _min_max_norm(pct_high_values, invert=False)

    scored: List[Tuple[float, RawStockData]] = []
    for i, stock in enumerate(survivors):
        composite: float = (
            0.40 * pe_scores[i]
            + 0.30 * mcap_scores[i]
            + 0.30 * pct_high_scores[i]
        )
        scored.append((composite, stock))
        logger.debug(
            "  %s → composite=%.4f (pe_score=%.4f, mcap_score=%.4f, pct_high_score=%.4f)",
            stock.ticker,
            composite,
            pe_scores[i],
            mcap_scores[i],
            pct_high_scores[i],
        )

    # Sort descending by composite score
    scored.sort(key=lambda t: t[0], reverse=True)
    return [s for _, s in scored]


def _to_screened_stock(raw: RawStockData) -> ScreenedStock:
    """Convert a RawStockData into the wire-ready ScreenedStock schema."""
    pct_high: float = _pct_from_52w_high(raw.current_price, raw.week_52_high)
    return ScreenedStock(
        ticker=raw.ticker,
        company_name=raw.company_name,
        market_cap_cr=round(raw.market_cap / INR_CRORE, 2),
        pe_ratio=round(raw.pe_ratio, 2) if raw.pe_ratio is not None else None,
        week_52_high=round(raw.week_52_high, 2),
        current_price=round(raw.current_price, 2),
        pct_from_52w_high=round(pct_high, 4),
    )


def run_screening_pipeline(
    universe: List[str],
    max_pe: float = MAX_PE_RATIO,
    min_market_cap_cr: float = MIN_MARKET_CAP_CR,
) -> Tuple[List[ScreenedStock], str]:
    """
    Full, deterministic screening pipeline.

    Parameters
    ----------
    universe:         List of NSE tickers (e.g. ["RELIANCE.NS", "TCS.NS"]).
    max_pe:           Maximum allowed trailing P/E ratio.
    min_market_cap_cr: Minimum market cap in Indian Crores.

    Returns
    -------
    (ranked_stocks, filter_summary)
    """
    logger.info(
        "Starting screening pipeline on %d candidates (max_pe=%.0f, min_mcap=₹%.0fCr)",
        len(universe),
        max_pe,
        min_market_cap_cr,
    )

    # ── Fetch data for all candidates (sequential to avoid yfinance rate limits)
    raw_data: List[RawStockData] = [_fetch_stock_data(ticker) for ticker in universe]

    # ── Apply filters
    survivors, summary = _apply_filters(raw_data, max_pe, min_market_cap_cr)

    # ── Score and rank
    ranked: List[RawStockData] = _score_and_rank(survivors)

    # ── Convert to wire schemas
    screened_stocks: List[ScreenedStock] = [_to_screened_stock(s) for s in ranked]

    logger.info("Screening complete. %s", summary)
    return screened_stocks, summary


# ═══════════════════════════════════════════════════════════════════════════════
# Agent initialisation
# ═══════════════════════════════════════════════════════════════════════════════

screener = Agent(
    name="QuantScreener",
    seed=SCREENER_SEED,
    port=SCREENER_PORT,
    endpoint=[f"http://localhost:{SCREENER_PORT}/submit"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# Fetch.ai event handlers
# ═══════════════════════════════════════════════════════════════════════════════


@screener.on_event("startup")
async def handle_startup(ctx: Context) -> None:
    ctx.logger.info("QuantScreener Agent started.")
    ctx.logger.info("  Address  : %s", ctx.address)
    ctx.logger.info("  Port     : %d", SCREENER_PORT)
    ctx.logger.info("  Universe : %s", CANDIDATE_UNIVERSE)
    ctx.logger.info(
        "  Filters  : max_pe=%.0f | min_mcap=₹%.0fCr",
        MAX_PE_RATIO,
        MIN_MARKET_CAP_CR,
    )
    ctx.logger.info(
        ">>> Copy the address above and set it as SCREENER_AGENT_ADDRESS "
        "in orchestrator.py (or .env) <<<"
    )


@screener.on_message(model=ScreeningRequest)
async def handle_screening_request(
    ctx: Context, sender: str, msg: ScreeningRequest
) -> None:
    """
    Receive a ScreeningRequest, run the full pipeline, reply to the Orchestrator.
    """
    ctx.logger.info(
        "Received ScreeningRequest from sender=%s  (request_id=%s)",
        sender,
        msg.request_id,
    )

    # Run the deterministic screening pipeline.
    screened_stocks, filter_summary = run_screening_pipeline(
        universe=CANDIDATE_UNIVERSE,
        max_pe=MAX_PE_RATIO,
        min_market_cap_cr=MIN_MARKET_CAP_CR,
    )

    response = RankedShortlistResponse(
        request_id=msg.request_id,
        screened_stocks=screened_stocks,
        filter_summary=filter_summary,
        portfolio_snapshot=None,  # Orchestrator holds the snapshot
        screener_version=SCREENER_VERSION,
    )

    # Reply to the Orchestrator using the address it embedded in the request.
    reply_to: str = msg.orchestrator_address
    ctx.logger.info(
        "Sending RankedShortlistResponse (%d stocks) to Orchestrator at %s",
        len(screened_stocks),
        reply_to,
    )
    await ctx.send(reply_to, response)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    screener.run()
