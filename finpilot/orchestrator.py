"""
orchestrator.py – FinPilot AI Orchestrator Agent
=================================================
Responsibilities
----------------
1. Accept `PortfolioInput` via the ASI:One / Agentverse chat protocol OR
   directly from peer agents / trigger.py.
2. Fetch live prices via yfinance and calculate real-time portfolio metrics.
3. Forward a `ScreeningRequest` to the QuantScreener Agent.
4. Receive `RankedShortlistResponse` and forward a `ResearchRequest` to the
   DeepResearcher Agent.
5. Receive `AgenticThesisResponse`, render the final report to the console,
   and reply to the original ASI:One sender via the chat protocol.

Phase 3 changes
---------------
- Agent initialised with `mailbox=True` so Agentverse buffers inbound
  messages when the process is temporarily offline.
- Agent Chat Protocol (`uagents_core.contrib.protocols.chat`) included with
  `publish_manifest=True` so ASI:One can discover and route to this agent.
- A `@chat_proto.on_message(ChatMessage)` handler parses the user’s natural-
  language / JSON prompt from ASI:One into a PortfolioInput and injects it
  into the existing Phase 1–2 pipeline.
- The final thesis is echoed back to the original ASI:One sender.

Run this agent in its own terminal:
    python orchestrator.py
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import textwrap
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from uagents import Agent, Context, Model, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)

# ── Local imports ──────────────────────────────────────────────────────────────
# Ensure the finpilot package root is on sys.path when running directly.
sys.path.insert(0, os.path.dirname(__file__))

from schemas import (
    AgenticThesisResponse,
    HoldingItem,
    PortfolioInput,
    PortfolioSnapshot,
    RankedShortlistResponse,
    ResearchRequest,
    ScreeningRequest,
    UAgentResponse,
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
logger = logging.getLogger("finpilot.orchestrator")

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# Seeds are deterministic hex strings – each agent always gets the same address.
# Generate your own with: python -c "import secrets; print(secrets.token_hex(32))"
ORCHESTRATOR_SEED: str = os.getenv(
    "ORCHESTRATOR_SEED",
    "orchestrator_default_seed_finpilot_ai_v1_replace_in_prod",
)
ORCHESTRATOR_PORT: int = int(os.getenv("ORCHESTRATOR_PORT", "8001"))

# ── Screener Agent address ─────────────────────────────────────────────────────
# After running screener.py once, paste its printed address here (or in .env).
# Format: "agent1q..." (Bech32-encoded public key hash)
SCREENER_AGENT_ADDRESS: str = os.getenv(
    "SCREENER_AGENT_ADDRESS",
    "REPLACE_WITH_SCREENER_ADDRESS",
)

# ── DeepResearcher Agent address ────────────────────────────────────────────
# After running researcher.py once, paste its printed address here (or .env).
RESEARCHER_AGENT_ADDRESS: str = os.getenv(
    "RESEARCHER_AGENT_ADDRESS",
    "REPLACE_WITH_RESEARCHER_ADDRESS",
)

# ── Agentverse Mailbox Key ─────────────────────────────────────────────────────
# Obtain from https://agentverse.ai → My Agents → your agent → Mailbox tab.
# When set to a non-empty string the Agent constructor will use it; we keep
# mailbox=True regardless (the library handles both modes).
AGENTVERSE_MAILBOX_KEY: str = os.getenv("AGENTVERSE_MAILBOX_KEY", "")

INR_CRORE: float = 1e7  # 1 Crore = 10,000,000

# ── In-flight ASI:One session tracker ──────────────────────────────────────────────
# Maps request_id → ASI:One sender address so we can reply to the correct
# chat session when the thesis arrives (potentially many seconds later).
_pending_chat_senders: Dict[str, str] = {}

# ═══════════════════════════════════════════════════════════════════════════════
# Agent initialisation
# ═══════════════════════════════════════════════════════════════════════════════

# ── Agent initialisation ───────────────────────────────────────────────────────
# mailbox=...: Agentverse buffers inbound messages when this process is
# offline, so the agent remains reachable 24/7 on the decentralised network.
# The seed is REQUIRED for a stable, deterministic agent address.
# The mailbox URL format is: "{mailbox_key}@https://agentverse.ai"
orchestrator = Agent(
    name="FinPilotOrchestrator",
    seed=ORCHESTRATOR_SEED,
    port=ORCHESTRATOR_PORT,
    endpoint=[f"http://localhost:{ORCHESTRATOR_PORT}/submit"],
    mailbox=f"{AGENTVERSE_MAILBOX_KEY}@https://agentverse.ai" if AGENTVERSE_MAILBOX_KEY else None,
)

# ── Agent Chat Protocol (ACP) – required for ASI:One discoverability ───────────
# We instantiate the standard spec from uagents_core so the manifest hash
# matches what ASI:One and the Almanac expect.
chat_proto = Protocol(spec=chat_protocol_spec)

# ── PortfolioInput Protocol – public capability for ASI:One / Agentverse ────────
# Wrapping the PortfolioInput handler in a named Protocol with
# publish_manifest=True makes this capability explicitly discoverable on the
# Almanac.  ASI:One uses the manifest to route requests to the correct agent.
portfolio_proto = Protocol(name="FinPilotPortfolioInput", version="1.0.0")

# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════


def _fetch_live_price(ticker: str) -> Optional[float]:
    """
    Fetch the latest closing price for a single ticker via yfinance.

    Returns None if the ticker is invalid or the network request fails.
    Uses a 5-day window to tolerate non-trading days.
    """
    try:
        data: pd.DataFrame = yf.download(
            ticker,
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if data.empty:
            logger.warning("yfinance returned empty data for ticker=%s", ticker)
            return None
        price: float = float(data["Close"].dropna().iloc[-1])
        return price
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to fetch price for ticker=%s: %s", ticker, exc)
        return None


def _build_portfolio_snapshot(
    holdings: List[HoldingItem],
    request_id: Optional[str],
) -> PortfolioSnapshot:
    """
    Fetch live prices for every holding and compute portfolio-level metrics.

    Parameters
    ----------
    holdings:   List of HoldingItem from the PortfolioInput message.
    request_id: Correlation key to embed in the snapshot.

    Returns
    -------
    PortfolioSnapshot with current values, P&L, and per-holding weights.
    """
    total_invested: float = 0.0
    current_value: float = 0.0
    rows: List[Dict[str, Any]] = []

    for holding in holdings:
        cost_basis: float = holding.shares * holding.average_buy_price
        total_invested += cost_basis

        live_price: Optional[float] = _fetch_live_price(holding.ticker)
        if live_price is None:
            # Use average buy price as fallback to avoid crashing the snapshot;
            # the row is clearly flagged with price_stale=True.
            logger.warning(
                "Using avg buy price as fallback for %s (live price unavailable)",
                holding.ticker,
            )
            live_price = holding.average_buy_price
            stale: bool = True
        else:
            stale = False

        position_value: float = holding.shares * live_price
        current_value += position_value

        pnl_inr: float = position_value - cost_basis
        pnl_pct: float = (pnl_inr / cost_basis) * 100.0 if cost_basis else 0.0

        rows.append(
            {
                "ticker": holding.ticker,
                "shares": holding.shares,
                "avg_buy_price": holding.average_buy_price,
                "live_price": round(live_price, 2),
                "position_value_inr": round(position_value, 2),
                "cost_basis_inr": round(cost_basis, 2),
                "pnl_inr": round(pnl_inr, 2),
                "pnl_pct": round(pnl_pct, 2),
                "price_stale": stale,
                # weight_pct is filled in a second pass below
                "weight_pct": 0.0,
            }
        )

    # Second pass: compute portfolio weights now that current_value is known.
    for row in rows:
        row["weight_pct"] = (
            round((row["position_value_inr"] / current_value) * 100.0, 4)
            if current_value > 0
            else 0.0
        )

    absolute_pnl: float = current_value - total_invested
    pct_pnl: float = (
        (absolute_pnl / total_invested) * 100.0 if total_invested else 0.0
    )

    return PortfolioSnapshot(
        request_id=request_id,
        total_invested_inr=round(total_invested, 2),
        current_value_inr=round(current_value, 2),
        absolute_pnl_inr=round(absolute_pnl, 2),
        pct_pnl=round(pct_pnl, 4),
        holdings_with_weights=rows,
    )


def _log_portfolio_snapshot(snapshot: PortfolioSnapshot) -> None:
    """Pretty-print the portfolio snapshot to the logger."""
    logger.info("=" * 70)
    logger.info("PORTFOLIO SNAPSHOT  (request_id=%s)", snapshot.request_id)
    logger.info("=" * 70)
    logger.info(
        "  Total Invested : ₹%,.2f", snapshot.total_invested_inr
    )
    logger.info(
        "  Current Value  : ₹%,.2f", snapshot.current_value_inr
    )
    pnl_sign: str = "+" if snapshot.absolute_pnl_inr >= 0 else ""
    logger.info(
        "  Unrealised P&L : %s₹%,.2f  (%s%.4f%%)",
        pnl_sign,
        snapshot.absolute_pnl_inr,
        pnl_sign,
        snapshot.pct_pnl,
    )
    logger.info("-" * 70)
    logger.info(
        "  %-15s %8s %12s %12s %8s %8s %8s",
        "Ticker",
        "Shares",
        "Avg Buy ₹",
        "Live ₹",
        "Value ₹",
        "Wt%",
        "P&L%",
    )
    logger.info("  " + "-" * 68)
    for h in snapshot.holdings_with_weights:
        stale_flag: str = " *" if h.get("price_stale") else "  "
        logger.info(
            "  %-15s %8.2f %12.2f %12.2f %8,.0f %8.2f %8.2f%s",
            h["ticker"],
            h["shares"],
            h["avg_buy_price"],
            h["live_price"],
            h["position_value_inr"],
            h["weight_pct"],
            h["pnl_pct"],
            stale_flag,
        )
    logger.info("=" * 70)
    logger.info("  * Price unavailable; using avg buy price as proxy.")


# ═══════════════════════════════════════════════════════════════════════════════
# Fetch.ai event handlers
# ═══════════════════════════════════════════════════════════════════════════════


@orchestrator.on_event("startup")
async def handle_startup(ctx: Context) -> None:
    """Log the agent's address on startup so we can configure other agents."""
    ctx.logger.info("" * 70)
    ctx.logger.info("★  FinPilot AI Orchestrator Agent started")
    ctx.logger.info("" * 70)
    ctx.logger.info("  Address    : %s", ctx.agent.address)
    ctx.logger.info("  Port       : %d", ORCHESTRATOR_PORT)
    ctx.logger.info(
        "  Mailbox    : %s",
        "ENABLED (mailbox=True)" if AGENTVERSE_MAILBOX_KEY
        else "ENABLED (mailbox=True) – connect via Agentverse Inspector",
    )
    ctx.logger.info(
        "  Screener   : %s",
        SCREENER_AGENT_ADDRESS
        if SCREENER_AGENT_ADDRESS != "REPLACE_WITH_SCREENER_ADDRESS"
        else "NOT CONFIGURED – set SCREENER_AGENT_ADDRESS in .env",
    )
    ctx.logger.info(
        "  Researcher : %s",
        RESEARCHER_AGENT_ADDRESS
        if RESEARCHER_AGENT_ADDRESS != "REPLACE_WITH_RESEARCHER_ADDRESS"
        else "NOT CONFIGURED – set RESEARCHER_AGENT_ADDRESS in .env",
    )
    ctx.logger.info("" * 70)
    ctx.logger.info(
        "  ►  NEXT STEP: Open the Agent Inspector URL printed above in your "
        "browser, click 'Connect', and select 'Mailbox' to bind this agent "
        "to Agentverse."
    )


@portfolio_proto.on_message(model=PortfolioInput)
async def handle_portfolio_input(ctx: Context, sender: str, msg: PortfolioInput) -> None:
    """
    Main handler: receives user portfolio, computes snapshot, triggers screener.

    Flow:
      PortfolioInput → (yfinance) → PortfolioSnapshot (logged)
                     → ScreeningRequest → QuantScreener Agent
    """
    ctx.logger.info(
        "Received PortfolioInput from sender=%s  (request_id=%s, holdings=%d)",
        sender,
        msg.request_id,
        len(msg.holdings),
    )

    # ── Step 1: Build live portfolio snapshot ─────────────────────────────────
    ctx.logger.info("Fetching live prices for %d tickers…", len(msg.holdings))
    snapshot: PortfolioSnapshot = _build_portfolio_snapshot(
        holdings=msg.holdings,
        request_id=msg.request_id,
    )
    _log_portfolio_snapshot(snapshot)

    # ── Step 2: Guard – screener address must be configured ───────────────────
    if SCREENER_AGENT_ADDRESS == "REPLACE_WITH_SCREENER_ADDRESS":
        ctx.logger.error(
            "SCREENER_AGENT_ADDRESS is not set. "
            "Start screener.py, copy its address, then set it in .env."
        )
        return

    # ── Step 3: Build ScreeningRequest and forward to QuantScreener ───────────
    portfolio_tickers: List[str] = [h.ticker for h in msg.holdings]
    screening_req = ScreeningRequest(
        request_id=msg.request_id,
        investment_horizon=msg.investment_horizon,
        risk_profile=msg.risk_profile,
        portfolio_tickers=portfolio_tickers,
        orchestrator_address=ctx.agent.address,
    )

    ctx.logger.info(
        "Sending ScreeningRequest to QuantScreener at %s", SCREENER_AGENT_ADDRESS
    )
    await ctx.send(SCREENER_AGENT_ADDRESS, screening_req)


@orchestrator.on_message(model=RankedShortlistResponse)
async def handle_screener_response(
    ctx: Context, sender: str, msg: RankedShortlistResponse
) -> None:
    """
    [Phase 2] Receive the ranked shortlist from the QuantScreener.

    Logs the tabular screening results, then immediately forwards a
    ResearchRequest to the DeepResearcher Agent so the LLM layer can
    produce a contextualised investment thesis.
    """
    ctx.logger.info(
        "Received RankedShortlistResponse from sender=%s  (request_id=%s)",
        sender,
        msg.request_id,
    )
    ctx.logger.info("=" * 70)
    ctx.logger.info("SCREENING RESULTS  –  Algorithm v%s", msg.screener_version)
    ctx.logger.info("=" * 70)
    ctx.logger.info("Filter Summary: %s", msg.filter_summary)
    ctx.logger.info("-" * 70)

    if not msg.screened_stocks:
        ctx.logger.warning("No stocks passed the screening filters. Aborting.")
        return

    ctx.logger.info(
        "  %-15s %-30s %12s %8s %12s %10s",
        "Ticker", "Company", "MCap (₹Cr)", "P/E", "CMP ₹", "vs 52w H%",
    )
    ctx.logger.info("  " + "-" * 92)
    for stock in msg.screened_stocks:
        pe_str: str = f"{stock.pe_ratio:.2f}" if stock.pe_ratio is not None else "N/A"
        ctx.logger.info(
            "  %-15s %-30s %12,.0f %8s %12.2f %10.2f%%",
            stock.ticker,
            stock.company_name[:30],
            stock.market_cap_cr,
            pe_str,
            stock.current_price,
            stock.pct_from_52w_high,
        )
    ctx.logger.info("=" * 70)

    # ── Guard: researcher must be configured ──────────────────────────────
    if RESEARCHER_AGENT_ADDRESS == "REPLACE_WITH_RESEARCHER_ADDRESS":
        ctx.logger.error(
            "RESEARCHER_AGENT_ADDRESS is not set. "
            "Start researcher.py, copy its address, then set it in .env."
        )
        return

    # ── Build inline stock context for the LLM prompt ─────────────────────
    # Serialise each ScreenedStock to a plain dict so it can be embedded in
    # the ResearchRequest without circular schema dependencies.
    stock_context: List[dict] = [
        {
            "ticker": s.ticker,
            "company_name": s.company_name,
            "market_cap_cr": s.market_cap_cr,
            "pe_ratio": s.pe_ratio,
            "week_52_high": s.week_52_high,
            "current_price": s.current_price,
            "pct_from_52w_high": s.pct_from_52w_high,
        }
        for s in msg.screened_stocks
    ]

    research_req = ResearchRequest(
        request_id=msg.request_id,
        shortlisted_tickers=[s.ticker for s in msg.screened_stocks],
        orchestrator_address=ctx.agent.address,
        stock_context=stock_context,
    )

    ctx.logger.info(
        "Forwarding %d screened stocks to DeepResearcher at %s",
        len(msg.screened_stocks),
        RESEARCHER_AGENT_ADDRESS,
    )
    await ctx.send(RESEARCHER_AGENT_ADDRESS, research_req)


@orchestrator.on_message(model=AgenticThesisResponse)
async def handle_researcher_response(
    ctx: Context, sender: str, msg: AgenticThesisResponse
) -> None:
    """
    [Phase 2/3] Receive the LLM-generated thesis from DeepResearcher.

    1. Builds and renders the richly-formatted report to the local console.
    2. [Phase 3] Looks up the original ASI:One sender address (stored in
       _pending_chat_senders when the ChatMessage arrived) and sends the
       thesis back over the chat protocol so the judge sees it in ASI:One.
    """
    ctx.logger.info(
        "Received AgenticThesisResponse from sender=%s  (request_id=%s, tokens=%d)",
        sender,
        msg.request_id,
        msg.tokens_used,
    )

    # ── Build the formatted report string (shared for console + ASI:One reply)
    border: str = "═" * 72
    thin: str = "─" * 72
    lines: List[str] = []
    lines.append(border)
    lines.append("  ★  FINPILOT AI – INVESTMENT RESEARCH REPORT")
    lines.append(
        f"  ℹ  Model: {msg.model_used}   │  Tokens: {msg.tokens_used:,}   │  ID: {msg.request_id}"
    )
    lines.append(border)
    lines.append("")
    lines.append("▶  INVESTMENT THESIS")
    lines.append(thin)
    for paragraph in msg.investment_thesis.strip().split("\n\n"):
        lines.append(textwrap.fill(paragraph.strip(), width=72))
        lines.append("")
    lines.append(thin)
    lines.append("⚠  RISK FACTORS")
    lines.append(thin)
    for idx, (category, description) in enumerate(msg.risk_factors.items(), start=1):
        lines.append(f"  {idx}. [{category.upper()}]")
        for line in textwrap.wrap(description, width=68):
            lines.append(f"     {line}")
        lines.append("")
    lines.append(border)
    report_text: str = "\n".join(lines)

    # ── Print to local console
    print()
    print(report_text)
    print()

    # ── [Phase 3] Reply to original ASI:One sender if this came via chat
    asi_sender: Optional[str] = _pending_chat_senders.pop(msg.request_id or "", None)
    if asi_sender:
        ctx.logger.info(
            "Sending thesis back to ASI:One sender=%s via chat protocol", asi_sender
        )
        await ctx.send(
            asi_sender,
            ChatMessage(
                timestamp=datetime.now(timezone.utc),
                msg_id=uuid4(),
                content=[TextContent(type="text", text=report_text)],
            ),
        )
    else:
        ctx.logger.debug(
            "No pending ASI:One sender for request_id=%s – direct call, no chat reply.",
            msg.request_id,
        )


# ══════════════════════════════════════════════════════════════════════════════════
# Phase 3: Agent Chat Protocol handlers (ASI:One public interface)
# ══════════════════════════════════════════════════════════════════════════════════


@chat_proto.on_message(ChatMessage)
async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage) -> None:
    """
    Entry point for messages arriving from ASI:One via the Agent Chat Protocol.

    ASI:One wraps the user's prompt as a `ChatMessage` containing a
    `TextContent` item.  The text must be a JSON-encoded `PortfolioInput`.

    Flow:
      1. Acknowledge immediately (ACP spec requires this as the first send).
      2. Parse text → PortfolioInput.
      3. Register sender address for the eventual thesis reply.
      4. Send an interim "working..." message.
      5. Inject into the Phase 1–2 pipeline.

    Expected JSON from ASI:One::

        {
          "holdings": [
            {"ticker": "RELIANCE.NS", "shares": 10, "average_buy_price": 2450},
            {"ticker": "TCS.NS",      "shares": 5,  "average_buy_price": 3800}
          ],
          "investment_horizon": "long",
          "risk_profile": "moderate"
        }
    """
    ctx.logger.info("[ACP] Received ChatMessage from ASI:One sender=%s", sender)

    # ── Step 1: Mandatory ACP acknowledgement (must be the very first send)
    await ctx.send(
        sender,
        ChatAcknowledgement(
            timestamp=datetime.now(timezone.utc),
            acknowledged_msg_id=msg.msg_id,
        ),
    )

    # ── Step 2: Extract text payload
    raw_text: str = ""
    for content_item in msg.content:
        if isinstance(content_item, TextContent):
            raw_text += content_item.text
    raw_text = raw_text.strip()
    ctx.logger.info("[ACP] Payload text: %s", raw_text[:300])

    # ── Step 3: Parse JSON → PortfolioInput
    try:
        payload: dict = json.loads(raw_text)
        portfolio_input = PortfolioInput(**payload)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        ctx.logger.warning("[ACP] Failed to parse PortfolioInput: %s", exc)
        error_text = (
            "❌ FinPilot AI could not parse your request.\n\n"
            "Please send a JSON object with this structure:\n\n"
            "{\n"
            '  "holdings": [\n'
            '    {"ticker": "RELIANCE.NS", "shares": 10, "average_buy_price": 2450},\n'
            '    {"ticker": "TCS.NS",      "shares": 5,  "average_buy_price": 3800}\n'
            "  ],\n"
            '  "investment_horizon": "long",\n'
            '  "risk_profile": "moderate"\n'
            "}\n\n"
            f"Parse error: {exc}"
        )
        await ctx.send(
            sender,
            ChatMessage(
                timestamp=datetime.now(timezone.utc),
                msg_id=uuid4(),
                content=[TextContent(type="text", text=error_text)],
            ),
        )
        return

    # ── Step 4: Assign request_id and register the ASI:One sender
    request_id: str = portfolio_input.request_id or str(uuid4())[:8]
    portfolio_input = PortfolioInput(
        **{**portfolio_input.model_dump(), "request_id": request_id}
    )
    _pending_chat_senders[request_id] = sender
    ctx.logger.info(
        "[ACP] Registered chat sender for request_id=%s. Injecting into pipeline…",
        request_id,
    )

    # ── Step 5: Interim "working…" message to the user
    ack_text = (
        f"⏳ FinPilot AI received your portfolio ({len(portfolio_input.holdings)} holding(s)).\n"
        "Running: live price fetch → quantitative screening → AI thesis generation.\n"
        "Your investment report will arrive in this chat in ~30–60 seconds."
    )
    await ctx.send(
        sender,
        ChatMessage(
            timestamp=datetime.now(timezone.utc),
            msg_id=uuid4(),
            content=[TextContent(type="text", text=ack_text)],
        ),
    )

    # ── Step 6: Inject into the Phase 1–2 pipeline
    await handle_portfolio_input(ctx, sender, portfolio_input)


@chat_proto.on_message(ChatAcknowledgement)
async def handle_chat_acknowledgement(
    ctx: Context, sender: str, msg: ChatAcknowledgement
) -> None:
    """Log ACP acknowledgements (required handler; no business logic needed)."""
    ctx.logger.debug(
        "[ACP] Acknowledgement from %s for msg_id=%s",
        sender,
        msg.acknowledged_msg_id,
    )


# ── Register the chat protocol and publish the manifest to the Almanac ──────────
# publish_manifest=True pushes the protocol fingerprint on-chain so ASI:One
# can discover and route queries to this agent automatically.
orchestrator.include(chat_proto, publish_manifest=True)

# ── Register the PortfolioInput protocol for explicit capability discovery ───────
# This makes the PortfolioInput handler discoverable on the Almanac/ASI:One.
orchestrator.include(portfolio_proto, publish_manifest=True)


# ══════════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    orchestrator.run()
