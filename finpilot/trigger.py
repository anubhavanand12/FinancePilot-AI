"""
trigger.py – Manual test client for FinPilot AI Phase 1
=======================================================
This script acts as a synthetic "user" by sending a PortfolioInput message
directly to the running Orchestrator Agent.

Usage:
    python trigger.py

Prerequisites:
    1. screener.py is running  (Terminal 1)
    2. orchestrator.py is running with SCREENER_AGENT_ADDRESS configured (Terminal 2)
    3. Copy the Orchestrator's printed address and set ORCHESTRATOR_ADDRESS below
       (or set it in .env as ORCHESTRATOR_AGENT_ADDRESS).
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from uagents import Agent, Context

sys.path.insert(0, os.path.dirname(__file__))

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from schemas import HoldingItem, InvestmentHorizon, PortfolioInput, RiskProfile

# ── Configuration ──────────────────────────────────────────────────────────────
# Paste the Orchestrator's address here (printed on startup), or set it in .env
ORCHESTRATOR_ADDRESS: str = os.getenv(
    "ORCHESTRATOR_AGENT_ADDRESS",
    "REPLACE_WITH_ORCHESTRATOR_ADDRESS",
)

TRIGGER_SEED: str = "trigger_client_seed_finpilot_ai_v1_not_production"
TRIGGER_PORT: int = int(os.getenv("TRIGGER_PORT", "8003"))

# ── Sample portfolio (edit freely) ─────────────────────────────────────────────
SAMPLE_PORTFOLIO = PortfolioInput(
    holdings=[
        HoldingItem(ticker="RELIANCE.NS", shares=10, average_buy_price=2450.00),
        HoldingItem(ticker="TCS.NS",      shares=5,  average_buy_price=3800.00),
        HoldingItem(ticker="HDFCBANK.NS", shares=20, average_buy_price=1600.00),
    ],
    investment_horizon=InvestmentHorizon.LONG,
    risk_profile=RiskProfile.MODERATE,
    request_id="test-run-001",
)

# ═══════════════════════════════════════════════════════════════════════════════
# Trigger agent
# ═══════════════════════════════════════════════════════════════════════════════

trigger = Agent(
    name="TriggerClient",
    seed=TRIGGER_SEED,
    port=TRIGGER_PORT,
    endpoint=[f"http://localhost:{TRIGGER_PORT}/submit"],
)


@trigger.on_event("startup")
async def send_portfolio(ctx: Context) -> None:
    if ORCHESTRATOR_ADDRESS == "REPLACE_WITH_ORCHESTRATOR_ADDRESS":
        ctx.logger.error(
            "Set ORCHESTRATOR_AGENT_ADDRESS in .env or edit trigger.py directly."
        )
        return

    ctx.logger.info("Sending sample PortfolioInput to Orchestrator…")
    ctx.logger.info("  Holdings: %s", [h.ticker for h in SAMPLE_PORTFOLIO.holdings])
    await ctx.send(ORCHESTRATOR_ADDRESS, SAMPLE_PORTFOLIO)
    ctx.logger.info("Message sent. Watch the Orchestrator and Screener terminals.")


if __name__ == "__main__":
    trigger.run()
