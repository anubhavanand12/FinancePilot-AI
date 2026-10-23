"""
schemas.py – Canonical Pydantic/uagents message schemas for FinPilot AI.

Every inter-agent message MUST be defined here so the entire system shares a
single, versioned contract.  Import this module in every agent file.

Fetch.ai requirement: all message classes must inherit from `uagents.Model`,
which itself inherits from `pydantic.BaseModel` (v1).  This gives us full Pydantic
v1 validation.

Phase 2 additions
-----------------
  ResearchRequest       – Orchestrator → DeepResearcher
  AgenticThesisResponse – DeepResearcher → Orchestrator

Phase 3 additions
-----------------
  UAgentResponse        – Orchestrator → ASI:One chat protocol reply
  PortfolioInput Field descriptions have been enriched with ASI:One UX copy
  so the Agentverse auto-generated form is maximally clear to judges.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic.v1 import Field, validator
from uagents import Model

# ═══════════════════════════════════════════════════════════════════════════════
# Enumerations
# ═══════════════════════════════════════════════════════════════════════════════


class InvestmentHorizon(str, Enum):
    """Allowed investment horizon labels."""

    SHORT = "short"       # < 3 months
    MEDIUM = "medium"     # 3–12 months
    LONG = "long"         # > 12 months


class RiskProfile(str, Enum):
    """Investor risk tolerance."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


# ═══════════════════════════════════════════════════════════════════════════════
# Sub-models (not sent directly over the wire; used as nested fields)
# ═══════════════════════════════════════════════════════════════════════════════


class HoldingItem(Model):
    """
    A single equity position held by the user.

    Attributes:
        ticker:            NSE ticker symbol including .NS suffix
                           (e.g. "RELIANCE.NS").
        shares:            Number of shares held (must be > 0).
        average_buy_price: Volume-weighted average cost of acquisition in INR.
    """

    ticker: str = Field(
        ...,
        description="NSE ticker symbol (must end with .NS or .BO)",
        examples=["RELIANCE.NS", "TCS.NS"],
    )
    shares: float = Field(..., gt=0, description="Number of shares held")
    average_buy_price: float = Field(
        ..., gt=0, description="Average buy price per share in INR"
    )

    @validator("ticker")
    def validate_ticker(cls, v: str) -> str:
        v = v.strip().upper()
        if not (v.endswith(".NS") or v.endswith(".BO")):
            raise ValueError(
                f"Ticker '{v}' must end with '.NS' (NSE) or '.BO' (BSE)."
            )
        return v


class ScreenedStock(Model):
    """
    A single stock that passed the quantitative screening filters.

    Attributes:
        ticker:          NSE ticker symbol.
        company_name:    Human-readable company name from yfinance.
        market_cap_cr:   Market capitalisation in Indian Crores (₹).
        pe_ratio:        Trailing price-to-earnings ratio (NaN → None).
        week_52_high:    52-week high price in INR.
        current_price:   Last traded price in INR at time of fetch.
        pct_from_52w_high: Percentage below the 52-week high (negative = below).
    """

    ticker: str
    company_name: str
    market_cap_cr: float = Field(..., description="Market cap in ₹ Crores")
    pe_ratio: Optional[float] = Field(
        None, description="Trailing P/E ratio; None if unavailable"
    )
    week_52_high: float = Field(..., description="52-week high price in INR")
    current_price: float = Field(..., description="Last traded price in INR")
    pct_from_52w_high: float = Field(
        ..., description="% discount from 52-week high (negative means below)"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Wire-level Messages (sent between agents)
# ═══════════════════════════════════════════════════════════════════════════════


class PortfolioInput(Model):
    """
    Message sent BY the user / ASI:One interface TO the Orchestrator Agent.

    This is the **primary public interface** of FinPilot AI.  ASI:One reads
    the `description` of each Field to auto-generate the input form shown
    to the end user.  Keep descriptions concise and user-facing.

    Attributes:
        holdings:           List of equity positions.
        investment_horizon: The user's investment time frame.
        risk_profile:       The user's stated risk tolerance.
        base_currency:      ISO 4217 code; default "INR".
        request_id:         Optional client-generated idempotency key.
    """

    holdings: List[HoldingItem] = Field(
        ...,
        min_items=1,
        title="Portfolio Holdings",
        description=(
            "Your current NSE equity positions. Each entry requires: "
            "'ticker' (e.g. RELIANCE.NS), 'shares' (number owned), and "
            "'average_buy_price' in INR. At least one holding is required."
        ),
    )
    investment_horizon: InvestmentHorizon = Field(
        InvestmentHorizon.LONG,
        title="Investment Horizon",
        description=(
            "Your target holding period: "
            "'short' (< 3 months), 'medium' (3–12 months), 'long' (> 12 months). "
            "Defaults to 'long'."
        ),
    )
    risk_profile: RiskProfile = Field(
        RiskProfile.MODERATE,
        title="Risk Profile",
        description=(
            "Your risk tolerance: 'conservative' (capital preservation), "
            "'moderate' (balanced growth), or 'aggressive' (high-conviction). "
            "Defaults to 'moderate'."
        ),
    )
    base_currency: str = Field(
        "INR",
        title="Base Currency",
        description="Portfolio valuation currency. Must be a 3-letter ISO 4217 code (e.g. INR, USD).",
        pattern=r"^[A-Z]{3}$",
    )
    request_id: Optional[str] = Field(
        None,
        title="Request ID",
        description="Optional unique identifier for this analysis request (for tracking / idempotency).",
    )


class ScreeningRequest(Model):
    """
    Message sent BY the Orchestrator TO the QuantScreener Agent.

    Carries the context required for the screener to select and rank stocks
    against the user's preference profile.

    Attributes:
        request_id:         Propagated from the originating PortfolioInput.
        investment_horizon: Forwarded from the user's input.
        risk_profile:       Forwarded from the user's input.
        portfolio_tickers:  Tickers already held; screener may prefer names
                            not in this list to avoid duplication.
        orchestrator_address: The Orchestrator's agent address so the screener
                              can reply correctly.
    """

    request_id: Optional[str] = None
    investment_horizon: InvestmentHorizon = InvestmentHorizon.LONG
    risk_profile: RiskProfile = RiskProfile.MODERATE
    portfolio_tickers: List[str] = Field(
        default_factory=list,
        description="Tickers already in the user's portfolio",
    )
    orchestrator_address: str = Field(
        ..., description="Reply-to address of the Orchestrator Agent"
    )


class PortfolioSnapshot(Model):
    """
    Calculated portfolio metrics produced by the Orchestrator.

    Emitted as a log entry / event; also embedded in RankedShortlistResponse.
    """

    request_id: Optional[str] = None
    total_invested_inr: float = Field(
        ..., description="Sum of (shares × avg_buy_price) across all holdings"
    )
    current_value_inr: float = Field(
        ..., description="Sum of (shares × live_price) across all holdings"
    )
    absolute_pnl_inr: float = Field(
        ..., description="current_value_inr − total_invested_inr"
    )
    pct_pnl: float = Field(
        ..., description="(absolute_pnl_inr / total_invested_inr) × 100"
    )
    holdings_with_weights: List[dict] = Field(
        ...,
        description=(
            "Each dict: {ticker, shares, avg_buy_price, live_price, "
            "position_value_inr, weight_pct, pnl_inr, pnl_pct}"
        ),
    )


class RankedShortlistResponse(Model):
    """
    Final output message sent BY the QuantScreener BACK to the Orchestrator
    (and ultimately surfaced to the user).

    Attributes:
        request_id:       Echoed from ScreeningRequest for correlation.
        screened_stocks:  Ordered list of stocks that passed all filters
                          (best first by composite score).
        filter_summary:   Human-readable explanation of which filters were
                          applied and how many candidates were eliminated.
        portfolio_snapshot: Live portfolio metrics computed by the Orchestrator.
        screener_version: Semantic version string of the screening algorithm.
    """

    request_id: Optional[str] = None
    screened_stocks: List[ScreenedStock] = Field(
        ..., description="Stocks that passed all quantitative filters"
    )
    filter_summary: str = Field(
        ...,
        description="Narrative summary: candidates in, filters applied, survivors out",
    )
    portfolio_snapshot: Optional[PortfolioSnapshot] = Field(
        None, description="Live portfolio metrics at time of request"
    )
    screener_version: str = Field(
        "1.0.0", description="Version of the screening algorithm"
    )


# ── Phase 2 messages ──────────────────────────────────────────────────────────


class ResearchRequest(Model):
    """
    Message sent BY the Orchestrator TO the DeepResearcher Agent.

    Carries the quant-screened shortlist so the LLM layer can produce a
    contextualised investment thesis grounded in real ticker data.

    Attributes:
        request_id:           Propagated correlation key.
        shortlisted_tickers:  Ordered list of NSE tickers that passed all
                              quantitative filters (best-ranked first).
        investment_horizon:   User's stated time horizon for the LLM to tailor
                              its recommendation accordingly.
        risk_profile:         User's risk tolerance – informs LLM tone.
        orchestrator_address: Reply-to address so the researcher knows where
                              to send the AgenticThesisResponse.
        stock_context:        Serialised list of ScreenedStock dicts (JSON-
                              compatible) so the LLM has fundamental data
                              (MCap, P/E, 52w high) inline without a re-fetch.
    """

    request_id: Optional[str] = None
    shortlisted_tickers: List[str] = Field(
        ...,
        min_items=1,
        description="Ordered NSE tickers that survived quantitative screening",
    )
    investment_horizon: InvestmentHorizon = InvestmentHorizon.LONG
    risk_profile: RiskProfile = RiskProfile.MODERATE
    orchestrator_address: str = Field(
        ..., description="Reply-to address of the Orchestrator Agent"
    )
    stock_context: List[Dict] = Field(
        default_factory=list,
        description="Serialised ScreenedStock dicts for inline LLM context",
    )


class AgenticThesisResponse(Model):
    """
    Message sent BY the DeepResearcher Agent BACK to the Orchestrator.

    Contains structured LLM output: a 2-paragraph investment thesis and a
    mapping of risk-factor category → description.

    Attributes:
        request_id:        Echoed from ResearchRequest for correlation.
        investment_thesis: Two-paragraph free-text thesis generated by the LLM.
        risk_factors:      Dict mapping risk category label to a concise
                           description, e.g.
                           {"Macro": "Rising US yields…",
                            "Regulatory": "SEBI F&O margin hike…",
                            "Valuation": "Premium to 5-yr avg P/E…"}.
        model_used:        The LLM model identifier used to generate this output
                           (e.g. "gpt-4o-mini"). Useful for audit trails.
        tokens_used:       Total token count for cost tracking (prompt + completion).
    """

    request_id: Optional[str] = None
    investment_thesis: str = Field(
        ...,
        description="Two-paragraph investment thesis from the LLM",
    )
    risk_factors: Dict[str, str] = Field(
        ...,
        description="Risk category → concise description mapping",
    )
    model_used: str = Field(
        "gpt-4o-mini",
        description="LLM model identifier used for this response",
    )
    tokens_used: int = Field(
        0,
        description="Total token usage (prompt + completion) for cost tracking",
    )


# ── Phase 3: ASI:One chat protocol reply ───────────────────────────────────


class UAgentResponse(Model):
    """
    Convenience model for sending a plain-text message back to ASI:One
    via the standard Fetch.ai chat protocol.

    This is used internally by the Orchestrator's chat handler to build
    a `ChatMessage` response wrapping this text.  It is NOT sent over the
    wire directly (the chat protocol wraps it); it is defined here so all
    agents share the same string format contract.

    Attributes:
        message: The formatted text to display to the ASI:One user.
    """

    message: str = Field(
        ...,
        title="Response Message",
        description="Formatted investment report text to display to the user.",
    )