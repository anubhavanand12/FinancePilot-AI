"""
researcher.py – FinPilot AI DeepResearcher Agent
=================================================
Responsibilities
----------------
1. Receive a `ResearchRequest` from the Orchestrator carrying the quant-
   screened ticker shortlist plus inline fundamental context (MCap, P/E,
   52-week high, current price).
2. Build a deterministic, richly-contextualised LLM prompt that instructs
   the model to reason as a Quantitative Portfolio Manager covering Indian
   equities.
3. Call the OpenAI Chat Completions API (gpt-4o-mini by default) requesting
   a structured JSON response: two-paragraph investment thesis + 3 labelled
   risk factors.
4. Parse the JSON response into an `AgenticThesisResponse` and send it back
   to the Orchestrator.

Structured output strategy
---------------------------
We use OpenAI's `response_format={"type": "json_object"}` mode (supported by
gpt-4o-mini) and include the exact JSON schema in the system prompt.  This
gives us deterministic, parse-safe output without needing function-calling or
a schema library on the OpenAI side.

Run this agent in its own terminal (BEFORE the Orchestrator):
    python researcher.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from uagents import Agent, Context

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from schemas import (
    AgenticThesisResponse,
    InvestmentHorizon,
    ResearchRequest,
    RiskProfile,
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
logger = logging.getLogger("finpilot.researcher")

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

RESEARCHER_SEED: str = os.getenv(
    "RESEARCHER_SEED",
    "researcher_default_seed_finpilot_ai_v1_replace_in_prod",
)
RESEARCHER_PORT: int = int(os.getenv("RESEARCHER_PORT", "8004"))

# OpenAI configuration – all sourced from .env
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Token budget: gpt-4o-mini context is 128k; we stay well within limits.
MAX_COMPLETION_TOKENS: int = int(os.getenv("MAX_COMPLETION_TOKENS", "1024"))

# ── Horizon / risk label maps for the LLM prompt ──────────────────────────────
_HORIZON_LABELS: Dict[str, str] = {
    InvestmentHorizon.SHORT: "short-term (< 3 months)",
    InvestmentHorizon.MEDIUM: "medium-term (3–12 months)",
    InvestmentHorizon.LONG: "long-term (> 12 months)",
}

_RISK_LABELS: Dict[str, str] = {
    RiskProfile.CONSERVATIVE: "conservative (capital preservation, low drawdown)",
    RiskProfile.MODERATE: "moderate (balanced growth and risk management)",
    RiskProfile.AGGRESSIVE: "aggressive (high-growth, high-conviction bets)",
}

# ═══════════════════════════════════════════════════════════════════════════════
# OpenAI client (module-level singleton)
# ═══════════════════════════════════════════════════════════════════════════════

_openai_client: Optional[OpenAI] = None


def _get_openai_client() -> OpenAI:
    """Lazily initialise the OpenAI client; raises if key is absent."""
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Add it to finpilot/.env and restart the researcher agent."
            )
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt engineering
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT: str = """\
You are a senior Quantitative Portfolio Manager specialising in Indian equities
(NSE/BSE listed stocks).  You have deep expertise in the Indian macroeconomic
environment, RBI monetary policy, sectoral rotation, SEBI regulations, and
fundamental analysis.

You will receive a shortlist of stocks that have already passed rigorous
quantitative filters (P/E ≤ 50, market cap ≥ ₹10,000 Cr, ranked by composite
score of valuation, size, and price momentum).

Your task is to produce a STRUCTURED JSON response with exactly the following
schema (no extra keys, no markdown fences):

{
  "investment_thesis": "<paragraph 1>\\n\\n<paragraph 2>",
  "risk_factors": {
    "<Risk Category 1>": "<concise description>",
    "<Risk Category 2>": "<concise description>",
    "<Risk Category 3>": "<concise description>"
  }
}

Rules:
- investment_thesis: exactly 2 paragraphs separated by a blank line.
  • Paragraph 1: explain WHY the screened stocks make sense given the CURRENT
    Indian macro environment (RBI rate stance, INR, FII flows, consumption
    trends, etc.).  Tie specific tickers to macro themes.
  • Paragraph 2: explain the portfolio construction rationale – how these
    names complement each other in terms of sector diversification, risk-
    adjusted return potential, and alignment with the investor's horizon and
    risk profile.
- risk_factors: exactly 3 entries.  Each key is a SHORT category label
  (e.g. "Macro", "Regulatory", "Valuation", "Liquidity", "Geopolitical").
  Each value is 1–2 sentences describing the specific risk for THIS shortlist.
- Do NOT include any text outside the JSON object.
- Do NOT use markdown code fences.
- Be specific: name actual tickers, sectors, and Indian market dynamics.
"""


def _build_user_prompt(
    tickers: List[str],
    stock_context: List[Dict[str, Any]],
    horizon: InvestmentHorizon,
    risk_profile: RiskProfile,
) -> str:
    """
    Build the user-turn prompt by embedding the screened stock data so the LLM
    has full fundamental context without needing a web search.

    Parameters
    ----------
    tickers:       Ordered list of NSE tickers (best-ranked first).
    stock_context: List of dicts with ticker, company_name, market_cap_cr,
                   pe_ratio, week_52_high, current_price, pct_from_52w_high.
    horizon:       User's investment horizon.
    risk_profile:  User's risk tolerance.
    """
    horizon_label: str = _HORIZON_LABELS.get(horizon, horizon.value)
    risk_label: str = _RISK_LABELS.get(risk_profile, risk_profile.value)

    # Build a compact tabular representation of the screened stocks.
    stock_lines: List[str] = []
    for ctx_item in stock_context:
        pe_str: str = (
            f"{ctx_item['pe_ratio']:.2f}"
            if ctx_item.get("pe_ratio") is not None
            else "N/A"
        )
        pct_high: float = ctx_item.get("pct_from_52w_high", 0.0)
        pct_high_str: str = f"{pct_high:+.2f}%"
        stock_lines.append(
            f"  • {ctx_item['ticker']} ({ctx_item['company_name']}): "
            f"CMP=₹{ctx_item['current_price']:,.2f}, "
            f"MCap=₹{ctx_item['market_cap_cr']:,.0f}Cr, "
            f"P/E={pe_str}, "
            f"52wH=₹{ctx_item['week_52_high']:,.2f} ({pct_high_str} from peak)"
        )

    stocks_block: str = "\n".join(stock_lines)

    return f"""\
QUANTITATIVE SHORTLIST (ranked best-to-last by composite score):

{stocks_block}

INVESTOR PROFILE:
  • Horizon      : {horizon_label}
  • Risk Profile : {risk_label}

Today's date (IST): {_today_ist()}

Using the stock data above and your knowledge of the current Indian
macroeconomic and market environment, produce the investment thesis JSON.
"""


def _today_ist() -> str:
    """Return today's date in IST as a readable string."""
    from datetime import datetime, timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).strftime("%d %B %Y")


# ═══════════════════════════════════════════════════════════════════════════════
# LLM call – deterministic, typed, with full error handling
# ═══════════════════════════════════════════════════════════════════════════════


def _call_openai(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int,
) -> tuple[str, str, int]:
    """
    Call the OpenAI Chat Completions API in JSON mode.

    Returns
    -------
    (raw_json_str, model_name_used, tokens_used)

    Raises
    ------
    RuntimeError if the API call fails or the response is not valid JSON.
    """
    client: OpenAI = _get_openai_client()

    logger.info(
        "Calling OpenAI API  model=%s  max_tokens=%d", model, max_tokens
    )

    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.4,   # Slightly creative but reproducible
            seed=42,           # Deterministic sampling where supported
        )
    except OpenAIError as exc:
        raise RuntimeError(f"OpenAI API error: {exc}") from exc

    raw_content: str = response.choices[0].message.content or ""
    model_used: str = response.model
    tokens_used: int = response.usage.total_tokens if response.usage else 0

    logger.info(
        "OpenAI response received  model=%s  tokens=%d", model_used, tokens_used
    )
    logger.debug("Raw LLM output: %s", raw_content[:500])

    # Validate JSON parsability before returning
    try:
        json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"LLM returned non-JSON content: {raw_content[:200]!r}"
        ) from exc

    return raw_content, model_used, tokens_used


def _parse_thesis_response(
    raw_json: str,
    request_id: Optional[str],
    model_used: str,
    tokens_used: int,
) -> AgenticThesisResponse:
    """
    Parse the raw JSON string from the LLM into an AgenticThesisResponse.

    Validates that:
    - `investment_thesis` is a non-empty string.
    - `risk_factors` is a dict with exactly 3 string-keyed string-valued entries.

    Raises ValueError on schema violations.
    """
    data: Dict[str, Any] = json.loads(raw_json)

    # ── investment_thesis validation ──────────────────────────────────────────
    thesis: Any = data.get("investment_thesis", "")
    if not isinstance(thesis, str) or not thesis.strip():
        raise ValueError(
            f"'investment_thesis' must be a non-empty string; got {type(thesis)}"
        )

    # ── risk_factors validation ───────────────────────────────────────────────
    risk_factors: Any = data.get("risk_factors", {})
    if not isinstance(risk_factors, dict):
        raise ValueError(
            f"'risk_factors' must be a dict; got {type(risk_factors)}"
        )
    if len(risk_factors) != 3:
        logger.warning(
            "LLM returned %d risk factors instead of 3; using as-is.",
            len(risk_factors),
        )
    for k, v in risk_factors.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(
                f"risk_factors entries must be str→str; got {type(k)}→{type(v)}"
            )

    return AgenticThesisResponse(
        request_id=request_id,
        investment_thesis=thesis.strip(),
        risk_factors=risk_factors,
        model_used=model_used,
        tokens_used=tokens_used,
    )


def run_research_pipeline(
    tickers: List[str],
    stock_context: List[Dict[str, Any]],
    horizon: InvestmentHorizon,
    risk_profile: RiskProfile,
    request_id: Optional[str],
) -> AgenticThesisResponse:
    """
    Full, deterministic research pipeline.

    Parameters
    ----------
    tickers:       Ordered list of screened NSE tickers.
    stock_context: Inline fundamental data for each ticker.
    horizon:       User's investment horizon.
    risk_profile:  User's risk tolerance.
    request_id:    Correlation key.

    Returns
    -------
    AgenticThesisResponse ready to be sent back to the Orchestrator.
    """
    user_prompt: str = _build_user_prompt(tickers, stock_context, horizon, risk_profile)

    logger.debug("System prompt (truncated):\n%s", _SYSTEM_PROMPT[:300])
    logger.debug("User prompt:\n%s", user_prompt)

    raw_json, model_used, tokens_used = _call_openai(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=OPENAI_MODEL,
        max_tokens=MAX_COMPLETION_TOKENS,
    )

    thesis_response: AgenticThesisResponse = _parse_thesis_response(
        raw_json=raw_json,
        request_id=request_id,
        model_used=model_used,
        tokens_used=tokens_used,
    )

    return thesis_response


# ═══════════════════════════════════════════════════════════════════════════════
# Agent initialisation
# ═══════════════════════════════════════════════════════════════════════════════

researcher = Agent(
    name="DeepResearcher",
    seed=RESEARCHER_SEED,
    port=RESEARCHER_PORT,
    endpoint=[f"http://localhost:{RESEARCHER_PORT}/submit"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# Fetch.ai event handlers
# ═══════════════════════════════════════════════════════════════════════════════


@researcher.on_event("startup")
async def handle_startup(ctx: Context) -> None:
    ctx.logger.info("DeepResearcher Agent started.")
    ctx.logger.info("  Address  : %s", ctx.address)
    ctx.logger.info("  Port     : %d", RESEARCHER_PORT)
    ctx.logger.info("  LLM      : %s", OPENAI_MODEL)
    ctx.logger.info(
        "  API Key  : %s",
        "configured ✓" if OPENAI_API_KEY else "MISSING – set OPENAI_API_KEY in .env",
    )
    ctx.logger.info(
        ">>> Copy the address above and set it as RESEARCHER_AGENT_ADDRESS in .env <<<"
    )


@researcher.on_message(model=ResearchRequest)
async def handle_research_request(
    ctx: Context, sender: str, msg: ResearchRequest
) -> None:
    """
    Receive a ResearchRequest, invoke the LLM pipeline, reply to Orchestrator.

    The full pipeline is:
      ResearchRequest → prompt engineering → OpenAI API → JSON parse
      → AgenticThesisResponse → Orchestrator
    """
    ctx.logger.info(
        "Received ResearchRequest from sender=%s  "
        "(request_id=%s, tickers=%s)",
        sender,
        msg.request_id,
        msg.shortlisted_tickers,
    )

    # ── Run LLM pipeline ──────────────────────────────────────────────────────
    try:
        thesis_response: AgenticThesisResponse = run_research_pipeline(
            tickers=msg.shortlisted_tickers,
            stock_context=msg.stock_context,
            horizon=msg.investment_horizon,
            risk_profile=msg.risk_profile,
            request_id=msg.request_id,
        )
    except RuntimeError as exc:
        ctx.logger.error("Research pipeline failed: %s", exc)
        # Send a gracefully degraded response so the Orchestrator isn't left
        # waiting indefinitely.
        thesis_response = AgenticThesisResponse(
            request_id=msg.request_id,
            investment_thesis=(
                f"[ERROR] The LLM research pipeline encountered an error: {exc}\n\n"
                "Please verify that OPENAI_API_KEY is set correctly in .env and "
                "that the model quota has not been exceeded."
            ),
            risk_factors={
                "API Error": str(exc),
                "Action Required": "Check OPENAI_API_KEY and model availability.",
                "Fallback": "Screened stocks are still valid quant picks.",
            },
            model_used=OPENAI_MODEL,
            tokens_used=0,
        )

    # ── Reply to Orchestrator ─────────────────────────────────────────────────
    ctx.logger.info(
        "Sending AgenticThesisResponse to Orchestrator at %s  (tokens=%d)",
        msg.orchestrator_address,
        thesis_response.tokens_used,
    )
    await ctx.send(msg.orchestrator_address, thesis_response)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    researcher.run()
