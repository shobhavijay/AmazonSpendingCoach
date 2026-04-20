"""
main.py
=======
Entry point for Financial Agent v2 — LangGraph edition.

Architecture:
  load_data
    ├──► classify_categories ─┐
    ├──► detect_impulse        ├──► write_report
    └──► forecast_spending ───┘  (if ENABLE_SPENDING_FORECAST=true)

Key differences from files/ (LangChain v1):
  - LangGraph StateGraph replaces the manual SpendingCoachAgent orchestrator
  - classify_categories and detect_impulse run in PARALLEL (asyncio)
  - No 30-second cooldown needed — classify_categories batches sequentially
    with a 3 s asyncio.sleep between each batch (non-blocking, so the other
    nodes make progress during the pause); impulse_detector and
    spending_forecast each issue only one LLM call after their Python
    pre-processing, which naturally staggers them from the first batch call
  - CSV loading uses csv.DictReader (no LangChain CSVLoader / unstructured)
  - All nodes are async — use asyncio.run(main())

Run:
    python main.py

Control what runs via .env feature flags — no code changes needed.
"""

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env FIRST — before any LangGraph/LangSmith imports so tracing picks up the keys
load_dotenv(Path(__file__).parent / ".env", override=True)

from config.settings import settings
from graph.builder import build_graph
from logging_config import setup_logging

logger = logging.getLogger(__name__)


async def main():
    setup_logging(settings.LOG_FILENAME)
    logger.info(settings.summary())

    # Build the graph (topology depends on enabled feature flags)
    graph = build_graph()

    # Initial state — all fields must be present even if empty
    initial_state = {
        "transaction_rows": [],
        "returns_rows":     [],
        "category_map":     {},
        "category_analysis": "",
        "impulse_analysis":  "",
        "forecast_analysis": "",
        "cart_analysis":     "",
        "report_path":       "",
        "errors":            [],
    }

    logger.info("Starting LangGraph pipeline (parallel analysis)...")

    # ainvoke runs the graph asynchronously:
    # - load_data runs first
    # - all enabled analysis nodes fire in parallel
    # - write_report waits for all of them, then writes the report
    result = await graph.ainvoke(initial_state)

    if result.get("category_analysis"):
        logger.info("── CATEGORY ANALYSIS ──\n%s", result["category_analysis"])

    if result.get("impulse_analysis"):
        logger.info("── IMPULSE BUYING ANALYSIS ──\n%s", result["impulse_analysis"])

    if result.get("forecast_analysis"):
        logger.info("── SPENDING FORECAST ──\n%s", result["forecast_analysis"])

    if result.get("report_path"):
        logger.info("Full report saved to: %s", result["report_path"])

    if result.get("errors"):
        for err in result["errors"]:
            logger.error("Pipeline error: %s", err)


def cli():
    asyncio.run(main())


if __name__ == "__main__":
    cli()
