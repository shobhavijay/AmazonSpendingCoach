"""
graph/nodes/write_report.py
============================
Fan-in node — runs after ALL parallel analysis nodes complete.

LangGraph automatically holds this node until every predecessor
(classify_categories, detect_impulse, forecast_spending) has
written its results into state. No explicit synchronization needed.

Writes a markdown report combining all available analysis sections.
Returns report_path so main.py can print the file location.

Feature flag: ENABLE_OUTPUT_FILE
"""

import logging
import os

from config.settings import settings
from graph.state import AgentState

logger = logging.getLogger(__name__)


async def run(state: AgentState) -> dict:
    """
    LangGraph fan-in node — collects all parallel analysis results
    and writes the combined markdown report.

    Receives from state (populated by parallel nodes):
        category_analysis  — from classify_categories
        impulse_analysis   — from detect_impulse
        forecast_analysis  — from forecast_spending (if ON)

    Returns partial state update:
        report_path — path to the written file
    """
    if not settings.ENABLE_OUTPUT_FILE:
        logger.info("SKIPPED — ENABLE_OUTPUT_FILE is OFF.")
        return {"report_path": "", "errors": []}

    lines = ["# Spending Coach Report — v2 (LangGraph)\n"]

    if state.get("category_analysis"):
        lines.append("## Category Analysis\n")
        lines.append(state["category_analysis"])
        lines.append("\n")

    if state.get("impulse_analysis"):
        lines.append("## Impulse Buying Analysis\n")
        lines.append(state["impulse_analysis"])
        lines.append("\n")

    if state.get("forecast_analysis"):
        lines.append("## Spending Forecast\n")
        lines.append(state["forecast_analysis"])
        lines.append("\n")

    if state.get("cart_analysis"):
        lines.append("## Cart Abandonment Analysis\n")
        lines.append(state["cart_analysis"])
        lines.append("\n")

    if not any([state.get("category_analysis"),
                state.get("impulse_analysis"),
                state.get("forecast_analysis"),
                state.get("cart_analysis")]):
        lines.append("_No analysis results. Check that feature flags are ON._\n")

    if state.get("errors"):
        lines.append("## Errors\n")
        for err in state["errors"]:
            lines.append(f"- {err}")
        lines.append("\n")

    report = "\n".join(lines)

    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(settings.OUTPUT_DIR, settings.REPORT_FILENAME)
    with open(filepath, "w") as f:
        f.write(report)

    logger.info("Report saved → %s", filepath)
    return {"report_path": filepath, "errors": []}
