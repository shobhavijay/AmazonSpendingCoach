"""
graph/builder.py
================
Constructs and compiles the LangGraph StateGraph.

Topology:

  START
    │
  load_data
    ├──► classify_categories ─┐
    ├──► detect_impulse        ├──► write_report ──► END
    └──► forecast_spending ───┘
         (if ENABLE_SPENDING_FORECAST=true)

Key LangGraph behaviors used here:

  Fan-out:  Multiple edges FROM load_data fire all target nodes
            concurrently when graph.ainvoke() is used.

  Fan-in:   LangGraph automatically waits for ALL predecessors of
            write_report to complete before running it. No explicit
            barriers or futures needed — the graph topology encodes
            the synchronization.

  State merge: Each parallel node returns a partial dict. LangGraph
            merges them into the shared AgentState. The `errors`
            field uses operator.add to concat lists from parallel nodes.

Adding a new analysis node:
  1. Create graph/nodes/new_node.py with:  async def run(state) -> dict
  2. Add ENABLE_NEW_NODE flag to config/settings.py
  3. Add 4 lines below (add_node + add_edge ×2 + append to list)
  That's it — parallelism, state merging, and fan-in are automatic.
"""

import logging

from langgraph.graph import END, START, StateGraph

from config.settings import settings

logger = logging.getLogger(__name__)
from graph.nodes import (
    cart_analyzer,
    category_classifier,
    detect_impulse,
    load_data,
    spending_forecast,
    write_report,
)
from graph.state import AgentState


def build_graph():
    """
    Build and compile the spending coach StateGraph.
    Feature flags determine which analysis nodes are included.
    Disabled nodes are excluded from the graph entirely — not just skipped.
    """
    builder = StateGraph(AgentState)

    # ── Always-on nodes ────────────────────────────────────────
    builder.add_node("load_data",    load_data.run)
    builder.add_node("write_report", write_report.run)
    builder.add_edge(START, "load_data")
    builder.add_edge("write_report", END)

    # ── Analysis nodes (parallel fan-out from load_data) ──────
    # Each enabled node gets two edges: load_data→node and node→write_report.
    # LangGraph fires all in parallel and holds write_report until all complete.
    analysis_nodes: list[str] = []

    if settings.ENABLE_CATEGORY_CLASSIFIER:
        builder.add_node("classify_categories", category_classifier.run)
        builder.add_edge("load_data",           "classify_categories")
        builder.add_edge("classify_categories",  "write_report")
        analysis_nodes.append("classify_categories")
        logger.info("+ classify_categories (parallel)")

    if settings.ENABLE_IMPULSE_DETECTOR:
        builder.add_node("detect_impulse", detect_impulse.run)
        builder.add_edge("load_data",      "detect_impulse")
        builder.add_edge("detect_impulse", "write_report")
        analysis_nodes.append("detect_impulse")
        logger.info("+ detect_impulse (parallel)")

    if settings.ENABLE_CART_ANALYZER:
        builder.add_node("cart_analyzer", cart_analyzer.run)
        builder.add_edge("load_data",     "cart_analyzer")
        builder.add_edge("cart_analyzer", "write_report")
        analysis_nodes.append("cart_analyzer")
        logger.info("+ cart_analyzer (parallel)")

    if settings.ENABLE_SPENDING_FORECAST:
        builder.add_node("forecast_spending", spending_forecast.run)
        builder.add_edge("load_data",         "forecast_spending")
        builder.add_edge("forecast_spending",  "write_report")
        analysis_nodes.append("forecast_spending")
        logger.info("+ forecast_spending (parallel)")

    # ── Fallback: no analysis flags ON ─────────────────────────
    if not analysis_nodes:
        logger.warning(
            "No analysis nodes enabled. Report will be empty. Check feature flags."
        )
        builder.add_edge("load_data", "write_report")

    logger.info(
        "Graph compiled — %d analysis node(s) run in parallel.", len(analysis_nodes)
    )
    return builder.compile()
