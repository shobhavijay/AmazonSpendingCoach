"""
tests/test_graph.py
====================
Basic tests for the LangGraph pipeline.

Run without API key (data ingestion only):
    pytest tests/test_graph.py -v

Run full pipeline (requires ANTHROPIC_API_KEY in .env):
    pytest tests/test_graph.py -v -k full
"""

import asyncio
import os
import pytest

from graph.state import AgentState
from graph.nodes.load_data import _load_csv


# ── Test 1: CSV loading (no API key needed) ───────────────────

def test_load_transactions():
    """Verify transaction CSV loads into plain dicts."""
    rows = _load_csv("../files/Order_history_4.csv", "transactions")
    assert len(rows) > 0, "Expected transaction rows"
    assert "Order ID" in rows[0], "Missing 'Order ID' column"
    assert "Total Amount" in rows[0], "Missing 'Total Amount' column"
    assert "Order Date" in rows[0], "Missing 'Order Date' column"
    print(f"  Loaded {len(rows)} transaction rows")


def test_load_returns():
    """Verify returns CSV loads into plain dicts."""
    rows = _load_csv("../files/Refund_Details_4.csv", "returns")
    assert len(rows) > 0, "Expected return rows"
    assert "Order ID" in rows[0], "Missing 'Order ID' column"
    assert "Refund Amount" in rows[0], "Missing 'Refund Amount' column"
    print(f"  Loaded {len(rows)} return rows")


# ── Test 2: State schema ──────────────────────────────────────

def test_state_keys():
    """AgentState must include all required keys."""
    required = {
        "transaction_rows", "returns_rows", "category_map",
        "category_analysis", "impulse_analysis", "forecast_analysis",
        "report_path", "errors",
    }
    annotations = AgentState.__annotations__
    missing = required - set(annotations.keys())
    assert not missing, f"AgentState missing keys: {missing}"


# ── Test 3: Graph builds without error ───────────────────────

def test_graph_builds():
    """Graph should compile without raising exceptions."""
    from graph.builder import build_graph
    graph = build_graph()
    assert graph is not None


# ── Test 4: load_data node (no API key) ──────────────────────

@pytest.mark.asyncio
async def test_load_data_node():
    """load_data node returns transaction and return rows."""
    from graph.nodes.load_data import run

    initial_state = {
        "transaction_rows": [], "returns_rows": [],
        "category_map": {}, "category_analysis": "",
        "impulse_analysis": "", "forecast_analysis": "",
        "report_path": "", "errors": [],
    }
    result = await run(initial_state)
    assert "transaction_rows" in result
    assert "returns_rows" in result
    assert len(result["transaction_rows"]) > 0


# ── Test 5: Full pipeline (requires API key) ──────────────────

@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)
async def test_full_graph():
    """End-to-end: run the full graph and verify report is created."""
    from graph.builder import build_graph
    from config.settings import settings

    graph = build_graph()
    initial_state = {
        "transaction_rows": [], "returns_rows": [],
        "category_map": {}, "category_analysis": "",
        "impulse_analysis": "", "forecast_analysis": "",
        "report_path": "", "errors": [],
    }

    result = await graph.ainvoke(initial_state)

    assert result.get("category_analysis") or result.get("impulse_analysis"), \
        "Expected at least one analysis result"

    if settings.ENABLE_OUTPUT_FILE:
        assert result.get("report_path"), "Expected report_path in result"
        assert os.path.exists(result["report_path"]), \
            f"Report file not found: {result['report_path']}"
