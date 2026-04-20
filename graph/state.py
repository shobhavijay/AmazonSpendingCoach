"""
graph/state.py
==============
Shared state for the LangGraph financial agent.

Every node reads from this state and returns a partial dict
with only the keys it updates. LangGraph merges parallel updates
automatically — no explicit locks or futures needed.

Data flow:
  load_data           → transaction_rows, returns_rows
  classify_categories → category_map, category_analysis   ─┐ parallel
  detect_impulse      → impulse_analysis                   ─┤ parallel
  forecast_spending   → forecast_analysis                  ─┤ parallel (when ON)
  cart_analyzer       → cart_analysis                      ─┘ parallel (when ON)
  write_report        → report_path                          (fan-in, fires last)

The `errors` field uses operator.add as a reducer so concurrent
nodes can each append errors without overwriting one another.
"""

import operator
from typing import Annotated, Dict, List, TypedDict


class AgentState(TypedDict):
    # ── Raw data (set by load_data) ───────────────────────────
    # Plain dicts from csv.DictReader — no LangChain Documents.
    # Keys match CSV column headers exactly, e.g. "Order ID".
    transaction_rows: List[Dict[str, str]]
    returns_rows:     List[Dict[str, str]]

    # ── Intermediate (set by classify_categories) ─────────────
    # Kept in state so future nodes (e.g. forecast) can read it
    # without re-running classification.
    category_map: Dict[str, str]   # order_id → category label

    # ── Analysis outputs (one per parallel node) ──────────────
    category_analysis: str         # markdown coaching narrative
    impulse_analysis:  str         # markdown behavioral narrative
    forecast_analysis: str         # markdown forecast narrative (v2)
    cart_analysis:     str         # markdown cart abandonment narrative

    # ── Final output (set by write_report) ────────────────────
    report_path: str

    # ── Error accumulation ────────────────────────────────────
    # operator.add reducer: parallel nodes each append their
    # errors; LangGraph concatenates the lists automatically.
    errors: Annotated[List[str], operator.add]
