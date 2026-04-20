"""
graph/nodes/load_data.py
========================
Node 0 — always runs first.

Loads both CSV files into plain Python dicts using csv.DictReader.
No LangChain Documents, no unstructured library — just stdlib csv.

Each row becomes a Dict[str, str] with keys matching CSV column headers:
  transactions: "Order ID", "Product Name", "Total Amount", "Order Date", ...
  returns:      "Order ID", "Refund Amount", "Refund Date", "Reversal Reason", ...

Downstream nodes read from:
  state["transaction_rows"]  — List[Dict[str, str]]
  state["returns_rows"]      — List[Dict[str, str]]

Feature flag: ENABLE_DATA_INGESTION
"""

import csv
import logging
import os
from typing import Dict, List

from config.settings import settings
from graph.state import AgentState

logger = logging.getLogger(__name__)


def _load_csv(filepath: str, label: str) -> List[Dict[str, str]]:
    """Load a CSV file into a list of row dicts. Returns [] on missing file."""
    if not os.path.exists(filepath):
        logger.warning("File not found: %s", filepath)
        return []

    rows: List[Dict[str, str]] = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from all keys and values
            rows.append({k.strip(): v.strip() for k, v in row.items()})

    logger.info("%s: %d records loaded ← %s", label, len(rows), filepath)
    return rows


async def run(state: AgentState) -> dict:
    """
    Load transaction and return CSV files into state.
    Returns partial state update with transaction_rows and returns_rows.
    """
    if not settings.ENABLE_DATA_INGESTION:
        logger.info("SKIPPED — ENABLE_DATA_INGESTION is OFF.")
        return {"transaction_rows": [], "returns_rows": [], "errors": []}

    logger.info("Loading data files...")
    transaction_rows = _load_csv(settings.TRANSACTIONS_CSV, "transactions")
    returns_rows     = _load_csv(settings.RETURNS_CSV,      "returns")

    errors = []
    if not transaction_rows:
        errors.append(f"LoadData: no transactions loaded from {settings.TRANSACTIONS_CSV}")
    if not returns_rows:
        errors.append(f"LoadData: no returns loaded from {settings.RETURNS_CSV}")

    logger.info("Done — %d transactions, %d returns.", len(transaction_rows), len(returns_rows))
    return {
        "transaction_rows": transaction_rows,
        "returns_rows":     returns_rows,
        "errors":           errors,
    }
