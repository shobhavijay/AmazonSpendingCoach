"""
graph/nodes/category_classifier.py
====================================
Node 1 — runs in parallel with detect_impulse.

Flow:
  IF categorized CSV exists:
      Load category_map from it (zero LLM tokens)
      Go to Step 3 (narrative)
  ELSE:
      Step 1 — LLM classifies orders in batches → category_map
      Step 2 — Write category_map back to CSV (idempotency)

Step 3 always runs:
  - _compute_totals()  → Python aggregation, zero tokens
  - _build_tables()    → Python markdown tables, zero tokens
  - narrative_chain    → LLM receives only ~43 tokens of pre-built
                         tables and writes the coaching report

Changes from files/ (LangChain) version:
  - Input: state["transaction_rows"] (List[Dict]) — no LangChain Documents
  - Removed _parse_doc_fields() — direct dict access instead
  - time.sleep → asyncio.sleep (non-blocking, allows impulse_detector
    to run concurrently during the 3s inter-batch pauses)
  - chain.invoke() → await chain.ainvoke() (async LLM call)
  - Returns partial state dict instead of plain string

Feature flag: ENABLE_CATEGORY_CLASSIFIER
"""

import asyncio
import csv
import logging
import os
from collections import defaultdict
from typing import Dict, List

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate

from config.settings import settings
from graph.state import AgentState

logger = logging.getLogger(__name__)

# ── Rate limit tuning ─────────────────────────────────────────
BATCH_SIZE       = 20
BATCH_SLEEP_SECS = 3   # asyncio.sleep → non-blocking, other nodes run during pause

# ── Categories ────────────────────────────────────────────────
CATEGORIES = [
    "food_and_grocery",
    "health_and_wellness",
    "fitness_and_sports",
    "electronics_and_tech",
    "home_and_kitchen",
    "clothing_and_apparel",
    "books_and_education",
    "beauty_and_personal_care",
    "toys_and_baby",
    "office_and_supplies",
    "garden_and_outdoor",
    "other",
]

# ── Prompt 1: per-batch classification ────────────────────────
_CLASSIFY_PROMPT = PromptTemplate(
    input_variables=["transactions", "categories"],
    template="""You are a financial analyst classifying Amazon orders.

Available categories: {categories}

Category rules:
- "food_and_grocery"         → food, snacks, beverages, supplements, protein powder, vitamins
- "health_and_wellness"      → medical supplies, first aid, health monitors, medications
- "fitness_and_sports"       → gym gear, running gear, sports nutrition, workout equipment
- "electronics_and_tech"     → gadgets, cables, chargers, projectors, batteries, tech accessories
- "home_and_kitchen"         → appliances, cookware, furniture, cleaning, organizers, filters
- "clothing_and_apparel"     → clothes, shoes, gloves, swimwear, accessories
- "books_and_education"      → books, textbooks, workbooks, flash cards, courses
- "beauty_and_personal_care" → cosmetics, skincare, hair care, toothbrushes, lipstick
- "toys_and_baby"            → baby items, toys, children's products
- "office_and_supplies"      → pens, erasers, paper, desk accessories
- "garden_and_outdoor"       → outdoor gear, garden tools, patio items
- "other"                    → anything that does not fit above

ORDERS:
{transactions}

Reply with ONLY one line per order in this exact format — no headers, no extra text:
ORDER_ID | category

Example:
107-7709656-0661065 | home_and_kitchen
103-6615380-6441024 | books_and_education
""",
)

# ── Prompt 2: narrative from pre-computed numbers ─────────────
# LLM receives only ~43 tokens of pre-aggregated table strings.
_NARRATIVE_PROMPT = PromptTemplate(
    input_variables=["totals_table", "top5_table",
                     "overall_total", "order_count", "date_range"],
    template="""You are a personal finance coach reviewing an Amazon spending summary.

Focus on 
SPENDING DATA (pre-computed):

Date range   : {date_range}
Total orders : {order_count}
Overall total: {overall_total}

CATEGORY TOTALS:
{totals_table}

TOP 5 CATEGORIES:
{top5_table}

Write a coaching report with:

### CATEGORY TOTALS
(reproduce the table above exactly as a markdown table)

### TOP 5 CATEGORIES
(reproduce the top 5 table above exactly as a markdown table with rank medals)

### OVERALL TOTAL
(one line with the total and date range)

### COACHING INSIGHTS
(3-4 sentences of honest, specific observations about this spending pattern)
""",
)


# ── Pure Python helpers — zero LLM tokens ────────────────────

def _parse_classifications(llm_output: str) -> Dict[str, str]:
    """Parse ORDER_ID | category lines → {order_id: category}."""
    result = {}
    for line in llm_output.strip().split("\n"):
        line = line.strip()
        if "|" in line:
            parts = line.split("|")
            if len(parts) == 2:
                order_id = parts[0].strip()
                category = parts[1].strip().lower()
                result[order_id] = category if category in CATEGORIES else "other"
    return result


def _load_category_map_from_csv(categorized_csv: str) -> Dict[str, str]:
    """Load order_id → category from existing categorized CSV. Zero LLM tokens."""
    category_map = {}
    with open(categorized_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            order_id = row.get("Order ID", "").strip()
            category = row.get("category", "other").strip()
            if order_id:
                category_map[order_id] = category
    return category_map


def _write_category_to_csv(
    input_path: str,
    output_path: str,
    category_map: Dict[str, str],
) -> int:
    """Write category column to CSV. Returns number of rows updated."""
    updated = 0
    with open(input_path, newline="", encoding="utf-8") as infile, \
         open(output_path, "w", newline="", encoding="utf-8") as outfile:
        reader     = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames) + ["category"]
        writer     = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            order_id = row.get("Order ID", "").strip()
            row["category"] = category_map.get(order_id, "other")
            if order_id in category_map:
                updated += 1
            writer.writerow(row)
    return updated


def _compute_totals(
    transaction_rows: List[Dict[str, str]],
    category_map: Dict[str, str],
) -> Dict:
    """
    Aggregate spend and count per category in Python.
    Zero LLM tokens — pure dict arithmetic.
    Input is now List[Dict] from csv.DictReader (no Document parsing needed).
    """
    spend_by_cat: Dict[str, float] = defaultdict(float)
    count_by_cat: Dict[str, int]   = defaultdict(int)
    dates = []

    for row in transaction_rows:
        order_id = row.get("Order ID", "").strip()
        category = category_map.get(order_id, "other")

        try:
            amount = float(row.get("Total Amount", "0").replace(",", ""))
        except ValueError:
            amount = 0.0

        spend_by_cat[category] += amount
        count_by_cat[category] += 1

        date = row.get("Order Date", "")[:10]
        if date:
            dates.append(date)

    return {
        "sorted_cats":   sorted(spend_by_cat.items(),
                                key=lambda x: x[1], reverse=True),
        "count_by_cat":  dict(count_by_cat),
        "overall_total": sum(spend_by_cat.values()),
        "date_range":    f"{min(dates)} to {max(dates)}" if dates else "unknown",
        "order_count":   len(transaction_rows),
    }


def _build_tables(totals: Dict) -> Dict[str, str]:
    """Build markdown table strings from pre-computed totals. Zero LLM tokens."""
    sorted_cats  = totals["sorted_cats"]
    count_by_cat = totals["count_by_cat"]

    rows = ["| Category | Total Spend | # of Line Items |",
            "|---|---|---|"]
    for cat, total in sorted_cats:
        rows.append(f"| {cat} | ${total:,.2f} | {count_by_cat.get(cat, 0)} |")
    totals_table = "\n".join(rows)

    medals    = ["🥇 1", "🥈 2", "🥉 3", "4", "5"]
    top5_rows = ["| Rank | Category | Total Spend |", "|---|---|---|"]
    for i, (cat, total) in enumerate(sorted_cats[:5]):
        top5_rows.append(f"| {medals[i]} | {cat} | ${total:,.2f} |")
    top5_table = "\n".join(top5_rows)

    return {"totals_table": totals_table, "top5_table": top5_table}


# ── Node entrypoint ───────────────────────────────────────────

async def run(state: AgentState) -> dict:
    """
    LangGraph node — classifies orders and generates a category narrative.
    Runs in parallel with detect_impulse after load_data completes.

    Returns partial state update:
        category_map      — used by future forecast node
        category_analysis — markdown coaching narrative
    """
    if not settings.ENABLE_CATEGORY_CLASSIFIER:
        logger.info("SKIPPED — ENABLE_CATEGORY_CLASSIFIER is OFF.")
        return {"category_map": {}, "category_analysis": "", "errors": []}

    transaction_rows = state.get("transaction_rows", [])
    if not transaction_rows:
        logger.warning("No transaction rows in state.")
        return {"category_map": {}, "category_analysis": "", "errors": []}

    # Lazy-init LLM inside node (not at import time)
    llm = ChatAnthropic(
        model=settings.ANTHROPIC_MODEL,
        temperature=0,
        api_key=settings.ANTHROPIC_API_KEY,
    )
    classify_chain  = _CLASSIFY_PROMPT  | llm
    narrative_chain = _NARRATIVE_PROMPT | llm

    categorized_csv = settings.CATEGORIZED_CSV

    # ── Conditional: skip Step 1+2 if categorized CSV exists ──
    if os.path.exists(categorized_csv):
        logger.info("Found existing categorized CSV: %s", categorized_csv)
        logger.info("Skipping LLM classification (already cached).")
        category_map = _load_category_map_from_csv(categorized_csv)
        logger.info("Loaded %d categories from CSV.", len(category_map))

    else:
        # ── Step 1: LLM classifies orders in batches ──────────
        total   = len(transaction_rows)
        batches = [
            transaction_rows[i:i + BATCH_SIZE]
            for i in range(0, total, BATCH_SIZE)
        ]
        logger.info(
            "Categorized CSV not found. Classifying %d orders in %d batches of %d...",
            total, len(batches), BATCH_SIZE,
        )

        category_map: Dict[str, str] = {}

        for i, batch in enumerate(batches, 1):
            logger.debug("Batch %d/%d...", i, len(batches))

            batch_lines = []
            for row in batch:
                batch_lines.append(
                    f"{row.get('Order ID', '?')} | "
                    f"{row.get('Product Name', '?')[:60]} | "
                    f"${row.get('Total Amount', '0')} | "
                    f"{row.get('Order Date', '')[:10]}"
                )

            max_retries = 3
            llm_output  = ""
            for attempt in range(1, max_retries + 1):
                try:
                    result     = await classify_chain.ainvoke({
                        "transactions": "\n".join(batch_lines),
                        "categories":   ", ".join(CATEGORIES),
                    })
                    llm_output = result.content
                    break
                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        wait = BATCH_SLEEP_SECS * (attempt * 2)
                        logger.warning(
                            "Rate limit hit. Waiting %ds (attempt %d/%d)...",
                            wait, attempt, max_retries,
                        )
                        await asyncio.sleep(wait)
                        if attempt == max_retries:
                            logger.error(
                                "Batch %d failed after %d retries. Skipping.",
                                i, max_retries,
                            )
                    else:
                        raise

            category_map.update(_parse_classifications(llm_output))

            # asyncio.sleep is non-blocking — impulse_detector runs during this pause
            if i < len(batches):
                await asyncio.sleep(BATCH_SLEEP_SECS)

        logger.info("Classified %d orders.", len(category_map))

        # ── Step 2: write category column to CSV (idempotency) ─
        input_csv = settings.TRANSACTIONS_CSV
        if os.path.exists(input_csv):
            updated = _write_category_to_csv(input_csv, categorized_csv, category_map)
            logger.info("CSV written → %s (%d rows categorized)", categorized_csv, updated)
        else:
            logger.warning("%s not found, skipping CSV write.", input_csv)

    # ── Step 3: compute totals in Python (zero tokens) ────────
    logger.info("Computing totals in Python...")
    totals = _compute_totals(transaction_rows, category_map)
    tables = _build_tables(totals)

    logger.info("Overall total : $%s", f"{totals['overall_total']:,.2f}")
    logger.info("Date range    : %s", totals["date_range"])
    logger.info("Orders        : %d", totals["order_count"])

    # ── Step 4: single LLM call for narrative (~43 tokens) ────
    logger.info("Generating narrative report (~43 tokens)...")
    result = await narrative_chain.ainvoke({
        "totals_table":  tables["totals_table"],
        "top5_table":    tables["top5_table"],
        "overall_total": f"${totals['overall_total']:,.2f}",
        "order_count":   str(totals["order_count"]),
        "date_range":    totals["date_range"],
    })

    logger.info("Done.")
    return {
        "category_map":      category_map,
        "category_analysis": result.content,
        "errors":            [],
    }
