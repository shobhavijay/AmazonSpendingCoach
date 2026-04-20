"""
graph/nodes/cart_analyzer.py
=============================
Analyzes Amazon Cart History — the "opposite" of impulse_detector.

Instead of looking at what was returned, this asks:
  For every item that sat in cart, was it eventually purchased or abandoned?

Join strategy (pure Python, zero tokens):
  - Cart History CSV  → keyed by ASIN
  - transaction_rows  → keyed by ASIN (already in state from load_data)
  - If ASIN appears in orders with order_date >= cart add_date → PURCHASED
  - Otherwise → PURGED (abandoned in cart)

Per-category Python analysis (zero tokens):
  - Total items, purchased count, purged count, purchase %
  - Avg days from cart-add to purchase (purchased items only)
  - Median days in cart across all items (purged items: days from add to today)

LLM receives only ~350 tokens of pre-computed compact tables.
LLM writes the narrative only — no raw data, no product names.

Token budget:
  summary stats       ≈  80 tokens
  category table      ≈ 150 tokens
  top abandoned       ≈  50 tokens
  top purchased       ≈  50 tokens
  prompt template     ≈  80 tokens
  ─────────────────────────────────
  Total               ≈ 410 tokens

Feature flag: ENABLE_CART_ANALYZER
CSV path:     CART_CSV
"""

import csv
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate

from config.settings import settings
from graph.state import AgentState

logger = logging.getLogger(__name__)


# ── Keyword-based category classifier (pure Python, zero tokens) ──────────────
# Maps lowercase keywords found in product name → category label.
# Checked in order; first match wins. "Other" is the fallback.
CATEGORY_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("Electronics",     ["usb", "cable", "charger", "battery", "keyboard", "mouse",
                          "monitor", "headphone", "speaker", "camera", "laptop",
                          "tablet", "phone", "adapter", "hdmi", "hub", "ssd",
                          "hard drive", "router", "microphone", "webcam", "earbu"]),
    ("Books & Media",   ["book", "guide", "manual", "novel", "journal", "paperback",
                          "hardcover", "audiobook", "dvd", "blu-ray", "magazine"]),
    ("Clothing",        ["shirt", "pant", "dress", "sock", "shoe", "boot", "jacket",
                          "coat", "mask", "glove", "hat", "belt", "jeans", "legging",
                          "sweater", "hoodie", "underwear", "bra", "brief"]),
    ("Home & Kitchen",  ["kitchen", "cook", "pan", "pot", "knife", "cup", "plate",
                          "bowl", "mug", "towel", "curtain", "pillow", "sheet",
                          "blanket", "lamp", "shelf", "organizer", "container",
                          "storage", "bin", "basket", "hook", "rack"]),
    ("Health & Beauty", ["vitamin", "supplement", "cream", "lotion", "shampoo",
                          "soap", "serum", "moisturizer", "sunscreen", "toothbrush",
                          "dental", "medicine", "bandage", "first aid", "protein",
                          "probiotic", "collagen", "omega"]),
    ("Office & Stationery", ["pen", "ink", "cartridge", "paper", "notebook",
                              "binder", "folder", "staple", "tape", "scissors",
                              "desk", "chair", "lamp", "whiteboard", "marker"]),
    ("Toys & Games",    ["toy", "game", "puzzle", "lego", "doll", "action figure",
                          "board game", "card game", "rc ", "remote control",
                          "playset", "craft kit"]),
    ("Pet Supplies",    ["dog", "cat", "pet", "leash", "collar", "crate", "litter",
                          "aquarium", "fish", "bird", "treat", "paw"]),
    ("Sports & Outdoors", ["yoga", "gym", "fitness", "sport", "exercise", "dumbbell",
                            "weight", "mat", "band", "resistance", "treadmill",
                            "bicycle", "hiking", "camping", "tent", "backpack"]),
    ("Automotive",      ["car", "auto", "vehicle", "tire", "wheel", "motor", "oil",
                          "brake", "wiper", "seat cover", "dashboard"]),
    ("Garden & Tools",  ["garden", "plant", "seed", "soil", "fertilizer", "pot",
                          "planter", "shovel", "drill", "screwdriver", "wrench",
                          "hammer", "tool", "paint", "brush"]),
]

def _classify_product(name: str) -> str:
    lower = name.lower()
    for category, keywords in CATEGORY_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return category
    return "Other"


# ── Date parsing ───────────────────────────────────────────────────────────────

def _parse_dt(s: str) -> Optional[datetime]:
    s = s.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


# ── Load cart CSV ──────────────────────────────────────────────────────────────

def _load_cart(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# ── Core analysis — pure Python, zero tokens ──────────────────────────────────

def _analyze(
    cart_rows: List[Dict[str, str]],
    transaction_rows: List[Dict[str, str]],
) -> Tuple[Dict, List[Dict]]:
    """
    Cross-reference cart items with order history by ASIN.
    Returns (stats_dict, enriched_items).

    Purchase rule: ASIN appears in orders AND earliest order_date >= cart add_date
                   (allows for coincidental prior purchase of same ASIN)
    """
    # Build ASIN → list of order dates from transaction history
    asin_order_dates: Dict[str, List[datetime]] = defaultdict(list)
    for row in transaction_rows:
        asin = row.get("ASIN", "").strip()
        dt   = _parse_dt(row.get("Order Date", ""))
        if asin and dt:
            asin_order_dates[asin].append(dt)

    today = datetime.now(tz=timezone.utc)
    items = []

    for row in cart_rows:
        asin       = row.get("ASIN", "").strip()
        name       = row.get("Product Name", "Unknown")
        added_dt   = _parse_dt(row.get("Date Added to Cart", ""))
        category   = _classify_product(name)

        # Determine purchased vs purged
        purchased      = False
        days_to_buy    = None
        days_in_cart   = None

        if asin in asin_order_dates and added_dt:
            # Find earliest order on or after the cart-add date
            later_orders = [d for d in asin_order_dates[asin] if d >= added_dt]
            if later_orders:
                purchased   = True
                earliest    = min(later_orders)
                days_to_buy = (earliest - added_dt).days

        if added_dt:
            days_in_cart = (today - added_dt).days

        items.append({
            "asin":         asin,
            "name":         name[:60],
            "category":     category,
            "added_dt":     added_dt,
            "purchased":    purchased,
            "days_to_buy":  days_to_buy,   # None if purged
            "days_in_cart": days_in_cart,  # days since added (purged only meaningful)
        })

    return items


# ── Aggregation helpers — pure Python, zero tokens ────────────────────────────

def _category_table(items: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Build per-category stats. Returns (markdown_table, raw_stats_list).
    """
    cat: Dict[str, Dict] = defaultdict(lambda: {
        "total": 0, "purchased": 0,
        "days_to_buy": [], "days_in_cart": [],
    })

    for item in items:
        c = item["category"]
        cat[c]["total"] += 1
        if item["purchased"]:
            cat[c]["purchased"] += 1
            if item["days_to_buy"] is not None:
                cat[c]["days_to_buy"].append(item["days_to_buy"])
        if item["days_in_cart"] is not None:
            cat[c]["days_in_cart"].append(item["days_in_cart"])

    rows = []
    for c, d in sorted(cat.items(), key=lambda x: x[1]["total"], reverse=True):
        total     = d["total"]
        purchased = d["purchased"]
        purged    = total - purchased
        pct       = purchased / total * 100
        avg_days  = (round(sum(d["days_to_buy"]) / len(d["days_to_buy"]))
                     if d["days_to_buy"] else None)
        avg_wait  = (round(sum(d["days_in_cart"]) / len(d["days_in_cart"]))
                     if d["days_in_cart"] else None)
        rows.append({
            "category": c, "total": total,
            "purchased": purchased, "purged": purged,
            "pct": pct, "avg_days_to_buy": avg_days,
            "avg_days_in_cart": avg_wait,
        })

    lines = ["Category              | Total | Bought | Purged | Buy% | AvgDaysToBuy"]
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        avg = f"{r['avg_days_to_buy']}d" if r["avg_days_to_buy"] is not None else "—"
        lines.append(
            f"{r['category']:<22}| {r['total']:5d} | {r['purchased']:6d} | "
            f"{r['purged']:6d} | {r['pct']:4.0f}% | {avg}"
        )
    return "\n".join(lines), rows


def _top_abandoned(cat_rows: List[Dict], n: int = 5) -> str:
    """Top N categories by purge count (most items left in cart)."""
    ranked = sorted(cat_rows, key=lambda x: x["purged"], reverse=True)[:n]
    lines  = []
    for r in ranked:
        lines.append(
            f"  {r['category']}: {r['purged']} items abandoned "
            f"({100 - r['pct']:.0f}% of {r['total']} cart items)"
        )
    return "\n".join(lines)


def _top_purchased(cat_rows: List[Dict], n: int = 5) -> str:
    """Top N categories by purchase rate (highest conversion)."""
    ranked = sorted(
        [r for r in cat_rows if r["total"] >= 2],  # min 2 items for meaningful %
        key=lambda x: x["pct"], reverse=True,
    )[:n]
    lines  = []
    for r in ranked:
        avg = f", avg {r['avg_days_to_buy']}d to buy" if r["avg_days_to_buy"] else ""
        lines.append(
            f"  {r['category']}: {r['pct']:.0f}% purchase rate "
            f"({r['purchased']}/{r['total']}){avg}"
        )
    return "\n".join(lines)


def _summary_stats(items: List[Dict], cat_rows: List[Dict]) -> str:
    total     = len(items)
    purchased = sum(1 for i in items if i["purchased"])
    purged    = total - purchased
    dtb       = [i["days_to_buy"] for i in items
                 if i["purchased"] and i["days_to_buy"] is not None]
    avg_dtb   = round(sum(dtb) / len(dtb)) if dtb else None
    dic       = [i["days_in_cart"] for i in items
                 if not i["purchased"] and i["days_in_cart"] is not None]
    avg_dic   = round(sum(dic) / len(dic)) if dic else None

    lines = [
        f"Total cart items     : {total}",
        f"Eventually purchased : {purchased} ({purchased/total*100:.1f}%)",
        f"Never purchased      : {purged} ({purged/total*100:.1f}%)",
        f"Avg days cart→buy    : {avg_dtb}d" if avg_dtb else "Avg days cart→buy    : n/a",
        f"Avg days abandoned   : {avg_dic}d" if avg_dic else "Avg days abandoned   : n/a",
        f"Categories tracked   : {len(cat_rows)}",
    ]
    return "\n".join(lines)


# ── Prompt (~80 template tokens + ~350 data tokens) ───────────────────────────

_NARRATIVE_PROMPT = PromptTemplate(
    input_variables=[
        "summary_stats", "category_table",
        "top_abandoned", "top_purchased",
    ],
    template="""You are a personal finance coach analyzing Amazon cart behavior.

All numbers below are pre-computed. Do not invent figures.

OVERALL SUMMARY:
{summary_stats}

CATEGORY BREAKDOWN (sorted by total cart items):
{category_table}

TOP ABANDONED CATEGORIES (most items left without buying):
{top_abandoned}

TOP PURCHASED CATEGORIES (highest conversion rate):
{top_purchased}

Write a concise coaching report with these sections:

### CART ABANDONMENT PATTERNS
- Which categories accumulate items that never get bought and what that reveals about browsing vs intent
- Note the overall abandonment rate and what it suggests

### PURCHASE INTENT ANALYSIS
- Which categories show strong follow-through (high buy rate)
- Average days from cart-add to purchase — what does the delay reveal

### ACTIONABLE RECOMMENDATIONS
- 4 specific steps tied to actual category names and percentages above
- Focus on reducing cart bloat and converting high-value abandoned categories

Use the actual numbers. Be direct and concise.
""",
)


# ── Node entrypoint ────────────────────────────────────────────────────────────

async def run(state: AgentState) -> dict:
    """
    LangGraph node — analyzes cart abandonment vs purchase patterns by category.
    Runs in parallel with other analysis nodes after load_data.

    Returns partial state update:
        cart_analysis — markdown coaching narrative
    """
    if not settings.ENABLE_CART_ANALYZER:
        logger.info("SKIPPED — ENABLE_CART_ANALYZER is OFF.")
        return {"cart_analysis": "", "errors": []}

    # ── Load cart CSV ──────────────────────────────────────────
    cart_path = settings.CART_CSV
    if not os.path.exists(cart_path):
        msg = f"Cart CSV not found: {cart_path}"
        logger.error(msg)
        return {"cart_analysis": "", "errors": [msg]}

    cart_rows        = _load_cart(cart_path)
    transaction_rows = state.get("transaction_rows", [])

    if not cart_rows:
        logger.warning("Cart CSV is empty.")
        return {"cart_analysis": "", "errors": []}

    logger.info("Loaded %d cart rows. Cross-referencing with %d order rows...",
                len(cart_rows), len(transaction_rows))

    # ── Python analysis — zero tokens ─────────────────────────
    items            = _analyze(cart_rows, transaction_rows)
    category_table_str, cat_rows = _category_table(items)

    payload = {
        "summary_stats":   _summary_stats(items, cat_rows),
        "category_table":  category_table_str,
        "top_abandoned":   _top_abandoned(cat_rows),
        "top_purchased":   _top_purchased(cat_rows),
    }

    est_tokens = sum(len(v) for v in payload.values()) // 4
    logger.info("Sending ~%d tokens to LLM for narrative...", est_tokens)

    # ── Single LLM call for narrative ─────────────────────────
    llm    = ChatAnthropic(
        model=settings.ANTHROPIC_MODEL,
        temperature=0,
        api_key=settings.ANTHROPIC_API_KEY,
    )
    result = await (_NARRATIVE_PROMPT | llm).ainvoke(payload)

    logger.info("Done.")
    return {"cart_analysis": result.content, "errors": []}
