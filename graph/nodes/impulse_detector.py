"""
graph/nodes/impulse_detector.py
================================
Node 2 — runs in parallel with classify_categories.

Skip conditions (checked in order):
  1. ENABLE_IMPULSE_DETECTOR flag is OFF → skip
  2. report_v2.md already exists         → skip (idempotency)
  3. No transaction or return rows       → skip

Strategy — Python does all analysis, LLM writes the narrative:
  - Join by Order ID in Python (dict lookup, zero tokens)
  - Compute all derived signals in Python:
      days_held, is_quick_return, is_partial_refund,
      refund_pct, money_lost, order_month, order_dow, reason_bucket
  - Aggregate in Python:
      quick returns, top returned products, monthly/seasonal patterns,
      partial refund analysis, delivery failures, money lost
  - LLM receives only ~500 tokens of pre-computed summary tables

Changes from files/ (LangChain) version:
  - Input: state["transaction_rows"], state["returns_rows"] (List[Dict])
  - Removed _parse_doc_fields() — direct dict access instead
  - chain.invoke() → await chain.ainvoke() (async LLM call)
  - Added report-exists skip check (idempotency)
  - Returns partial state dict instead of plain string

Token budget:
  Python aggregation text ≈ 500 tokens
  Prompt template         ≈  75 tokens
  Total                   ≈ 575 tokens

Feature flag: ENABLE_IMPULSE_DETECTOR
"""

import logging
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate

from config.settings import settings
from graph.state import AgentState

logger = logging.getLogger(__name__)


# ── Return reason taxonomy ────────────────────────────────────
REASON_BUCKETS = {
    "customer return":                   "change_of_mind",
    "item not satisfactory":             "quality_issue",
    "wrong item was sent":               "fulfillment_error",
    "refused to accept delivery":        "delivery_refusal",
    "shipping address is undeliverable": "delivery_failure",
    "account adjustment":                "account_adjustment",
}

MONTH_NAMES = {
    1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May",  6:"Jun",
    7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec",
}

# ── Prompt: narrative from pre-computed summary (~500 tokens) ─
_NARRATIVE_PROMPT = PromptTemplate(
    input_variables=[
        "summary_stats", "quick_returns", "partial_refunds",
        "monthly_pattern", "dow_pattern", "reason_breakdown",
        "top_products", "delivery_issues", "money_lost",
    ],
    template="""You are a behavioral finance coach analyzing Amazon return patterns.

All data below has been pre-computed from the order and refund history.

SUMMARY STATISTICS:
{summary_stats}

QUICK RETURNS (returned within 7 days — strongest impulse signal):
Please just do not use "returned within 7 days" as an impulse signal and relate that buying pattern with other activities like buying for prom, festivals etc.
Make sure you reason deeply for possible other reasons besides just impulse.
For example proms are held in June so buying goes up in May. Indian festivals are typically in September and October and 
Christmas is in December. Offer other plauible reasons besides impulse.
{quick_returns}

PARTIAL REFUNDS (paid more than refunded — money lost):
{partial_refunds}

MONTHLY RETURN PATTERN:
{monthly_pattern}

DAY-OF-WEEK ORDER PATTERN (for returned items):
{dow_pattern}

RETURN REASON BREAKDOWN:
{reason_breakdown}

TOP RETURNED PRODUCTS:
{top_products}

DELIVERY ISSUES (refused / undeliverable):
{delivery_issues}

MONEY LOST ON PARTIAL REFUNDS:
{money_lost}

Write a coaching report with these sections:

### IMPULSE BUYING SIGNALS
- Lead with the quick-return items (≤7 days) as the strongest evidence
- Note any day-of-week or time-of-month patterns that suggest impulsive ordering
- Comment on the ratio of change-of-mind returns vs quality/fulfillment issues
- 

### FINANCIAL IMPACT
- Total amount spent on returned items
- Total refunded vs total paid (money permanently lost)
- Which partial refund cost the most

### BEHAVIORAL PATTERNS
- What the monthly pattern suggests (holiday shopping, seasonal trends, prime days )
- What ordering on certain days of the week suggests
- What the mix of return reasons reveals about buying behavior

### RECOMMENDATIONS
- 5 specific, actionable steps tied directly to the patterns above
- Each recommendation must reference actual data (amounts, dates, products)

Be direct, specific, and use the actual numbers. Avoid generic advice.
""",
)


# ── Helper functions — pure Python, zero tokens ───────────────

def _parse_date(s: str) -> Optional[datetime]:
    """Parse Amazon date strings robustly (with/without milliseconds)."""
    s = s.strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _safe_float(s: str) -> float:
    """Parse float safely, return 0.0 on failure."""
    try:
        return float(s.replace(",", "").replace("$", "").strip())
    except (ValueError, AttributeError):
        return 0.0


def _categorize_reason(raw_reason: str) -> str:
    """Map raw Reversal Reason to a behaviour bucket."""
    return REASON_BUCKETS.get(raw_reason.strip().lower(), "other")


def _join_and_enrich(
    transaction_rows: List[Dict[str, str]],
    returns_rows: List[Dict[str, str]],
) -> Tuple[List[Dict], List[Dict]]:
    """
    Join transactions and returns by Order ID in Python.
    Input is now List[Dict] from csv.DictReader (no Document parsing needed).
    Returns (matched, unmatched).
    """
    tx_lookup: Dict[str, Dict] = {
        row.get("Order ID", "").strip(): row
        for row in transaction_rows
        if row.get("Order ID", "").strip()
    }

    matched   = []
    unmatched = []

    for r in returns_rows:
        oid = r.get("Order ID", "").strip()
        if oid not in tx_lookup:
            unmatched.append(r)
            continue

        tx = tx_lookup[oid]

        order_dt  = _parse_date(tx.get("Order Date", ""))
        refund_dt = _parse_date(r.get("Refund Date", ""))

        days_held = None
        if order_dt and refund_dt:
            days_held = (refund_dt - order_dt).days

        paid   = _safe_float(tx.get("Total Amount", "0"))
        refund = _safe_float(r.get("Refund Amount", "0"))

        is_partial    = paid > 0 and refund < (paid - 0.01)
        money_lost    = round(paid - refund, 2) if is_partial else 0.0
        refund_pct    = round((refund / paid) * 100, 1) if paid > 0 else 100.0
        reason_bucket = _categorize_reason(r.get("Reversal Reason", ""))

        matched.append({
            "order_id":      oid,
            "product":       tx.get("Product Name", "?"),
            "order_date":    tx.get("Order Date", "")[:10],
            "refund_date":   r.get("Refund Date", "")[:10],
            "days_held":     days_held,
            "is_quick":      days_held is not None and days_held <= 7,
            "paid":          paid,
            "refund":        refund,
            "is_partial":    is_partial,
            "money_lost":    money_lost,
            "refund_pct":    refund_pct,
            "reason_raw":    r.get("Reversal Reason", "?"),
            "reason_bucket": reason_bucket,
            "order_month":   order_dt.month if order_dt else None,
            "order_dow":     order_dt.strftime("%A") if order_dt else None,
        })

    return matched, unmatched


# ── Aggregation functions — pure Python, zero tokens ─────────

def _summary_stats(matched: List[Dict], unmatched: List[Dict]) -> str:
    total_pairs    = len(matched)
    total_paid     = sum(m["paid"]       for m in matched)
    total_refunded = sum(m["refund"]     for m in matched)
    total_lost     = sum(m["money_lost"] for m in matched)
    quick_count    = sum(1 for m in matched if m["is_quick"])
    partial_count  = sum(1 for m in matched if m["is_partial"])
    days_list      = [m["days_held"] for m in matched if m["days_held"] is not None]
    avg_days       = round(sum(days_list) / len(days_list), 1) if days_list else 0

    return (
        f"Total matched pairs          : {total_pairs}\n"
        f"Total paid on returned items : ${total_paid:,.2f}\n"
        f"Total refunded               : ${total_refunded:,.2f}\n"
        f"Total money lost (partials)  : ${total_lost:,.2f}\n"
        f"Quick returns (≤7 days)      : {quick_count} ({quick_count/total_pairs*100:.0f}%)\n"
        f"Partial refunds              : {partial_count} ({partial_count/total_pairs*100:.0f}%)\n"
        f"Average days held            : {avg_days} days\n"
        f"Unmatched returns (no tx)    : {len(unmatched)}"
    )


def _quick_returns_table(matched: List[Dict]) -> str:
    quick = sorted([m for m in matched if m["is_quick"]], key=lambda x: x["days_held"])
    if not quick:
        return "No returns within 7 days detected."
    lines = ["Days | Product | Paid | Refund | Reason"]
    for m in quick:
        lines.append(
            f"{m['days_held']}d   | {m['product'][:45]} | "
            f"${m['paid']:.2f} | ${m['refund']:.2f} | {m['reason_raw']}"
        )
    return "\n".join(lines)


def _partial_refunds_table(matched: List[Dict]) -> str:
    partials = sorted([m for m in matched if m["is_partial"]],
                      key=lambda x: x["money_lost"], reverse=True)
    if not partials:
        return "No partial refunds detected."
    lines = ["Product | Paid | Refunded | Lost | Refund%"]
    for m in partials[:10]:
        lines.append(
            f"{m['product'][:40]} | ${m['paid']:.2f} | "
            f"${m['refund']:.2f} | ${m['money_lost']:.2f} | {m['refund_pct']}%"
        )
    return "\n".join(lines)


def _monthly_pattern(matched: List[Dict]) -> str:
    counts = Counter(m["order_month"] for m in matched if m["order_month"])
    if not counts:
        return "No date data available."
    return "\n".join(
        f"{MONTH_NAMES[m]:3s}: {counts[m]:3d}  {'█' * counts[m]}"
        for m in sorted(counts)
    )


def _dow_pattern(matched: List[Dict]) -> str:
    counts    = Counter(m["order_dow"] for m in matched if m["order_dow"])
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    if not counts:
        return "No date data available."
    return "\n".join(
        f"{day:9s}: {counts[day]:3d}  {'█' * counts[day]}"
        for day in day_order if day in counts
    )


def _reason_breakdown(matched: List[Dict]) -> str:
    bucket_counts = Counter(m["reason_bucket"] for m in matched)
    raw_counts    = Counter(m["reason_raw"]    for m in matched)
    lines = ["Bucket                | Count | Raw reason"]
    for bucket, count in bucket_counts.most_common():
        raw = [r for r, _ in raw_counts.most_common()
               if _categorize_reason(r) == bucket]
        lines.append(f"{bucket:21s} | {count:5d} | {', '.join(raw[:3])}")
    return "\n".join(lines)


def _top_products(matched: List[Dict]) -> str:
    spend_by_product: Dict[str, float] = defaultdict(float)
    count_by_product: Dict[str, int]   = defaultdict(int)
    for m in matched:
        key = m["product"][:40]
        spend_by_product[key] += m["paid"]
        count_by_product[key] += 1
    sorted_products = sorted(spend_by_product.items(), key=lambda x: x[1], reverse=True)
    lines = ["Product | Times returned | Total paid"]
    for product, total in sorted_products[:10]:
        lines.append(
            f"{product:40s} | {count_by_product[product]:14d} | ${total:,.2f}"
        )
    return "\n".join(lines)


def _delivery_issues(matched: List[Dict]) -> str:
    issues = [m for m in matched
              if m["reason_bucket"] in ("delivery_refusal", "delivery_failure")]
    if not issues:
        return "No delivery issues detected."
    lines = [f"Total delivery issues: {len(issues)}"]
    for m in issues:
        lines.append(f"  {m['product'][:50]} | ${m['paid']:.2f} | {m['reason_raw']}")
    return "\n".join(lines)


def _money_lost_summary(matched: List[Dict]) -> str:
    total_lost = sum(m["money_lost"] for m in matched)
    if total_lost == 0:
        return "No money lost — all refunds were full."
    worst = sorted(matched, key=lambda x: x["money_lost"], reverse=True)[:5]
    lines = [f"Total money lost across all partial refunds: ${total_lost:,.2f}\n",
             "Top 5 costliest partial refunds:"]
    for m in worst:
        lines.append(
            f"  ${m['money_lost']:.2f} lost — {m['product'][:50]} "
            f"(paid ${m['paid']:.2f}, refunded ${m['refund']:.2f})"
        )
    return "\n".join(lines)


# ── Node entrypoint ───────────────────────────────────────────

async def run(state: AgentState) -> dict:
    """
    LangGraph node — detects impulse buying patterns and generates narrative.
    Runs in parallel with classify_categories after load_data completes.

    Skip conditions:
      - Flag is OFF
      - report_v2.md already exists (idempotency — analysis already done)
      - No data in state

    Returns partial state update:
        impulse_analysis — markdown behavioral coaching narrative
    """
    # ── Skip check 1: feature flag ─────────────────────────────
    if not settings.ENABLE_IMPULSE_DETECTOR:
        logger.info("SKIPPED — ENABLE_IMPULSE_DETECTOR is OFF.")
        return {"impulse_analysis": "", "errors": []}

    # ── Skip check 2: report already exists (idempotency) ─────
    report_path = os.path.join(settings.OUTPUT_DIR, settings.REPORT_FILENAME)
    if os.path.exists(report_path):
        logger.info("SKIPPED — report already exists: %s", report_path)
        logger.info("Delete the report file to re-run this analysis.")
        return {"impulse_analysis": "", "errors": []}

    # ── Skip check 3: no data ─────────────────────────────────
    transaction_rows = state.get("transaction_rows", [])
    returns_rows     = state.get("returns_rows", [])

    if not transaction_rows:
        logger.warning("No transaction rows in state.")
        return {"impulse_analysis": "", "errors": []}

    if not returns_rows:
        logger.warning("No returns rows in state.")
        return {"impulse_analysis": "", "errors": []}

    # ── Step 1: join and enrich in Python (zero tokens) ───────
    logger.info(
        "Joining %d transactions with %d returns...",
        len(transaction_rows), len(returns_rows),
    )
    matched, unmatched = _join_and_enrich(transaction_rows, returns_rows)
    logger.info("Matched: %d | Unmatched: %d", len(matched), len(unmatched))

    if not matched:
        return {"impulse_analysis": "No matching transaction-return pairs found.",
                "errors": []}

    # ── Step 2: compute all insights in Python (zero tokens) ──
    logger.info("Computing insights in Python...")
    payload = {
        "summary_stats":    _summary_stats(matched, unmatched),
        "quick_returns":    _quick_returns_table(matched),
        "partial_refunds":  _partial_refunds_table(matched),
        "monthly_pattern":  _monthly_pattern(matched),
        "dow_pattern":      _dow_pattern(matched),
        "reason_breakdown": _reason_breakdown(matched),
        "top_products":     _top_products(matched),
        "delivery_issues":  _delivery_issues(matched),
        "money_lost":       _money_lost_summary(matched),
    }

    est_tokens = sum(len(v) for v in payload.values()) // 4
    logger.info("Sending ~%d tokens to LLM...", est_tokens)

    # ── Step 3: single async LLM call for narrative ───────────
    llm = ChatAnthropic(
        model=settings.ANTHROPIC_MODEL,
        temperature=0,
        api_key=settings.ANTHROPIC_API_KEY,
    )
    chain  = _NARRATIVE_PROMPT | llm
    result = await chain.ainvoke(payload)

    logger.info("Done.")
    return {"impulse_analysis": result.content, "errors": []}
