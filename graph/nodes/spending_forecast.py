"""
graph/nodes/spending_forecast.py
==================================
Node 3 — runs in parallel with classify_categories and detect_impulse.

Controlled by two env vars:
  ENABLE_SPENDING_FORECAST=true       — activates the node
  FORECAST_MODEL=prophet|arima        — selects the model (default: prophet)

Model comparison:
  Prophet (Facebook)
    ✓ Handles missing months naturally (your data has gaps)
    ✓ Built-in yearly seasonality (holiday spikes, back-to-school)
    ✓ Robust to outliers (a few huge purchases won't skew it)
    ✗ Heavier install (~pystan compile on first run)
    deps: pip install prophet pandas

  ARIMA (statsmodels)
    ✓ Lightweight, fast, no compilation step
    ✓ Good for short-to-medium range forecasts (3 months)
    ✓ Interpretable — order (p,d,q) explains the model
    ✗ Needs a continuous monthly series (gaps filled with 0)
    ✗ No built-in seasonality (uses differencing instead)
    deps: pip install statsmodels pandas

Design — Python-heavy, minimal LLM tokens:
  Step 1 — Build monthly time series from transaction_rows      (Python)
  Step 2 — Route to Prophet or ARIMA based on FORECAST_MODEL   (Python)
  Step 3 — Normalize both models to same output format         (Python)
  Step 4 — Detect anomalies vs 12-month historical baseline    (Python)
  Step 5 — Detect seasonal patterns from history               (Python)
  Step 6 — Build compact summary table (~300 tokens)           (Python)
  Step 7 — Single LLM call for coaching narrative

Note on parallelism and category_map:
  Reads transaction_rows directly — no dependency on classify_categories.
  Runs fully in parallel. To use LLM-assigned category labels in forecasts,
  move this node to run after classify_categories in builder.py.

Feature flag: ENABLE_SPENDING_FORECAST
Model flag:   FORECAST_MODEL
"""

import logging
from collections import defaultdict
from typing import Dict, List

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate

from config.settings import settings
from graph.state import AgentState

logger = logging.getLogger(__name__)


# ── Valid model names ─────────────────────────────────────────
VALID_MODELS = ("prophet", "arima")

# ── LLM narrative prompt (~300 tokens input) ─────────────────
_NARRATIVE_PROMPT = PromptTemplate(
    input_variables=[
        "model_name", "historical_summary",
        "forecast_table", "anomalies", "seasonal_notes",
    ],
    template="""You are a personal finance coach reviewing a 3-month Amazon spending forecast.

Forecast model used: {model_name}
All numbers below were computed from historical order data — no raw transactions sent.

HISTORICAL MONTHLY AVERAGE (last 6 months):
{historical_summary}

FORECAST — NEXT 3 MONTHS:
{forecast_table}

ANOMALIES / RISK FLAGS:
{anomalies}

SEASONAL PATTERNS (from history):
{seasonal_notes}

Write a concise coaching report with:

### 3-MONTH SPENDING FORECAST
(reproduce the forecast table as a markdown table)

### RISK FLAGS
(highlight months or trends that warrant attention, referencing actual numbers)

### SEASONAL AWARENESS
(note which upcoming months historically spike and suggest preparation).
Make sure you include substantial data/actual examples from the data in csv and list them tying back to forecast data.

### ACTION PLAN
(3 specific, actionable steps the user can take now, tied to the forecast data)

Use actual numbers. Be direct. Avoid generic advice.
""",
)


# ── Step 1: build monthly time series ────────────────────────

def _build_monthly_series(transaction_rows: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Aggregate total spend per calendar month from raw transaction rows.
    Returns {YYYY-MM: total_spend} sorted chronologically.
    Zero LLM tokens — pure Python dict arithmetic.
    """
    monthly: Dict[str, float] = defaultdict(float)
    for row in transaction_rows:
        date_str = row.get("Order Date", "")[:7]   # YYYY-MM
        if not date_str or len(date_str) < 7:
            continue
        try:
            amount = float(row.get("Total Amount", "0").replace(",", ""))
        except ValueError:
            amount = 0.0
        monthly[date_str] += amount
    return dict(sorted(monthly.items()))


def _fill_gaps(monthly: Dict[str, float]) -> Dict[str, float]:
    """
    Fill missing months with 0.0 so ARIMA gets a continuous series.
    Prophet handles gaps natively so this is only used for ARIMA.
    """
    if not monthly:
        return monthly
    import pandas as pd
    start = min(monthly.keys()) + "-01"
    end   = max(monthly.keys()) + "-01"
    full_idx = pd.date_range(start=start, end=end, freq="MS")
    return {dt.strftime("%Y-%m"): monthly.get(dt.strftime("%Y-%m"), 0.0)
            for dt in full_idx}


# ── Step 2a: Prophet model ────────────────────────────────────

def _forecast_with_prophet(
    monthly: Dict[str, float],
    periods: int = 3,
) -> List[Dict]:
    """
    Fit Prophet and forecast `periods` months ahead.
    Returns normalized list: [{month, yhat, yhat_lower, yhat_upper}]

    Prophet advantages for this dataset:
    - Handles the irregular gaps in the 2005-2026 order history
    - Automatically detects yearly seasonality (holiday spikes)
    - Robust to the outlier months with very high spend
    """
    import pandas as pd
    from prophet import Prophet

    df = pd.DataFrame([
        {"ds": f"{ym}-01", "y": spend}
        for ym, spend in monthly.items()
    ])
    df["ds"] = pd.to_datetime(df["ds"])

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        uncertainty_samples=100,      # enable uncertainty intervals
        seasonality_mode="multiplicative",  # better for spend that scales with volume
    )
    model.fit(df)

    future   = model.make_future_dataframe(periods=periods, freq="MS")
    forecast = model.predict(future)
    tail     = forecast.tail(periods)

    return [
        {
            "month":      row["ds"].strftime("%b %Y"),
            "yhat":       max(0.0, row["yhat"]),
            "yhat_lower": max(0.0, row["yhat_lower"]),
            "yhat_upper": max(0.0, row["yhat_upper"]),
        }
        for _, row in tail.iterrows()
    ]


# ── Step 2b: ARIMA model ──────────────────────────────────────

def _forecast_with_arima(
    monthly: Dict[str, float],
    periods: int = 3,
) -> List[Dict]:
    """
    Fit ARIMA(1,1,1) and forecast `periods` months ahead.
    Returns normalized list: [{month, yhat, yhat_lower, yhat_upper}]

    ARIMA order (1,1,1):
    - p=1: one lag of the series (last month influences this month)
    - d=1: first-order differencing (removes linear trend, makes stationary)
    - q=1: one lag of the forecast error (corrects for last month's miss)
    This is a standard choice for monthly spending data with a trend.

    ARIMA advantages for this dataset:
    - Lightweight — no compilation, fast startup
    - Interpretable coefficients
    - Good 3-month lookahead accuracy for trended spend series
    """
    import pandas as pd
    from statsmodels.tsa.arima.model import ARIMA

    # Fill gaps so the series is continuous (required for ARIMA)
    filled  = _fill_gaps(monthly)
    values  = list(filled.values())
    idx     = pd.date_range(
        start=f"{min(filled.keys())}-01",
        periods=len(filled),
        freq="MS",
    )
    series = pd.Series(values, index=idx)

    # Require at least 24 data points for a reliable ARIMA fit
    if len(series) < 24:
        raise ValueError(
            f"ARIMA needs ≥24 months of data; got {len(series)}. "
            "Use Prophet for shorter histories."
        )

    model  = ARIMA(series, order=(1, 1, 1))
    fitted = model.fit()

    fc_result  = fitted.get_forecast(steps=periods)
    fc_mean    = fc_result.predicted_mean
    fc_ci      = fc_result.conf_int(alpha=0.05)   # 95% confidence interval

    return [
        {
            "month":      fc_mean.index[i].strftime("%b %Y"),
            "yhat":       max(0.0, fc_mean.iloc[i]),
            "yhat_lower": max(0.0, fc_ci.iloc[i, 0]),
            "yhat_upper": max(0.0, fc_ci.iloc[i, 1]),
        }
        for i in range(periods)
    ]


# ── Steps 4–6: Python analysis helpers (zero tokens) ─────────

def _build_forecast_table(forecasts: List[Dict], historical_avg: float) -> str:
    """Shared by both models — same output format."""
    lines = ["| Month | Forecast | vs 6mo Avg | 95% Range |",
             "|---|---|---|---|"]
    for f in forecasts:
        pct  = ((f["yhat"] - historical_avg) / historical_avg * 100) if historical_avg else 0
        arrow = "↑" if pct > 5 else ("↓" if pct < -5 else "→")
        lines.append(
            f"| {f['month']} | ${f['yhat']:,.0f} | {arrow} {pct:+.0f}% | "
            f"${f['yhat_lower']:,.0f}–${f['yhat_upper']:,.0f} |"
        )
    return "\n".join(lines)


def _detect_anomalies(forecasts: List[Dict], historical_avg: float,
                      historical_max: float) -> str:
    """Flag forecast months that significantly exceed historical norms."""
    flags = []
    for f in forecasts:
        pct = ((f["yhat"] - historical_avg) / historical_avg * 100) if historical_avg else 0
        if pct > 20:
            flags.append(
                f"  ⚠ {f['month']}: ${f['yhat']:,.0f} is {pct:+.0f}% above 6-month avg"
            )
        if f["yhat"] > historical_max:
            flags.append(
                f"  ⚠ {f['month']}: ${f['yhat']:,.0f} would be a new all-time monthly high"
            )
    return "\n".join(flags) if flags else "No significant anomalies detected."


def _seasonal_notes(monthly: Dict[str, float]) -> str:
    """
    Compute average spend per calendar month from history.
    Identifies which months consistently run high or low.
    Zero LLM tokens — pure Python aggregation.
    """
    month_totals: Dict[int, List[float]] = defaultdict(list)
    for ym, spend in monthly.items():
        try:
            m = int(ym[5:7])
            month_totals[m].append(spend)
        except (ValueError, IndexError):
            continue

    if not month_totals:
        return "Insufficient history for seasonal analysis."

    month_names = {
        1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May",  6:"Jun",
        7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec",
    }
    avgs      = {m: sum(v)/len(v) for m, v in month_totals.items()}
    overall   = sum(avgs.values()) / len(avgs)
    high_months = [month_names[m] for m, a in sorted(avgs.items())
                   if a > overall * 1.2]
    low_months  = [month_names[m] for m, a in sorted(avgs.items())
                   if a < overall * 0.8]

    lines = []
    if high_months:
        lines.append(f"Historically HIGH months (>20% above avg): {', '.join(high_months)}")
    if low_months:
        lines.append(f"Historically LOW months (>20% below avg):  {', '.join(low_months)}")
    if not lines:
        lines.append("No strong seasonal pattern detected.")
    return "\n".join(lines)


# ── Node entrypoint ───────────────────────────────────────────

async def run(state: AgentState) -> dict:
    """
    LangGraph node — forecasts next 3 months of spending.
    Runs in parallel with classify_categories and detect_impulse.

    Routing:
      FORECAST_MODEL=prophet → _forecast_with_prophet() [default]
      FORECAST_MODEL=arima   → _forecast_with_arima()

    Returns partial state update:
        forecast_analysis — markdown forecast narrative
    """
    # ── Gate 1: feature flag ───────────────────────────────────
    if not settings.ENABLE_SPENDING_FORECAST:
        logger.info("SKIPPED — ENABLE_SPENDING_FORECAST is OFF.")
        return {"forecast_analysis": "", "errors": []}

    # ── Gate 2: validate model choice ────────────────────────
    model_name = settings.FORECAST_MODEL
    if model_name not in VALID_MODELS:
        msg = f"Invalid FORECAST_MODEL='{model_name}'. Choose: {VALID_MODELS}"
        logger.error(msg)
        return {"forecast_analysis": "", "errors": [msg]}

    # ── Gate 3: data present ──────────────────────────────────
    transaction_rows = state.get("transaction_rows", [])
    if not transaction_rows:
        logger.warning("No transaction rows in state.")
        return {"forecast_analysis": "", "errors": []}

    # ── Gate 4: check dependencies ────────────────────────────
    try:
        import pandas  # noqa: F401 — needed by both models
        if model_name == "prophet":
            import prophet  # noqa: F401
        elif model_name == "arima":
            import statsmodels  # noqa: F401
    except ImportError as e:
        install_cmd = (
            "pip install prophet pandas"
            if model_name == "prophet"
            else "pip install statsmodels pandas"
        )
        msg = f"Missing dependency: {e}. Run: {install_cmd}"
        logger.error(msg)
        return {"forecast_analysis": "", "errors": [msg]}

    logger.info("Running with model=%s...", model_name)

    # ── Step 1: build monthly time series (Python) ────────────
    monthly = _build_monthly_series(transaction_rows)
    if len(monthly) < 6:
        msg = (f"Only {len(monthly)} months of data — "
               "need at least 6 for a reliable forecast.")
        logger.warning(msg)
        return {"forecast_analysis": "", "errors": [msg]}

    logger.info(
        "Built %d-month series (%s → %s)",
        len(monthly), min(monthly.keys()), max(monthly.keys()),
    )

    # ── Step 2: fit model and forecast 3 months (Python) ──────
    try:
        if model_name == "prophet":
            forecasts = _forecast_with_prophet(monthly, periods=3)
        else:
            forecasts = _forecast_with_arima(monthly, periods=3)
    except Exception as e:
        msg = f"{model_name} forecast failed: {e}"
        logger.error(msg)
        return {"forecast_analysis": "", "errors": [msg]}

    # ── Step 3: compute stats from history (Python) ───────────
    recent_values   = list(monthly.values())[-6:]
    historical_avg  = sum(recent_values) / len(recent_values)
    historical_max  = max(monthly.values())

    # ── Steps 4–6: build all tables in Python (zero tokens) ───
    forecast_table = _build_forecast_table(forecasts, historical_avg)
    anomalies      = _detect_anomalies(forecasts, historical_avg, historical_max)
    seasonal       = _seasonal_notes(monthly)

    hist_summary = (
        f"${historical_avg:,.0f}/month avg (last 6 months) | "
        f"All-time monthly high: ${historical_max:,.0f}"
    )

    est_tokens = (
        len(forecast_table) + len(anomalies) + len(seasonal) + len(hist_summary)
    ) // 4
    logger.info("Sending ~%d tokens to LLM for narrative...", est_tokens)

    # ── Step 7: single LLM call for narrative ─────────────────
    llm = ChatAnthropic(
        model=settings.ANTHROPIC_MODEL,
        temperature=0,
        api_key=settings.ANTHROPIC_API_KEY,
    )
    prompt = _NARRATIVE_PROMPT
    result = await (prompt | llm).ainvoke({
        "model_name":        model_name.upper(),
        "historical_summary": hist_summary,
        "forecast_table":    forecast_table,
        "anomalies":         anomalies,
        "seasonal_notes":    seasonal,
    })

    logger.info("Done.")
    return {"forecast_analysis": result.content, "errors": []}
