# AmazonSpendingCoach

An AI-powered financial coaching agent that analyzes your Amazon order history and gives you honest, actionable insights into your spending habits — powered by [Claude](https://www.anthropic.com) and [LangGraph](https://langchain-ai.github.io/langgraph/).

AmazonSpendingCoach reads your exported Amazon CSVs (order history, returns, cart history) and runs four parallel AI analyses:

| Analysis | What it surfaces |
|---|---|
| **Category Classifier** | Breaks your spending into categories (Electronics, Books, Household, etc.) and shows where your money actually goes |
| **Impulse Detector** | Identifies patterns in returned items and flags purchases you likely regretted |
| **Spending Forecast** | Projects future spending trends using Prophet or ARIMA time-series models |
| **Cart Analyzer** | Compares your saved-for-later cart against your actual purchases — reveals what you almost bought vs. what you held back on |

All four analyses run in parallel and are combined into a single Markdown report saved to `output/report_v4.md`.

---

## Architecture & Tech Stack

```
START
  │
  └─► load_data
        │
        ├─► classify_categories ──┐
        ├─► detect_impulse ───────┤  (PARALLEL - fan-out)
        ├─► forecast_spending ────┤
        └─► cart_analyzer ────────┤
                                  │
                            write_report  (fan-in - waits for all)
                                  │
                                 END
```

- **`load_data`** — Ingests all three CSVs into shared state. No external loaders — plain Python `csv.DictReader`.
- **Fan-out** — LangGraph fires all four analysis nodes concurrently via `asyncio`. Each node does Python pre-processing first to minimize LLM token usage, then makes a single targeted LLM call.
- **`write_report`** — Fan-in node that waits for every parallel branch to complete, then writes the combined Markdown report.

| Component | Technology |
|---|---|
| LLM | [Anthropic Claude](https://www.anthropic.com) (`claude-sonnet-4-6`) |
| Orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) `StateGraph` with async nodes |
| Forecasting | [Prophet](https://facebook.github.io/prophet/) or [ARIMA](https://www.statsmodels.org/) (optional) |
| Observability | [LangSmith](https://smith.langchain.com/) tracing (optional) |
| Data loading | Python stdlib `csv.DictReader` |

---

## Step 1 — Export Your Amazon Data

1. Go to **Amazon.com → Account & Lists → Your Account**
2. Scroll to **Data and Privacy → Request My Data**
3. Select **"Your Orders"** and **"Your Returns & Refunds"**
4. Amazon emails you a download link — unzip it

The three files you need are:

| File | Location in ZIP |
|---|---|
| `Order History.csv` | `All Data Categories/Your Amazon Orders/` |
| `Refund Details.csv` | `All Data Categories/Your Returns & Refunds/` |
| `Cart History.csv` | `All Data Categories/Your Amazon Orders/` |

---

## Step 2 — Trim the CSVs

Amazon's export includes many columns the agent doesn't use — addresses, payment methods, gift details, tracking numbers. Trimming to only the required columns keeps files small and avoids sending unnecessary personal data to the LLM.

### Order History.csv — keep 5 columns

| Keep | Used by |
|---|---|
| `ASIN` | Cart Analyzer |
| `Order ID` | Category Classifier, Impulse Detector |
| `Order Date` | All nodes |
| `Product Name` | Category Classifier, Impulse Detector, Cart Analyzer |
| `Total Amount` | Category Classifier, Impulse Detector, Spending Forecast |

Drop: `Billing Address`, `Carrier Name & Tracking Number`, `Currency`, `Gift Message`, `Gift Recipient Contact`, `Gift Sender Name`, `Item Serial Number`, `Order Status`, `Original Quantity`, `Payment Method Type`, `Product Condition`, `Purchase Order Number`, `Ship Date`, `Shipment Item Subtotal`, `Shipment Item Subtotal Tax`, `Shipment Status`, `Shipping Address`, `Shipping Charge`, `Shipping Option`, `Total Discounts`, `Unit Price`, `Unit Price Tax`, `Website`

### Refund Details.csv — keep 4 columns

| Keep | Used by |
|---|---|
| `Order ID` | Impulse Detector |
| `Refund Amount` | Impulse Detector |
| `Refund Date` | Impulse Detector |
| `Reversal Reason` | Impulse Detector |

Drop: `Creation Date`, `Currency`, `Direct Debit Refund Amount`, `Disbursement Type`, `Payment Status`, `Quantity`, `Reversal Amount State`, `Reversal Status`, `Website`

### Cart History.csv — keep 3 columns

| Keep | Used by |
|---|---|
| `ASIN` | Cart Analyzer |
| `Date Added to Cart` | Cart Analyzer |
| `Product Name` | Cart Analyzer |

Drop: `Add-on Item`, `Cart Domain`, `Cart List`, `Cart Source`, `Gift Wrapped`, `One-Click Enabled`, `Order Quantity`, `Pantry Item`, `Prime Subscription`

### Trim all three at once

Save this as `trim_csvs.py` in the same folder as your downloaded CSVs:

```python
import pandas as pd

pd.read_csv("Order History.csv")[
    ["ASIN", "Order ID", "Order Date", "Product Name", "Total Amount"]
].to_csv("Order History.csv", index=False)

pd.read_csv("Refund Details.csv")[
    ["Order ID", "Refund Amount", "Refund Date", "Reversal Reason"]
].to_csv("Refund Details.csv", index=False)

pd.read_csv("Cart History.csv")[
    ["ASIN", "Date Added to Cart", "Product Name"]
].to_csv("Cart History.csv", index=False)

print("Done — CSVs trimmed.")
```

```bash
pip install pandas
python trim_csvs.py
```

---

## Step 3 — Install & Run

### Install from the release

```bash
pip install https://github.com/shobhavijay/AmazonSpendingCoach/releases/download/v2.0.0/spending_coach-2.0.0-py3-none-any.whl
```

Or download the `.whl` manually from the [v2.0.0 Release page](https://github.com/shobhavijay/AmazonSpendingCoach/releases/tag/v2.0.0) and install it locally:

```bash
pip install spending_coach-2.0.0-py3-none-any.whl
```

### Run

Pass your Anthropic API key and CSV paths as environment variables:

```bash
ANTHROPIC_API_KEY=sk-... \
TRANSACTIONS_CSV=/path/to/Order\ History.csv \
RETURNS_CSV=/path/to/Refund\ Details.csv \
spending-coach
```

To include cart history or enable the spending forecast:

```bash
ANTHROPIC_API_KEY=sk-... \
TRANSACTIONS_CSV=/path/to/Order\ History.csv \
RETURNS_CSV=/path/to/Refund\ Details.csv \
CART_CSV=/path/to/Cart\ History.csv \
ENABLE_SPENDING_FORECAST=true \
spending-coach
```

The report is written to `output/report_v4.md`.

---

## Running from Source

If you prefer to clone and run directly:

```bash
git clone https://github.com/shobhavijay/AmazonSpendingCoach.git
cd AmazonSpendingCoach
pip install -r requirements.txt
cp .env.example .env          # fill in your API key and CSV paths
python main.py
```

For spending forecast support, also install:

```bash
pip install prophet pandas     # Prophet model
pip install statsmodels pandas # ARIMA model
```

---

## Feature Flags

All features are controlled via environment variables or `.env` — no code changes needed.

| Flag | Default | Description |
|---|---|---|
| `ENABLE_DATA_INGESTION` | `true` | Load CSV files |
| `ENABLE_CATEGORY_CLASSIFIER` | `true` | Category breakdown analysis |
| `ENABLE_IMPULSE_DETECTOR` | `true` | Impulse/returns analysis |
| `ENABLE_CART_ANALYZER` | `true` | Cart abandonment analysis |
| `ENABLE_SPENDING_FORECAST` | `false` | Time-series spending forecast |
| `ENABLE_OUTPUT_FILE` | `true` | Write markdown report |
| `ENABLE_DEBUG_MODE` | `false` | LangGraph state transition logs |
| `FORECAST_MODEL` | `prophet` | `prophet` or `arima` |

Optional LangSmith tracing:

```env
LANGSMITH_API_KEY=your_key_here
LANGSMITH_TRACING=true
```

---

## Project Structure

```
├── main.py                  # Entry point — builds and runs the graph
├── graph/
│   ├── builder.py           # LangGraph StateGraph topology
│   ├── state.py             # Shared AgentState TypedDict
│   └── nodes/
│       ├── load_data.py
│       ├── category_classifier.py
│       ├── impulse_detector.py
│       ├── spending_forecast.py
│       ├── cart_analyzer.py
│       └── write_report.py
├── config/
│   └── settings.py          # All config and feature flags
├── logging_config.py
├── requirements.txt
└── .env.example
```

---

## License

MIT
