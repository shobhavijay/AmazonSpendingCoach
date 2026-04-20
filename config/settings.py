"""
config/settings.py
==================
Single source of truth for all config and feature flags.
Reads from .env file. All flags default to False so nothing
runs unless explicitly turned on.

Usage:
    from config.settings import settings
    if settings.ENABLE_CATEGORY_CLASSIFIER:
        ...
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)


def _flag(key: str, default: bool = False) -> bool:
    """Read a boolean feature flag from environment."""
    return os.getenv(key, str(default)).strip().lower() == "true"


class Settings:

    # ── Anthropic ─────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str   = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    # ── CSV paths ─────────────────────────────────────────────
    # Default to the sibling files/ directory so no data copying is needed.
    TRANSACTIONS_CSV: str = os.getenv("TRANSACTIONS_CSV", "../files/Order_history_4.csv")
    CATEGORIZED_CSV: str  = os.getenv("CATEGORIZED_CSV",  "../files/Order_history_4_categorized.csv")
    RETURNS_CSV: str      = os.getenv("RETURNS_CSV",       "../files/Refund_Details_4.csv")

    # ── Output ────────────────────────────────────────────────
    OUTPUT_DIR:      str = os.getenv("OUTPUT_DIR",      "output")
    REPORT_FILENAME: str = os.getenv("REPORT_FILENAME", "report_v4.md")

    # ── v1 feature flags ──────────────────────────────────────
    ENABLE_DATA_INGESTION:      bool = _flag("ENABLE_DATA_INGESTION",      True)
    ENABLE_CATEGORY_CLASSIFIER: bool = _flag("ENABLE_CATEGORY_CLASSIFIER", True)
    ENABLE_IMPULSE_DETECTOR:    bool = _flag("ENABLE_IMPULSE_DETECTOR",    True)
    ENABLE_OUTPUT_FILE:         bool = _flag("ENABLE_OUTPUT_FILE",         True)

    # ── Cart analyzer ─────────────────────────────────────────
    ENABLE_CART_ANALYZER: bool = _flag("ENABLE_CART_ANALYZER", True)
    CART_CSV: str              = os.getenv("CART_CSV", "../files/Cart History.csv")

    # ── v2 feature flags ──────────────────────────────────────
    # Set ENABLE_SPENDING_FORECAST=true in .env to activate.
    # Choose model via FORECAST_MODEL=prophet|arima (default: prophet)
    # Prophet deps : pip install prophet pandas
    # ARIMA deps   : pip install statsmodels pandas
    ENABLE_SPENDING_FORECAST: bool = _flag("ENABLE_SPENDING_FORECAST", False)
    FORECAST_MODEL: str            = os.getenv("FORECAST_MODEL", "prophet").strip().lower()

    # ── Logging / observability ───────────────────────────────
    # ENABLE_DEBUG_MODE=true  → LangGraph prints every state transition to console
    # LangSmith tracing auto-activates when LANGSMITH_API_KEY + LANGSMITH_TRACING=true
    ENABLE_DEBUG_MODE: bool = _flag("ENABLE_DEBUG_MODE", False)
    LOG_FILENAME: str       = os.getenv("LOG_FILENAME", "agent.log")

    # ── Future stubs ──────────────────────────────────────────
    ENABLE_LIFE_PATTERN:     bool = _flag("ENABLE_LIFE_PATTERN",     False)
    ENABLE_OUTPUT_EMAIL:     bool = _flag("ENABLE_OUTPUT_EMAIL",     False)
    ENABLE_OUTPUT_DASHBOARD: bool = _flag("ENABLE_OUTPUT_DASHBOARD", False)

    def summary(self) -> str:
        """Print current feature flag state at startup."""
        flags = [
            ("Data ingestion",      self.ENABLE_DATA_INGESTION),
            ("Category classifier", self.ENABLE_CATEGORY_CLASSIFIER),
            ("Impulse detector",    self.ENABLE_IMPULSE_DETECTOR),
            ("Output: file",        self.ENABLE_OUTPUT_FILE),
            ("Cart analyzer",       self.ENABLE_CART_ANALYZER),
            ("-- v2 --",            None),
            ("Spending forecast",   self.ENABLE_SPENDING_FORECAST),
            ("-- future --",        None),
            ("Life pattern",        self.ENABLE_LIFE_PATTERN),
            ("Output: email",       self.ENABLE_OUTPUT_EMAIL),
            ("Output: dashboard",   self.ENABLE_OUTPUT_DASHBOARD),
        ]
        lines = ["\n╔══ Feature Flags (LangGraph v2) ═══════╗"]
        for name, val in flags:
            if val is None:
                lines.append("║  ────────────────────────────────────  ║")
            else:
                status = "✓ ON " if val else "✗ OFF"
                lines.append(f"║  [{status}]  {name:<26} ║")
        if self.ENABLE_SPENDING_FORECAST:
            lines.append(f"║         model: {self.FORECAST_MODEL:<22} ║")
        lines.append("║  ────────────────────────────────────  ║")
        debug_status = "✓ ON " if self.ENABLE_DEBUG_MODE else "✗ OFF"
        lines.append(f"║  [{debug_status}]  {'LangGraph debug mode':<26} ║")
        langsmith_on = bool(os.getenv("LANGSMITH_API_KEY") and os.getenv("LANGSMITH_TRACING", "").lower() == "true")
        ls_status = "✓ ON " if langsmith_on else "✗ OFF"
        lines.append(f"║  [{ls_status}]  {'LangSmith tracing':<26} ║")
        lines.append("╚════════════════════════════════════════╝")
        return "\n".join(lines)


settings = Settings()
