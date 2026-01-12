"""Central environment and configuration helpers for Atlas."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"

APP_DIR = Path.home() / ".atlas"
CONFIG_DIR = APP_DIR / "config"
DATA_DIR = APP_DIR / "data"
LOGS_DIR = APP_DIR / "logs"
CACHE_DIR = APP_DIR / "cache"
BACKTESTS_DIR = DATA_DIR / "backtests"


def _ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


_ensure_directories((APP_DIR, CONFIG_DIR, DATA_DIR, LOGS_DIR, CACHE_DIR, BACKTESTS_DIR))


def load_dotenv(path: Optional[Path] = None) -> None:
    """Load environment variables from a simple KEY=VALUE .env file."""
    env_path = Path(path or DEFAULT_ENV_PATH)
    if not env_path.exists():
        return

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith("\"") and value.endswith("\"")) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)


# ---------------------------------------------------------------------------
# LLM Configuration (zai.org / GLM-4)
# ---------------------------------------------------------------------------

DEFAULT_LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_LLM_MODEL = "glm-4.7"

DEFAULT_SYSTEM_PROMPT = """You are Atlas, an AI-powered financial research assistant.
You help users with:
- Backtesting trading strategies
- Probabilistic analysis and Monte Carlo simulations
- ML regression and forecasting
- Market research and thesis development
- Monitoring macroeconomic events, Fed announcements, COT reports

When users request analysis or backtests, use the available tools to execute their requests.
Be concise but thorough. Explain your reasoning when providing analysis."""


def get_llm_api_key() -> str | None:
    """Return the LLM API key."""
    return os.environ.get("ZAI_API_KEY") or os.environ.get("ATLAS_LLM_API_KEY")


def get_llm_base_url() -> str:
    """Return the LLM API base URL."""
    url = os.environ.get("ATLAS_LLM_BASE_URL", DEFAULT_LLM_BASE_URL)
    return url.rstrip("/")


def get_llm_model(override: Optional[str] = None) -> str:
    """Return the LLM model name to use."""
    if override and override.strip():
        return override.strip()
    return os.environ.get("ATLAS_LLM_MODEL", DEFAULT_LLM_MODEL).strip()


def get_system_prompt() -> str:
    """Return the system prompt for the AI assistant."""
    return os.environ.get("ATLAS_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# MCP Docker Configuration (for backtesting agent)
# ---------------------------------------------------------------------------

MCP_ENDPOINT = os.environ.get("ATLAS_MCP_ENDPOINT", "http://localhost:3000")
MCP_TOKEN = os.environ.get("ATLAS_MCP_TOKEN")
DEFAULT_BACKTEST_IMAGE = os.environ.get("ATLAS_BACKTEST_IMAGE", "python:3.11-slim")


def get_mcp_endpoint() -> str:
    """Return the MCP server endpoint."""
    return MCP_ENDPOINT


def get_mcp_token() -> str | None:
    """Return the optional MCP authentication token."""
    token = MCP_TOKEN
    return token.strip() if token else None


def get_backtest_image() -> str:
    """Return the default Docker image used for backtesting."""
    return DEFAULT_BACKTEST_IMAGE


# ---------------------------------------------------------------------------
# Search Configuration (SearxNG)
# ---------------------------------------------------------------------------

SEARXNG_ENDPOINT = os.environ.get("ATLAS_SEARXNG_ENDPOINT", "http://localhost:8080/search")
SEARXNG_CATEGORIES = os.environ.get("ATLAS_SEARXNG_CATEGORIES")


def get_searxng_endpoint() -> str:
    """Return the SearxNG search endpoint."""
    return SEARXNG_ENDPOINT.rstrip("/")


def get_searxng_categories() -> list[str] | None:
    """Return default categories for SearxNG searches."""
    if not SEARXNG_CATEGORIES:
        return None
    return [part.strip() for part in SEARXNG_CATEGORIES.split(",") if part.strip()]


# ---------------------------------------------------------------------------
# Data Feed Configuration
# ---------------------------------------------------------------------------

def get_fred_api_key() -> str | None:
    """Return the FRED API key for economic data."""
    return os.environ.get("FRED_API_KEY")


def get_polygon_api_key() -> str | None:
    """Return the Polygon.io API key for market data."""
    return os.environ.get("POLYGON_API_KEY")


def get_twitter_bearer_token() -> str | None:
    """Return the Twitter/X bearer token for social monitoring."""
    return os.environ.get("TWITTER_BEARER_TOKEN")
