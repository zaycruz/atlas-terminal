"""Central environment and configuration helpers for the Atlas CLI."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"

APP_DIR = Path.home() / ".atlas"
CONFIG_DIR = APP_DIR / "config"
DATA_DIR = APP_DIR / "data"
LOGS_DIR = APP_DIR / "logs"


def _ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


_ensure_directories((APP_DIR, CONFIG_DIR, DATA_DIR, LOGS_DIR))


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


def get_alpaca_credentials() -> Tuple[str, str]:
    """Return Alpaca API credentials from the environment."""
    api_key = os.environ.get("ALPACA_API_KEY_ID")
    secret_key = os.environ.get("ALPACA_API_SECRET_KEY")
    if not api_key or not secret_key:
        missing = [
            name
            for name, value in (
                ("ALPACA_API_KEY_ID", api_key),
                ("ALPACA_API_SECRET_KEY", secret_key),
            )
            if not value
        ]
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )
    return api_key, secret_key


def resolve_environment(target: Optional[str]) -> str:
    """Normalize the trading environment string."""
    choice = (target or os.environ.get("ATLAS_ENV") or "paper").lower()
    if choice not in {"paper", "live"}:
        raise ValueError("Trading environment must be either 'paper' or 'live'.")
    return choice

DEFAULT_AI_PROMPT = "You are Atlas, a trading assistant. Use tools when helpful and confirm critical actions."

def get_ai_model(override: Optional[str] = None) -> str:
    """Return the LLM model name to use for AI mode."""
    if override and override.strip():
        return override.strip()
    return os.environ.get("ATLAS_AI_MODEL", "llama3.2").strip()

def get_ai_system_prompt() -> str:
    """Return the system prompt for the AI assistant."""
    return os.environ.get("ATLAS_AI_SYSTEM_PROMPT", DEFAULT_AI_PROMPT)

def get_ollama_host() -> str:
    """Return the base URL for the Ollama server."""
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    return host.rstrip("/")


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
