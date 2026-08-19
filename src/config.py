import os
import logging
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


def validate_api_key(key: str, name: str, required: bool = False, min_length: int = 20) -> Optional[str]:
    """
    Validate an API key.

    Args:
        key: The API key value
        name: Name of the key for error messages
        required: Whether the key is required
        min_length: Minimum length for a valid key

    Returns:
        The validated key or None if empty and not required

    Raises:
        ConfigurationError: If required key is missing or invalid
    """
    if not key or key.startswith("your_") or key == "":
        if required:
            raise ConfigurationError(
                f"{name} is required but not set. "
                f"Please set it in your .env file."
            )
        return None

    if len(key) < min_length:
        if required:
            raise ConfigurationError(
                f"{name} appears to be invalid (too short). "
                f"Expected at least {min_length} characters."
            )
        logger.warning(f"{name} appears to be invalid (too short)")
        return None

    return key


@dataclass
class Config:
    api_key: str
    api_secret: str
    testnet: bool
    dry_run: bool
    log_level: str
    default_symbol: str
    default_interval: str
    max_leverage: int
    risk_per_trade: float
    # OpenAI settings
    openai_api_key: str
    openai_model: str
    openai_base_url: str
    enable_llm: bool


def load_config(validate_binance: bool = False, validate_openai: bool = False) -> Config:
    """
    Load configuration from environment variables.

    Args:
        validate_binance: If True, raise error if Binance keys are missing
        validate_openai: If True, raise error if OpenAI key is missing

    Returns:
        Config object with validated settings

    Raises:
        ConfigurationError: If required keys are missing or invalid
    """
    dry_run = os.getenv("DRY_RUN", "true").lower() == "true"
    enable_llm = os.getenv("ENABLE_LLM", "false").lower() == "true"

    # Binance keys are required only if not in dry-run mode or explicitly requested
    binance_required = validate_binance or not dry_run
    api_key = validate_api_key(
        os.getenv("BINANCE_API_KEY", ""),
        "BINANCE_API_KEY",
        required=binance_required
    )
    api_secret = validate_api_key(
        os.getenv("BINANCE_API_SECRET", ""),
        "BINANCE_API_SECRET",
        required=binance_required
    )

    # OpenAI-compatible key is required only if LLM is enabled or explicitly requested
    openai_required = validate_openai or enable_llm
    openai_key_env = (
        os.getenv("OPENAI_API_KEY", "")
        or os.getenv("DEEPSEEK_API_KEY", "")
        or os.getenv("LLM_API_KEY", "")
    )
    openai_api_key = validate_api_key(
        openai_key_env,
        "OPENAI_API_KEY",
        required=openai_required
    )

    openai_model = (
        os.getenv("OPENAI_MODEL", "")
        or os.getenv("DEEPSEEK_MODEL", "")
        or os.getenv("LLM_MODEL", "")
        or "gpt-4o-mini"
    )
    openai_base_url = (
        os.getenv("OPENAI_BASE_URL", "")
        or os.getenv("DEEPSEEK_API_BASE", "")
        or os.getenv("LLM_API_BASE", "")
    )

    return Config(
        api_key=api_key or "",
        api_secret=api_secret or "",
        testnet=os.getenv("BINANCE_TESTNET", "false").lower() == "true",
        dry_run=dry_run,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        default_symbol=os.getenv("DEFAULT_SYMBOL", "BTCUSDT"),
        default_interval=os.getenv("DEFAULT_INTERVAL", "15m"),
        max_leverage=int(os.getenv("MAX_LEVERAGE", "5")),
        risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.01")),
        # OpenAI settings
        openai_api_key=openai_api_key or "",
        openai_model=openai_model,
        openai_base_url=openai_base_url,
        enable_llm=enable_llm,
    )