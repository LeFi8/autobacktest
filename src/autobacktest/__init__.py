"""AutoBacktest: Autonomous AI-driven strategy optimization loop."""

import logging
import warnings

# Suppress LiteLLM noise about missing optional backends (botocore, sagemaker).
# This must run before any submodule import that triggers litellm module init.
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)

# Always suppress common numerical RuntimeWarnings to prevent backtest strategy spam
warnings.filterwarnings("ignore", category=RuntimeWarning)

__version__ = "0.1.0"


def configure_verbosity(quiet: bool = False) -> None:
    """Set logging levels and warning filters based on verbosity.

    When ``quiet=True``, suppress non-critical warnings (numpy all-NaN,
    yfinance "possibly delisted", urllib3 connection chatter) and raise
    autobacktest's internal loggers to ERROR level.
    """
    if quiet:
        for name in ("autobacktest", "yfinance", "urllib3", "requests"):
            logging.getLogger(name).setLevel(logging.ERROR)
