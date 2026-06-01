"""AutoBacktest: Autonomous AI-driven strategy optimization loop."""

import logging

# Suppress LiteLLM noise about missing optional backends (botocore, sagemaker).
# This must run before any submodule import that triggers litellm module init.
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)

__version__ = "0.1.0"
