"""Unit tests for AutoBacktest configuration management."""

import os
from pathlib import Path
from unittest import mock

from autobacktest.config import Settings


def test_default_fallbacks() -> None:
    """Verify default fallbacks when environment variables are empty."""
    # Ensure environment is clean of our variables
    with mock.patch.dict(os.environ, {}, clear=True):
        settings = Settings()
        assert settings.llm_provider == "litellm"
        assert settings.llm_model == "openai/gpt-4o"
        assert settings.llm_temperature == 0.7
        assert settings.llm_max_tokens == 4096
        assert settings.litellm_debug is False
        assert settings.default_start_date == "2015-01-01"
        assert settings.default_end_date == "2026-01-01"
        assert settings.default_holdout_years == 3
        assert settings.run_dir == Path("runs")
        assert settings.cache_dir == Path("data/cache")
        assert settings.strategies_dir == Path("strategies")
        assert settings.configs_dir == Path("configs")
        assert settings.ledger_db_name == "ledger.db"
        assert settings.max_file_size_kb == 100
        assert settings.db_timeout == 15.0
        assert settings.ledger_db_path == Path("runs/ledger.db")

        assert "pandas" in settings.parsed_safe_imports
        assert "numpy" in settings.parsed_safe_imports


def test_env_var_override() -> None:
    """Verify that settings are correctly overridden by environment variables."""
    custom_env = {
        "AUTOBACKTEST_LLM_PROVIDER": "custom_provider",
        "AUTOBACKTEST_LLM_MODEL": "custom_model/v1",
        "AUTOBACKTEST_LLM_TEMPERATURE": "0.2",
        "AUTOBACKTEST_LLM_MAX_TOKENS": "512",
        "AUTOBACKTEST_LITELLM_DEBUG": "True",
        "AUTOBACKTEST_DEFAULT_START_DATE": "2020-06-01",
        "AUTOBACKTEST_DEFAULT_END_DATE": "2025-12-31",
        "AUTOBACKTEST_DEFAULT_HOLDOUT_YEARS": "1",
        "AUTOBACKTEST_RUN_DIR": "test_runs",
        "AUTOBACKTEST_CACHE_DIR": "test_cache",
        "AUTOBACKTEST_STRATEGIES_DIR": "test_strategies",
        "AUTOBACKTEST_CONFIGS_DIR": "test_configs",
        "AUTOBACKTEST_LEDGER_DB_NAME": "test_ledger.db",
        "AUTOBACKTEST_MAX_FILE_SIZE_KB": "50",
        "AUTOBACKTEST_SAFE_IMPORTS_WHITELIST": "math,typing,my_module",
        "AUTOBACKTEST_DB_TIMEOUT": "5.5",
    }

    with mock.patch.dict(os.environ, custom_env, clear=True):
        settings = Settings()
        assert settings.llm_provider == "custom_provider"
        assert settings.llm_model == "custom_model/v1"
        assert settings.llm_temperature == 0.2
        assert settings.llm_max_tokens == 512
        assert settings.litellm_debug is True
        assert settings.default_start_date == "2020-06-01"
        assert settings.default_end_date == "2025-12-31"
        assert settings.default_holdout_years == 1
        assert settings.run_dir == Path("test_runs")
        assert settings.cache_dir == Path("test_cache")
        assert settings.strategies_dir == Path("test_strategies")
        assert settings.configs_dir == Path("test_configs")
        assert settings.ledger_db_name == "test_ledger.db"
        assert settings.max_file_size_kb == 50
        assert settings.db_timeout == 5.5
        assert settings.ledger_db_path == Path("test_runs/test_ledger.db")

        # Whitelist parsing checks
        assert settings.parsed_safe_imports == {"math", "typing", "my_module"}
