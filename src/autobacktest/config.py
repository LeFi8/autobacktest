"""Central configuration management for AutoBacktest.

Loads environment variables from ``.env`` via ``python-dotenv`` and provides
a global ``settings`` singleton of type ``Settings``.  ``Settings`` is a
Pydantic ``BaseModel`` whose fields resolve lazily through ``default_factory``
closures, allowing environment variables to be set at any point before the
first field access.

Key responsibilities:
- LLM provider/model/temperature configuration
- Backtesting date windows and holdout length
- System directory paths (run dir, cache, strategies, configs)
- Optimisation loop parameters (n_candidates, importance thresholds, early_stop_patience)
- Safety gate limits (file size, cyclomatic complexity, import whitelist)
- SQLite database timeout configuration
"""

import os
from functools import cached_property
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load any local .env file on module import
load_dotenv()


def _env_str(key: str, default: str) -> Any:
    """Create a Pydantic Field that reads a string value from an environment variable.

    Args:
        key: Environment variable name (e.g. ``AUTOBACKTEST_LLM_MODEL``).
        default: Fallback value when the env var is unset or empty.

    Returns:
        A ``Field`` with a ``default_factory`` that resolves the env var at access time.
    """
    return Field(default_factory=lambda: os.getenv(key, default))


def _env_int(key: str, default: str) -> Any:
    """Create a Pydantic Field that reads an integer value from an environment variable.

    Args:
        key: Environment variable name.
        default: Fallback value as a string (converted to ``int`` at access time).

    Returns:
        A ``Field`` with a ``default_factory`` that resolves and casts the env var.
    """
    return Field(default_factory=lambda: int(os.getenv(key, default)))


def _env_float(key: str, default: str) -> Any:
    """Create a Pydantic Field that reads a float value from an environment variable.

    Args:
        key: Environment variable name.
        default: Fallback value as a string (converted to ``float`` at access time).

    Returns:
        A ``Field`` with a ``default_factory`` that resolves and casts the env var.
    """
    return Field(default_factory=lambda: float(os.getenv(key, default)))


def _env_bool(key: str, default: str) -> Any:
    """Create a Pydantic Field that reads a boolean value from an environment variable.

    Truthy values are ``"true"``, ``"1"``, and ``"1"`` (case-insensitive).

    Args:
        key: Environment variable name.
        default: Fallback value as a string.

    Returns:
        A ``Field`` with a ``default_factory`` that resolves and parses the env var.
    """
    return Field(default_factory=lambda: os.getenv(key, default).lower() in ("true", "1", "yes"))


def _env_path(key: str, default: str) -> Any:
    """Create a Pydantic Field that reads a Path value from an environment variable.

    Args:
        key: Environment variable name.
        default: Fallback value as a string (converted to ``Path`` at access time).

    Returns:
        A ``Field`` with a ``default_factory`` that resolves and wraps the env var.
    """
    return Field(default_factory=lambda: Path(os.getenv(key, default)))


class Settings(BaseModel):
    """Central configuration singleton for AutoBacktest.

    All fields are resolved lazily from environment variables (prefixed
    ``AUTOBACKTEST_``) via ``default_factory`` closures. Environment variables
    can be set at any point before the first field access — typically loaded
    from ``.env`` via ``python-dotenv`` on module import.

    Configuration categories:
    - **LLM Client**: provider, model, temperature, token limits, caching
    - **Backtest Windows**: start/end dates, holdout period length
    - **System Directories**: run dir, cache dir, strategies dir, configs dir
    - **Optimization Loop**: candidate count, parallelism, early stopping
    - **Safety Gates**: file size, complexity, import whitelist, sandbox timeout
    - **Repair & Salvage**: codemod, LLM repair, config jitter, JSON salvage
    - **Diversity Gates**: config similarity, returns correlation thresholds
    - **Verbosity**: warning suppression
    - **Storage**: SQLite timeout

    Properties:
        parsed_safe_imports: Resolved set of whitelisted import module names.
        ledger_db_path: Full path to the SQLite ledger database file.
        lessons_db_path: Full path to the SQLite lessons database file.
    """

    # --- LLM CLIENT CONFIGURATION ---
    llm_provider: str = _env_str("AUTOBACKTEST_LLM_PROVIDER", "litellm")
    llm_model: str = _env_str("AUTOBACKTEST_LLM_MODEL", "openai/gpt-4o")
    llm_temperature: float = _env_float("AUTOBACKTEST_LLM_TEMPERATURE", "0.7")
    llm_max_tokens: int = _env_int("AUTOBACKTEST_LLM_MAX_TOKENS", "4096")
    litellm_debug: bool = _env_bool("AUTOBACKTEST_LITELLM_DEBUG", "False")
    llm_request_timeout: float = _env_float("AUTOBACKTEST_LLM_REQUEST_TIMEOUT", "600.0")
    llm_prompt_cache: bool = _env_bool("AUTOBACKTEST_LLM_PROMPT_CACHE", "true")

    # --- BACKTEST WINDOWS ---
    default_start_date: str = _env_str("AUTOBACKTEST_DEFAULT_START_DATE", "2015-01-01")
    default_end_date: str = _env_str("AUTOBACKTEST_DEFAULT_END_DATE", "2026-01-01")
    default_holdout_years: int = _env_int("AUTOBACKTEST_DEFAULT_HOLDOUT_YEARS", "3")

    # --- SYSTEM DIRECTORIES & STORAGE ---
    run_dir: Path = _env_path("AUTOBACKTEST_RUN_DIR", "runs")
    cache_dir: Path = _env_path("AUTOBACKTEST_CACHE_DIR", "data/cache")
    strategies_dir: Path = _env_path("AUTOBACKTEST_STRATEGIES_DIR", "strategies")
    configs_dir: Path = _env_path("AUTOBACKTEST_CONFIGS_DIR", "configs")
    ledger_db_name: str = _env_str("AUTOBACKTEST_LEDGER_DB_NAME", "ledger.db")

    # --- OPTIMIZATION LOOP ---
    n_candidates: int = _env_int("AUTOBACKTEST_N_CANDIDATES", "10")
    eval_max_workers: int = _env_int("AUTOBACKTEST_EVAL_MAX_WORKERS", "4")
    importance_min_attempts: int = _env_int("AUTOBACKTEST_IMPORTANCE_MIN_ATTEMPTS", "6")
    importance_p_threshold: float = _env_float("AUTOBACKTEST_IMPORTANCE_P_THRESHOLD", "0.20")
    early_stop_patience: int = _env_int("AUTOBACKTEST_EARLY_STOP_PATIENCE", "10")
    holdout_peek_limit: int = _env_int("AUTOBACKTEST_HOLDOUT_PEEK_LIMIT", "20")
    stuck_threshold: int = _env_int("AUTOBACKTEST_STUCK_THRESHOLD", "5")
    exploit_patience: int = _env_int("AUTOBACKTEST_EXPLOIT_PATIENCE", "3")
    min_temp: float = _env_float("AUTOBACKTEST_MIN_TEMP", "0.1")

    # --- SAFETY GATES ---
    max_file_size_kb: int = _env_int("AUTOBACKTEST_MAX_FILE_SIZE_KB", "100")
    max_cyclomatic_complexity: int = _env_int("AUTOBACKTEST_MAX_CYCLOMATIC_COMPLEXITY", "25")
    max_function_lines: int = _env_int("AUTOBACKTEST_MAX_FUNCTION_LINES", "100")
    safe_imports_whitelist: str = _env_str(
        "AUTOBACKTEST_SAFE_IMPORTS_WHITELIST",
        "pandas,numpy,math,typing,scipy,dataclasses,collections,itertools,functools,decimal,statistics,numbers,json",
    )
    sandbox_timeout: int = _env_int("AUTOBACKTEST_SANDBOX_TIMEOUT", "15")
    enable_codemod_repair: bool = _env_bool("AUTOBACKTEST_ENABLE_CODEMOD_REPAIR", "true")
    enable_config_diversity_gate: bool = _env_bool("AUTOBACKTEST_ENABLE_CONFIG_DIVERSITY_GATE", "true")
    enable_llm_repair: bool = _env_bool("AUTOBACKTEST_ENABLE_LLM_REPAIR", "true")
    max_repair_attempts: int = _env_int("AUTOBACKTEST_MAX_REPAIR_ATTEMPTS", "2")
    enable_config_jitter: bool = _env_bool("AUTOBACKTEST_ENABLE_CONFIG_JITTER", "true")
    config_jitter_max_attempts: int = _env_int("AUTOBACKTEST_CONFIG_JITTER_MAX_ATTEMPTS", "12")
    config_jitter_rel_step: float = _env_float("AUTOBACKTEST_CONFIG_JITTER_REL_STEP", "0.15")
    enable_json_salvage: bool = _env_bool("AUTOBACKTEST_ENABLE_JSON_SALVAGE", "true")
    enable_candidate_directives: bool = _env_bool("AUTOBACKTEST_ENABLE_CANDIDATE_DIRECTIVES", "true")
    enable_explored_config_injection: bool = _env_bool("AUTOBACKTEST_ENABLE_EXPLORED_CONFIG_INJECTION", "true")
    explored_config_max_configs: int = _env_int("AUTOBACKTEST_EXPLORED_CONFIG_MAX_CONFIGS", "30")
    enable_identical_behavior_guard: bool = _env_bool("AUTOBACKTEST_ENABLE_IDENTICAL_BEHAVIOR_GUARD", "true")
    identical_behavior_epsilon: float = _env_float("AUTOBACKTEST_IDENTICAL_BEHAVIOR_EPSILON", "1e-6")
    diversity_config_threshold: float = _env_float("AUTOBACKTEST_DIVERSITY_CONFIG_THRESHOLD", "0.95")
    diversity_returns_threshold: float = _env_float("AUTOBACKTEST_DIVERSITY_RETURNS_THRESHOLD", "0.95")
    diversity_compare_mode: str = _env_str("AUTOBACKTEST_DIVERSITY_COMPARE_MODE", "recent")
    diversity_recent_n: int = _env_int("AUTOBACKTEST_DIVERSITY_RECENT_N", "5")
    diversity_hard_threshold: float = _env_float("AUTOBACKTEST_DIVERSITY_HARD_THRESHOLD", "0.999")
    diversity_returns_penalty: float = _env_float("AUTOBACKTEST_DIVERSITY_RETURNS_PENALTY", "0.0")

    # --- CHEAP IN-SAMPLE PRE-SCREEN ---
    enable_cheap_prescreen: bool = _env_bool("AUTOBACKTEST_ENABLE_CHEAP_PRESCREEN", "false")
    prescreen_sharpe_floor: float = _env_float("AUTOBACKTEST_PRESCREEN_SHARPE_FLOOR", "0.0")
    prescreen_return_floor: float = _env_float("AUTOBACKTEST_PRESCREEN_RETURN_FLOOR", "0.0")

    # --- VERBOSITY CONTROL ---
    quiet: bool = _env_bool("AUTOBACKTEST_QUIET", "false")

    # --- SQLITE STORAGE CONFIGURATION ---
    db_timeout: float = _env_float("AUTOBACKTEST_DB_TIMEOUT", "15.0")

    @cached_property
    def parsed_safe_imports(self) -> set[str]:
        """Parse the comma-separated import whitelist into a set of module names.

        Returns:
            Set of stripped, non-empty module names from ``safe_imports_whitelist``.
        """
        return {x.strip() for x in self.safe_imports_whitelist.split(",") if x.strip()}

    @property
    def ledger_db_path(self) -> Path:
        """Resolve the full filesystem path to the SQLite ledger database.

        Composed from ``run_dir / ledger_db_name`` (default: ``runs/ledger.db``).
        """
        return self.run_dir / self.ledger_db_name

    @property
    def lessons_db_path(self) -> Path:
        """Resolve the full filesystem path to the SQLite lessons database.

        Always located at ``run_dir / lessons.db``.
        """
        return self.run_dir / "lessons.db"


# Export global settings instance
settings = Settings()
