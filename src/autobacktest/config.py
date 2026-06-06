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
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load any local .env file on module import
load_dotenv()


class Settings(BaseModel):
    """Settings manager loaded from env variables with safe fallbacks."""

    # --- LLM CLIENT CONFIGURATION ---
    llm_provider: str = Field(default_factory=lambda: os.getenv("AUTOBACKTEST_LLM_PROVIDER", "litellm"))
    llm_model: str = Field(default_factory=lambda: os.getenv("AUTOBACKTEST_LLM_MODEL", "openai/gpt-4o"))
    llm_temperature: float = Field(default_factory=lambda: float(os.getenv("AUTOBACKTEST_LLM_TEMPERATURE", "0.7")))
    llm_max_tokens: int = Field(default_factory=lambda: int(os.getenv("AUTOBACKTEST_LLM_MAX_TOKENS", "4096")))
    litellm_debug: bool = Field(
        default_factory=lambda: os.getenv("AUTOBACKTEST_LITELLM_DEBUG", "False").lower() in ("true", "1", "yes")
    )
    llm_request_timeout: float = Field(
        default_factory=lambda: float(os.getenv("AUTOBACKTEST_LLM_REQUEST_TIMEOUT", "600.0"))
    )
    llm_prompt_cache: bool = Field(
        default_factory=lambda: os.getenv("AUTOBACKTEST_LLM_PROMPT_CACHE", "true").lower() in ("true", "1", "yes")
    )

    # --- BACKTEST WINDOWS ---
    default_start_date: str = Field(default_factory=lambda: os.getenv("AUTOBACKTEST_DEFAULT_START_DATE", "2015-01-01"))
    default_end_date: str = Field(default_factory=lambda: os.getenv("AUTOBACKTEST_DEFAULT_END_DATE", "2026-01-01"))
    default_holdout_years: int = Field(
        default_factory=lambda: int(os.getenv("AUTOBACKTEST_DEFAULT_HOLDOUT_YEARS", "3"))
    )

    # --- SYSTEM DIRECTORIES & STORAGE ---
    run_dir: Path = Field(default_factory=lambda: Path(os.getenv("AUTOBACKTEST_RUN_DIR", "runs")))
    cache_dir: Path = Field(default_factory=lambda: Path(os.getenv("AUTOBACKTEST_CACHE_DIR", "data/cache")))
    strategies_dir: Path = Field(default_factory=lambda: Path(os.getenv("AUTOBACKTEST_STRATEGIES_DIR", "strategies")))
    configs_dir: Path = Field(default_factory=lambda: Path(os.getenv("AUTOBACKTEST_CONFIGS_DIR", "configs")))
    ledger_db_name: str = Field(default_factory=lambda: os.getenv("AUTOBACKTEST_LEDGER_DB_NAME", "ledger.db"))

    # --- OPTIMIZATION LOOP ---
    n_candidates: int = Field(default_factory=lambda: int(os.getenv("AUTOBACKTEST_N_CANDIDATES", "3")))
    importance_min_attempts: int = Field(
        default_factory=lambda: int(os.getenv("AUTOBACKTEST_IMPORTANCE_MIN_ATTEMPTS", "6"))
    )
    importance_p_threshold: float = Field(
        default_factory=lambda: float(os.getenv("AUTOBACKTEST_IMPORTANCE_P_THRESHOLD", "0.20"))
    )
    early_stop_patience: int = Field(default_factory=lambda: int(os.getenv("AUTOBACKTEST_EARLY_STOP_PATIENCE", "10")))

    # --- SAFETY GATES ---
    max_file_size_kb: int = Field(default_factory=lambda: int(os.getenv("AUTOBACKTEST_MAX_FILE_SIZE_KB", "100")))
    max_cyclomatic_complexity: int = Field(
        default_factory=lambda: int(os.getenv("AUTOBACKTEST_MAX_CYCLOMATIC_COMPLEXITY", "25"))
    )
    max_function_lines: int = Field(default_factory=lambda: int(os.getenv("AUTOBACKTEST_MAX_FUNCTION_LINES", "100")))
    safe_imports_whitelist: str = Field(
        default_factory=lambda: os.getenv(
            "AUTOBACKTEST_SAFE_IMPORTS_WHITELIST",
            "pandas,numpy,math,typing,scipy,dataclasses,collections,itertools,functools,decimal,statistics,numbers,json",
        )
    )
    sandbox_timeout: int = Field(default_factory=lambda: int(os.getenv("AUTOBACKTEST_SANDBOX_TIMEOUT", "15")))
    enable_codemod_repair: bool = Field(
        default_factory=lambda: os.getenv("AUTOBACKTEST_ENABLE_CODEMOD_REPAIR", "true").lower() == "true"
    )
    enable_config_diversity_gate: bool = Field(
        default_factory=lambda: os.getenv("AUTOBACKTEST_ENABLE_CONFIG_DIVERSITY_GATE", "true").lower() == "true"
    )
    enable_llm_repair: bool = Field(
        default_factory=lambda: os.getenv("AUTOBACKTEST_ENABLE_LLM_REPAIR", "true").lower() == "true"
    )
    max_repair_attempts: int = Field(default_factory=lambda: int(os.getenv("AUTOBACKTEST_MAX_REPAIR_ATTEMPTS", "2")))
    enable_config_jitter: bool = Field(
        default_factory=lambda: os.getenv("AUTOBACKTEST_ENABLE_CONFIG_JITTER", "true").lower() == "true"
    )
    config_jitter_max_attempts: int = Field(
        default_factory=lambda: int(os.getenv("AUTOBACKTEST_CONFIG_JITTER_MAX_ATTEMPTS", "12"))
    )
    config_jitter_rel_step: float = Field(
        default_factory=lambda: float(os.getenv("AUTOBACKTEST_CONFIG_JITTER_REL_STEP", "0.15"))
    )
    enable_json_salvage: bool = Field(
        default_factory=lambda: os.getenv("AUTOBACKTEST_ENABLE_JSON_SALVAGE", "true").lower() == "true"
    )
    enable_candidate_directives: bool = Field(
        default_factory=lambda: os.getenv("AUTOBACKTEST_ENABLE_CANDIDATE_DIRECTIVES", "true").lower() == "true"
    )
    enable_explored_config_injection: bool = Field(
        default_factory=lambda: os.getenv("AUTOBACKTEST_ENABLE_EXPLORED_CONFIG_INJECTION", "true").lower() == "true"
    )
    explored_config_max_configs: int = Field(
        default_factory=lambda: int(os.getenv("AUTOBACKTEST_EXPLORED_CONFIG_MAX_CONFIGS", "30"))
    )
    enable_identical_behavior_guard: bool = Field(
        default_factory=lambda: os.getenv("AUTOBACKTEST_ENABLE_IDENTICAL_BEHAVIOR_GUARD", "true").lower() == "true"
    )
    identical_behavior_epsilon: float = Field(
        default_factory=lambda: float(os.getenv("AUTOBACKTEST_IDENTICAL_BEHAVIOR_EPSILON", "1e-6"))
    )
    diversity_config_threshold: float = Field(
        default_factory=lambda: float(os.getenv("AUTOBACKTEST_DIVERSITY_CONFIG_THRESHOLD", "0.95"))
    )
    diversity_returns_threshold: float = Field(
        default_factory=lambda: float(os.getenv("AUTOBACKTEST_DIVERSITY_RETURNS_THRESHOLD", "0.95"))
    )

    # --- VERBOSITY CONTROL ---
    quiet: bool = Field(
        default_factory=lambda: os.getenv("AUTOBACKTEST_QUIET", "false").lower() in ("true", "1", "yes")
    )

    # --- SQLITE STORAGE CONFIGURATION ---
    db_timeout: float = Field(default_factory=lambda: float(os.getenv("AUTOBACKTEST_DB_TIMEOUT", "15.0")))

    @property
    def parsed_safe_imports(self) -> set[str]:
        """Convert comma-separated imports string into set."""
        return {x.strip() for x in self.safe_imports_whitelist.split(",") if x.strip()}

    @property
    def ledger_db_path(self) -> Path:
        """Resolve full ledger path."""
        return self.run_dir / self.ledger_db_name

    @property
    def lessons_db_path(self) -> Path:
        """Resolve full lessons database path."""
        return self.run_dir / "lessons.db"


# Export global settings instance
settings = Settings()
