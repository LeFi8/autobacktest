"""Central configuration management for AutoBacktest."""

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

    # --- SAFETY GATES ---
    max_file_size_kb: int = Field(default_factory=lambda: int(os.getenv("AUTOBACKTEST_MAX_FILE_SIZE_KB", "100")))
    max_cyclomatic_complexity: int = Field(
        default_factory=lambda: int(os.getenv("AUTOBACKTEST_MAX_CYCLOMATIC_COMPLEXITY", "15"))
    )
    max_function_lines: int = Field(default_factory=lambda: int(os.getenv("AUTOBACKTEST_MAX_FUNCTION_LINES", "100")))
    safe_imports_whitelist: str = Field(
        default_factory=lambda: os.getenv(
            "AUTOBACKTEST_SAFE_IMPORTS_WHITELIST",
            "pandas,numpy,math,typing,scipy,dataclasses,collections,itertools,functools,decimal,statistics,numbers,json",
        )
    )
    sandbox_timeout: int = Field(default_factory=lambda: int(os.getenv("AUTOBACKTEST_SANDBOX_TIMEOUT", "15")))

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


# Export global settings instance
settings = Settings()
