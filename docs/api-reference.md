# API Reference

This document provides a comprehensive overview of the public interfaces and core utility libraries in AutoBacktest.

---

## 1. Data Provider Module (`autobacktest.data`)

### `DataProvider` (Abstract Base Class)
Abstract interface defining price data retrievers.
```python
class DataProvider(ABC):
    @abstractmethod
    def get_prices(
        self,
        tickers: list[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch close prices for a list of tickers over a date range.

        Args:
            tickers: List of ticker symbols.
            start: Start date string (YYYY-MM-DD).
            end: End date string (YYYY-MM-DD).
            interval: Data interval (e.g. "1d").

        Returns:
            pd.DataFrame: DataFrame with DatetimeIndex and columns as tickers.
        """
```

### `CachedDataProvider(DataProvider)`
Decorator that intercepts data queries to cache close price history as Apache Parquet files under `data/cache/` to minimize remote downloads.
- `__init__(provider: DataProvider, cache_dir: str = "data/cache")`

### `YFinanceProvider(DataProvider)`
Standard provider accessing the Yahoo Finance API via `yfinance` library.

---

## 2. Evaluation Engine (`autobacktest.evaluator`)

### `run_vectorized_backtest`
Vectorized daily return computation. Incorporates lookahead-bias protection by lagging weights by 1 day.
```python
def run_vectorized_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    asset_returns: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Execute a vectorized backtest with lookahead-bias protection.

    Args:
        prices: Daily close prices DataFrame (index=DatetimeIndex).
        weights: Portfolio weights DataFrame (index=DatetimeIndex).
        asset_returns: Pre-computed daily asset returns (prices.pct_change()).
            When provided, prices.pct_change() is skipped for efficiency.

    Returns:
        tuple containing:
            - Daily portfolio returns (pd.Series)
            - Portfolio cumulative equity curve (pd.Series)
            - Aligned weights daily DataFrame (pd.DataFrame)
    """
```

### `calculate_turnover_and_costs`
Computes daily rebalancing turnover and penalizes returns by transaction costs (commissions, bid-ask spreads, and market impact).
```python
def calculate_turnover_and_costs(
    daily_returns: pd.Series,
    daily_weights: pd.DataFrame,
    prices: pd.DataFrame,
    commission_bps: float = 5.0,
    spread_bps: float = 5.0,
    impact_coef: float = 0.0,
    *,
    asset_returns: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.Series, float]:
    """Calculate portfolio turnover and returns adjusted for transaction costs.

    Args:
        daily_returns: Daily portfolio gross returns series.
        daily_weights: Aligned daily asset weights.
        prices: Daily asset close prices.
        commission_bps: Commission fee in basis points (1 bp = 0.0001).
        spread_bps: Bid-ask spread in basis points.
        impact_coef: Market impact parameter (quadratic/linear cost).
        asset_returns: Pre-computed daily asset returns.

    Returns:
        tuple containing:
            - Daily net portfolio returns (pd.Series)
            - Net cumulative equity curve (pd.Series)
            - Total annualized turnover (float)
    """
```

### `calculate_psr_dsr`
Computes Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR) to account for data-snooping and multiple testing.
```python
def calculate_psr_dsr(
    net_returns: pd.Series,
    historical_sharpes: list[float] | None = None,
    effective_trials: int = 1,
    benchmark_sharpe: float = 0.0,
) -> float:
    """Calculate the Deflated Sharpe Ratio (DSR) for a strategy's returns.

    Args:
        net_returns: Daily net returns series.
        historical_sharpes: Collection of Sharpe ratios from previous trial runs.
        effective_trials: Estimated independent trials (derived if historical_sharpes omitted).
        benchmark_sharpe: Threshold target Sharpe ratio (default: 0.0).

    Returns:
        float: Deflated Sharpe Ratio (representing confidence level [0.0, 1.0]).
    """
```

### `run_block_bootstrap`
Performs stationary block bootstrapping to determine Sharpe ratio significance thresholds under Monte Carlo.
```python
def run_block_bootstrap(
    returns: pd.Series,
    n_paths: int = 1000,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Execute block bootstrap to calculate Sharpe ratio percentiles.

    Args:
        returns: Daily net returns series.
        n_paths: Number of simulation iterations.
        seed: Random state seed.

    Returns:
        tuple containing:
            - 5th percentile Sharpe ratio (float)
            - 50th percentile Sharpe ratio (float)
            - 95th percentile Sharpe ratio (float)
    """
```

### `evaluate_strategy`
Primary coordinator running full training walk-forward windows and Out-of-Sample holdout checks. Thin wrapper around `evaluate_strategy_detailed` that returns only the report.
```python
def evaluate_strategy(
    strategy_name: str,
    generate_signals_fn: Any,
    config: dict[str, Any] | StrategyConfig,
    start_date: str = settings.default_start_date,
    end_date: str = settings.default_end_date,
    **kwargs: Any,
) -> EvaluationReport:
```

### `evaluate_strategy_detailed`
Full evaluation returning both the report and the in-sample returns Series for DSR/deflation.
```python
def evaluate_strategy_detailed(
    strategy_name: str,
    generate_signals_fn: Any,
    config: dict[str, Any] | StrategyConfig,
    start_date: str = settings.default_start_date,
    end_date: str = settings.default_end_date,
    *,
    _prices: pd.DataFrame | None = None,
    _bench_returns: pd.Series | None = None,
    _eval_cache: dict[int, tuple[EvaluationReport, pd.Series]] | None = None,
    _strategy_code: str | None = None,
) -> tuple[EvaluationReport, pd.Series[Any]]:
    """Run full deterministic walk-forward & holdout evaluation lifecycle.

    Returns a tuple of (EvaluationReport, in_sample_net_returns Series).
    The report carries ``holdout_net_returns`` for the holdout confirmation
    gate; the second element is the in-sample basis used for the selection
    DSR and diversity correlation checks.

    Parameters prefixed with ``_`` are internal optimisations:
    * ``_prices`` / ``_bench_returns`` — pre-fetched data to avoid redundant API calls.
    * ``_eval_cache`` — memoization dict skipping re-evaluation of identical edits.
    * ``_strategy_code`` — source text required for the eval cache key.
    """
```

---

## 3. Strategy Registry & Pre-flight Validation (`autobacktest.strategy`)

### `preflight`
Runs all six pre-flight validations on a target strategy and configuration.
```python
def preflight(
    strategy_name: str,
    strategies_dir: Path,
    configs_dir: Path,
) -> ValidationResult:
    """Run all six pre-flight validations on a target strategy and config.

    Validations include:
    1. Path traversal security check
    2. AST static whitelist check to prevent imports of unauthorized packages
    3. Pydantic configuration validation against StrategyConfig schema
    4. Dynamic compilation and import using isolated compilation blocks
    5. Smoke testing with synthetic prices to verify execution correctness
    6. Sub-window rebalance stability validation for lookahead bias sniffing

    Args:
        strategy_name: Name of the strategy to validate.
        strategies_dir: Path to directory containing strategy modules.
        configs_dir: Path to directory containing YAML configs.

    Returns:
        ValidationResult: Pass/fail outcome and diagnostic errors if rejected.
    """
```

### `StrategyConfig`
Unified configuration validator inheriting from `pydantic.BaseModel`.
- `from_yaml(path: Path) -> StrategyConfig`: Parses YAML file and instantiates schema.
- `to_flat_dict() -> dict[str, Any]`: Flattens configurations including sub-parameter schemas to single-depth dictionary.

### Strategy Fingerprint & Diversity Validation (`autobacktest.strategy.diversity`)

#### `ConfigFingerprint`
Normalised representation of a strategy configuration for structural similarity comparison.
- `numeric_params`: Dict mapping parameter names to raw float values (e.g. `momentum_lookback`).
- `set_fields`: Dict mapping field names to sets of string members (e.g. `universe` list).

#### `extract_config_fingerprint`
Parses a raw YAML configuration string, flattens any nested parameters under the `params` key, and builds a stable `ConfigFingerprint`.
```python
def extract_config_fingerprint(config_yaml: str) -> ConfigFingerprint:
```

#### `config_similarity`
Calculates a weighted similarity score in `[0.0, 1.0]` between two configuration fingerprints.
```python
def config_similarity(a: ConfigFingerprint, b: ConfigFingerprint) -> float:
```
Calculates similarity via:
$$\text{Similarity} = 0.7 \times \text{CosineSimilarity}(\text{numeric\_params}) + 0.3 \times \text{MeanJaccard}(\text{set\_fields})$$
Aligned numeric vectors are constructed using min-max normalization via `KNOWN_RANGES`. Unmatched keys default to `0.5` to prevent similarity inflation.

#### `max_config_similarity`
Compares a proposed candidate configuration with a collection of historical configuration strings.
```python
def max_config_similarity(
    candidate_yaml: str,
    historical_ymls: list[str],
) -> float:
```
Normalizes unknown parameters by establishing dynamic sample-space boundaries across the entire set of compared configurations.

#### `check_returns_correlation`
Analyzes whether a strategy candidate produces returns that are functionally identical or highly correlated to any previous optimization trial.
```python
def check_returns_correlation(
    candidate_returns: pd.Series,
    historical_returns_matrix: pd.DataFrame,
    threshold: float = 0.90,
    min_overlap_days: int = 60,
) -> tuple[bool, float]:
    """Check returns correlation against historical backtests.

    Args:
        candidate_returns: Daily net returns of the active strategy candidate.
        historical_returns_matrix: DataFrame containing daily return columns of past attempts.
        threshold: Maximum permitted Pearson correlation coefficient (default: 0.90).
        min_overlap_days: Minimum overlapping trading days required to compute correlation.

    Returns:
        tuple containing:
            - passed (bool): True if candidate returns are sufficiently unique (all correlations <= threshold).
            - max_correlation (float): Maximum correlation coefficient observed.
    """
```

---


## 4. Gating Check & Metric Optimization (`autobacktest.gate`)

The gate system is split into two phases to prevent holdout overfitting:

### `select`
In-sample selection gate — evaluated on **every** candidate. The holdout is never consulted.
```python
def select(
    report: EvaluationReport,
    baseline: EvaluationReport | None,
    target_metric: TargetMetric = TargetMetric.SHARPE,
    dd_limit: float | None = None,
    turnover_limit: float | None = None,
    min_improvement: float | None = None,
    require_dsr_non_degradation: bool | None = None,
    config: Any = None,
) -> GateResult:
    """In-sample selection gate.

    Hard constraints (in-sample walk-forward aggregate):
    1. Max drawdown <= dd_limit (resolved from config or default 0.20)
    2. Regime stress tests: regime_passed is True
    3. Turnover <= turnover_limit (resolved from config or default 2.0)

    If all pass, tie-breaker (when baseline is present):
    4. Target metric improvement: candidate > baseline + min_improvement
    5. DSR non-degradation: candidate's in-sample DSR does not degrade below baseline's
       (configurable via require_dsr_non_degradation, always-on by default)
    """
```

### `confirm`
Holdout confirmation gate — only reached when `select` passes. Each call consumes one holdout peek.
```python
def confirm(
    report: EvaluationReport,
    baseline: EvaluationReport | None,
    dd_limit: float | None = None,
    turnover_limit: float | None = None,
    holdout_min_improvement: float | None = None,
    config: Any = None,
) -> GateResult:
    """Holdout confirmation gate.

    Hard constraints on holdout_metrics:
    1. Max drawdown <= dd_limit
    2. Turnover <= turnover_limit

    Confirmation (when baseline is present):
    3. Holdout DSR non-degradation with holdout_min_improvement tolerance
    """
```

### `accept` (backward-compatible wrapper)
Composes `select` + `confirm` as a single call for the standalone evaluation path.
```python
def accept(
    report: EvaluationReport,
    baseline: EvaluationReport | None,
    target_metric: TargetMetric = TargetMetric.SHARPE,
    dd_limit: float | None = None,
    turnover_limit: float | None = None,
    min_improvement: float | None = None,
    require_dsr_non_degradation: bool | None = None,
    config: Any = None,
) -> GateResult:
    """Backward-compatible wrapper: composes select + confirm.
    The require_dsr_non_degradation parameter is propagated to select.
    """
```

---

## 5. Storage Store & History Ledger (`autobacktest.ledger`)

### `LedgerStore`
SQLite relational database interface manager.
- `create_run(run_id, strategy_name, ...)`: Commits structured metadata for a new optimization session.
- `record_attempt(run_id, iteration, ...)`: Records attempt parameter values, performance metrics, gating decisions, and out-of-sample daily returns.
- `fetch_historical_returns(dataset_hash) -> tuple[pd.DataFrame, list[float]]`: Retrieves all historical return series and Sharpe ratios matching active dataset universe for DSR effective trials calculation.
- `fetch_configs(dataset_hash) -> list[str]`: Retrieves historical YAML config strings for config diversity checks.
- `fetch_attempt_summaries(dataset_hash) -> list[dict]`: Builds summarized attempt history for LLM context.
- `fetch_holdout_history(dataset_hash) -> tuple[pd.DataFrame, list[float]]`: Retrieves holdout return streams for DSR deflation (peek count tracking).
- `fetch_param_importance_data(dataset_hash)`: Returns configs and metrics for Spearman rank correlation analysis.

---

## 6. LLM Provider Layer (`autobacktest.llm`)

### `LLMProvider` (Abstract Base Class)
Contract for LLM drivers that consume context and produce structured edits.
```python
class LLMProvider(ABC):
    temperature: float

    @abstractmethod
    def generate_edit(self, context: AgentContext) -> AgentEdit:
        """Consume context and generate strategy modifications.

        Raises:
            LLMError: If LLM service, parser, or network fails.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the unique string identification of the provider."""
```

### `AgentContext`
Immutable input dataclass provided to the LLM agent for strategy generation.
```python
@dataclass(frozen=True)
class AgentContext:
    strategy_name: str
    strategy_code: str
    config_yaml: str
    program_text: str
    evaluation_report: EvaluationReport | None
    iteration: int
    lessons_text: str = ""
    n_historical_configs: int = 0
    last_attempt: dict[str, Any] | None = None
    attempt_history: list[dict[str, Any]] | None = None
    mode: str = "explore"  # "explore" | "exploit"
```

### `AgentEdit`
Immutable structured edit returned by the LLM driver.
```python
@dataclass(frozen=True)
class AgentEdit:
    strategy_code: str
    config_yaml: str
    reasoning: str
    raw_response: str
    lessons_text: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
```

### `LLMError`
Domain exception raised when an LLM provider fails.
```python
class LLMError(Exception):
    def __init__(self, provider: str, model: str, detail: str,
                 retryable: bool = True, finish_reason: str | None = None):
```

---

## 7. Optimization Orchestrator (`autobacktest.orchestrator`)

### `run_optimization`
Fires and coordinates the iterative quantitative strategy optimization process.
```python
def run_optimization(
    program_path: Path,
    strategy_name: str,
    iterations: int,
    provider: LLMProvider,
    run_dir: Path,
    *,
    strategies_dir: Path = settings.strategies_dir,
    configs_dir: Path = settings.configs_dir,
    target_metric: TargetMetric = TargetMetric.SHARPE,
    repo_path: Path = Path(),
    start_date: str = settings.default_start_date,
    end_date: str = settings.default_end_date,
    holdout_peek_limit: int = 20,
    early_stop_patience: int = 10,
    resume: str | None = None,
) -> OrchestratorResult:
    """Run the LLM-driven strategy optimization loop.

    Generates 3 parallel LLM candidate edits per iteration, validates via
    preflight checks, diversity gates (config similarity + returns correlation),
    two-phase gate system (select + confirm), and commits winners to git.

    Args:
        program_path: Path to the markdown program objective file.
        strategy_name: The name of the target strategy to optimize.
        iterations: Total optimization runs to execute.
        provider: Enclosing LiteLLM/Mock provider to generate candidate mutations.
        run_dir: Directory where the run database and events log are written.
        strategies_dir: Directory enclosing target strategy modules.
        configs_dir: Directory enclosing YAML parameters files.
        target_metric: Metric choice to target during gate checks.
        repo_path: Root repository path for Git workspace operations.
        start_date: Starting date boundary for evaluation.
        end_date: Ending date boundary for evaluation.
        holdout_peek_limit: Maximum holdout peeks before early termination.
        early_stop_patience: Consecutive rejections before early stopping.
        resume: Run ID to resume a previously interrupted optimization.

    Returns:
        OrchestratorResult: Summary of the final optimization run outcomes.
    """
```

---

## 8. Centralized Configuration Module (`autobacktest.config`)

### `Settings`
Manages system configuration settings loaded from environment variables with safe fallbacks and Pydantic validation.
- `llm_provider`: LiteLLM/Mock provider type string (default: `"litellm"`).
- `llm_model`: Targeted LLM model identifier string (default: `"openai/gpt-4o"`).
- `llm_temperature`: Generation temperature float parameter (default: `0.7`).
- `llm_max_tokens`: Token boundary limit integer (default: `4096`).
- `litellm_debug`: Boolean flag to toggle deep LiteLLM logs.
- `llm_request_timeout`: Maximum duration in seconds for LLM call requests (default: `600.0`).
- `default_start_date`: Standard backtesting starting date string (default: `"2015-01-01"`).
- `default_end_date`: Standard backtesting ending date string (default: `"2026-01-01"`).
- `default_holdout_years`: Holdout dataset division length integer (default: `3`).
- `run_dir`: Output database and event log files folder path.
- `cache_dir`: Local daily price files local cache folder path.
- `strategies_dir`: Strategies folder path.
- `configs_dir`: Configuration templates folder path.
- `ledger_db_name`: Relational ledger storage filename.
- `n_candidates`: Number of parallel LLM candidates per iteration (default: `3`).
- `importance_min_attempts`: Minimum attempts required for parameter importance computation (default: `6`).
- `importance_p_threshold`: P-value threshold for significance in parameter importance (default: `0.20`).
- `max_file_size_kb`: Maximum allowed candidate code file length (default: `100`).
- `max_cyclomatic_complexity`: Maximum cyclomatic complexity allowed for functions (default: `15`).
- `max_function_lines`: Maximum physical lines allowed for functions (default: `100`).
- `safe_imports_whitelist`: Comma-separated allowed module imports.
- `sandbox_timeout`: Strategy signal execution limit integer (default: `15`).
- `db_timeout`: Database block lock timeout limit (default: `15.0`).
- `parsed_safe_imports` (property): Resolved set of whitelisted import names.
- `ledger_db_path` (property): Resolved full Path to `ledger.db`.
- `lessons_db_path` (property): Resolved full Path to `lessons.db`.

---

## 9. Strategy Normalization (`autobacktest.strategy.normalization`)

### `normalize_python_code`
Normalizes Python source code by stripping comments and docstrings, producing a stable hash key for the eval cache.
```python
def normalize_python_code(code: str) -> str:
    """Remove comments, docstrings, and standardize whitespace."""
```

---

## 10. Parameter Importance Tracking (`autobacktest.strategy.parameter_importance`)

### `compute_parameter_importance`
Computes Spearman rank correlation between numeric config parameters and the target metric across all attempts.
```python
def compute_parameter_importance(
    flat_configs: list[dict[str, Any]],
    metrics: list[float],
    min_attempts: int = 6,
    p_threshold: float = 0.20,
) -> dict[str, dict[str, float]]:
    """Returns dict mapping parameter names to {spearman_r, p_value} for significant correlations."""
```

### `format_importance_lessons`
Formats significant parameter correlations into lesson text for LLM consumption.
```python
def format_importance_lessons(importance: dict[str, dict[str, float]]) -> str:
```

---

## 11. Lessons Memory Store (`autobacktest.lessons`)

### `LessonStore`
SQLite-backed deduplicated lesson store with per-strategy filtering. Replaces the flat `lessons.md` file.
- `migrate_from_file(lessons_md_path: Path, strategy: str) -> int`: Imports entries from legacy `lessons.md` into the database.
- `ingest_markdown(markdown_text: str, strategy: str)`: Parses markdown lessons and inserts deduplicated entries.
- `get_filtered_markdown(strategy: str) -> str`: Renders stored lessons as markdown for LLM context.
- `close()`: Closes the database connection.

---

## 12. Reporting Module (`autobacktest.reports`)

### `plot_equity_curves`
Generates a Matplotlib cumulative returns comparison chart.
```python
def plot_equity_curves(
    baseline_returns: pd.Series,
    final_returns: pd.Series,
    run_id: str,
    output_dir: Path,
) -> Path:
    """Generate a comparison chart of cumulative returns and save as equity_curves.png."""
```

### `compile_failure_summary`
Aggregates candidate rejections and compiler exceptions from the events log.
```python
def compile_failure_summary(run_dir: Path) -> dict[str, Any]:
    """Parse events.jsonl and compile failure statistics."""
```

### `compile_strategy_report`
Compiles the self-contained institutional quantitative Markdown strategy report.
```python
def compile_strategy_report(
    baseline_report: EvaluationReport,
    final_report: EvaluationReport,
    run_id: str,
    output_dir: Path,
    program_text: str,
    config_yaml: str,
    failure_summary: dict[str, Any],
    strategy_code: str,
) -> Path:
    """Generate and write strategy_report.md inside the run directory."""
```



