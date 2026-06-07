# API Reference

This document provides a comprehensive overview of the public interfaces and core utility libraries in AutoBacktest.

---

## 1. Data Provider Module (`autobacktest.data`)

The data layer is defined across three files: `base.py` (abstract contract), `cache.py` (caching decorator), and `yfinance_provider.py` (concrete Yahoo Finance implementation).

### `DataProvider` (Abstract Base Class, defined in `base.py`)
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
    borrow_cost_bps: float = 100.0,
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
        borrow_cost_bps: Short borrowing cost in basis points annualized (default: 100.0).

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
    observed_daily_returns: pd.Series,
    historical_sharpes: list[float] | None = None,
    effective_trials: int = 1,
) -> float:
    """Calculate the Deflated Sharpe Ratio (DSR) for a strategy's returns.

    Args:
        observed_daily_returns: Daily net returns series.
        historical_sharpes: Collection of Sharpe ratios from previous trial runs.
        effective_trials: Estimated independent trials (derived if historical_sharpes omitted).

    Returns:
        float: Deflated Sharpe Ratio (representing confidence level [0.0, 1.0]).
    """
```

### `run_block_bootstrap`
Performs stationary block bootstrapping to determine Sharpe ratio significance thresholds under Monte Carlo.
```python
def run_block_bootstrap(
    returns: pd.Series,
    n_paths: int = 10000,
    block_size: int = 21,
    seed: int | None = None,
) -> tuple[float, float, float, np.ndarray]:
    """Execute block bootstrap to calculate Sharpe ratio percentiles.

    Args:
        returns: Daily net returns series.
        n_paths: Number of simulation iterations (default: 10000).
        block_size: Block size for stationary bootstrap (default: 21 trading days).
        seed: Random state seed.

    Returns:
        tuple containing:
            - 5th percentile Sharpe ratio (float)
            - 50th percentile Sharpe ratio (float)
            - 95th percentile Sharpe ratio (float)
            - Array of bootstrapped Sharpe ratios (np.ndarray)
    """
```

### `partition_holdout_data`
Splits a DatetimeIndex into in-sample and out-of-sample holdout segments.
```python
def partition_holdout_data(
    index: pd.DatetimeIndex,
    holdout_years: int = 3,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
```

### `generate_walk_forward_windows`
Generates rolling walk-forward train/test date window tuples.
```python
def generate_walk_forward_windows(
    index: pd.DatetimeIndex,
    train_years: int = 5,
    test_years: int = 1,
    step_years: int = 1,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
```

### `evaluate_stress_regimes`
Calculates drawdowns during historical crash regimes and returns a passed indicator.
```python
def evaluate_stress_regimes(
    net_returns: pd.Series,
    daily_weights: pd.DataFrame | None = None,
    n_tickers: int = 0,
) -> tuple[dict[str, float], bool]:
```
Tests drawdown thresholds against three regimes: 2008 GFC (max 25% drawdown),
2020 COVID (max 15%), and 2022 bear market (max 20%). Also performs a
minimum-exposure check (flags >80% cash held for >10 consecutive trading days).

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

### `EvaluationReport` / `WindowReport` (defined in `report.py`)
Key dataclasses for structured performance output. `EvaluationReport` carries all aggregated metrics, holdout/shadowe returns, benchmark comparisons, and robustness diagnostics including:
- `pbo`: Probability of Backtest Overfitting (float, default `0.0`), computed by `calculate_pbo` via Combinatorially Symmetric Cross-Validation. See Section 15.
- `mc_sharpes`: Array of bootstrapped Sharpe ratios from Monte Carlo simulation.
- `holdout_net_returns`: Out-of-sample daily returns for holdout gate confirmation.
- `deflated_sharpe` / `holdout_deflated_sharpe`: In-sample and holdout DSR values.
- `in_sample_metrics` / `holdout_metrics`: `WindowReport` instances with Sharpe, Sortino, max drawdown, turnover, annualized return/volatility.

---

## 3. Strategy Registry & Pre-flight Validation (`autobacktest.strategy`)

### `preflight`
Runs all pre-flight validations on a target strategy and configuration.
```python
def preflight(
    strategy_name: str,
    strategies_dir: Path,
    configs_dir: Path,
) -> ValidationResult:
    """Run all pre-flight validations on a target strategy and config.

    Validations include:
    1. Path traversal security check
    2. AST static whitelist check to prevent imports of unauthorized packages
    3. Pydantic configuration validation against StrategyConfig schema
    4. Dynamic compilation and import using isolated compilation blocks
    5. Smoke testing with synthetic prices to verify execution correctness
    6. Sub-window rebalance stability validation for lookahead bias sniffing
    7. Cyclomatic complexity and function size bounds
    8. Undefined-name AST scan — catches LLM hallucinations referencing
       out-of-scope variables or misspelled identifiers

    Args:
        strategy_name: Name of the strategy to validate.
        strategies_dir: Path to directory containing strategy modules.
        configs_dir: Path to directory containing YAML configs.

    Returns:
        ValidationResult: Pass/fail outcome and diagnostic errors if rejected.
    """
```

### `StrategyConfig`
Unified configuration validator inheriting from `pydantic.BaseModel` with ``extra="forbid"`` — strategy-specific custom parameters must be placed in the ``params`` dictionary to prevent injection of arbitrary top-level keys.

**Fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `universe` | `list[str]` | (required) | List of asset tickers (min 1) |
| `benchmark` | `str` | `"SPY"` | Benchmark index ticker |
| `momentum_lookback` | `int` | `12` | Momentum lookback window (months, >= 1) |
| `max_drawdown_limit` | `float` | `0.20` | Max permitted drawdown [0, 1] |
| `turnover_limit` | `float` | `2.0` | Max annualized turnover (>= 0, <= 10) |
| `borrow_cost_bps` | `float` | `100.0` | Annualized short borrowing cost in bps |
| `cscv_blocks` | `int` | `10` | Blocks for CSCV PBO calculation (>= 4) |
| `pbo_limit` | `float\|None` | `None` | PBO ceiling for select gate |
| `cscv_embargo_days` | `int` | `5` | CSCV block embargo days |
| `adaptive_slippage` | `bool` | `False` | Use volatility-adaptive slippage |
| `slippage_vol_window` | `int` | `21` | Volatility window for adaptive slippage |
| `slippage_vol_cap` | `float` | `3.0` | Volatility cap multiplier for adaptive slippage |
| `mc_bootstrap_method` | `str` | `"stationary"` | Bootstrap method: ``"circular"`` or ``"stationary"`` |
| `regime_benchmark` | `str\|None` | `None` | Alternative benchmark for launch-regime timing haircut |
| `params` | `dict[str, Any]` | `{}` | Strategy-specific custom parameters |
| `min_improvement` | `float` | `0.0` | Min target-metric improvement epsilon for select gate |
| `select_min_return_ratio` | `float` | `0.5` | Min fraction of baseline return for select gate |
| `require_dsr_non_degradation` | `bool` | `True` | Enforce DSR non-degradation in select gate |
| `holdout_min_improvement` | `float` | `0.0` | Tolerance for holdout DSR non-degradation |
| `enable_holdout_confirmation` | `bool` | `True` | If True, select-passing candidates confirmed on holdout |
| `dsr_floor` | `float\|None` | `None` | Optional absolute DSR floor (reserved) |

**Methods:**
- `from_yaml(path: Path) -> StrategyConfig`: Parses YAML file and instantiates schema.
- `to_flat_dict() -> dict[str, Any]`: Flattens configurations including sub-parameter schemas to single-depth dictionary.
- `constraints_text() -> str`: Renders schema field constraints and default values as text for LLM prompt injection.

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
Analyzes whether a strategy candidate produces returns that are functionally identical or highly correlated to any previous optimization trial. Uses raw (signed) Pearson correlation — negatively correlated candidates (diversification benefit) pass the gate.
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
        min_overlap_days: Minimum overlapping trading days required to compute
            a meaningful correlation (default: 60).

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
    min_return_ratio: float | None = None,
    pbo_limit: float | None = None,
    config: Any = None,
) -> GateResult:
    """In-sample selection gate.

    Hard constraints (in-sample walk-forward aggregate):
    1. Max drawdown <= dd_limit (resolved from config or default 0.20)
    2. Regime stress tests: regime_passed is True
    3. Turnover <= turnover_limit (resolved from config or default 2.0)
    4. PBO limit (when specified): report.pbo <= pbo_limit

    If all pass, tie-breaker (when baseline is present):
    5. Target metric improvement: candidate > baseline + min_improvement
    6. Annualized return >= baselines annualized return * min_return_ratio (default 0.5)
    7. DSR non-degradation: candidate's in-sample DSR does not degrade below baseline's
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
    min_return_ratio: float | None = None,
    pbo_limit: float | None = None,
    config: Any = None,
) -> GateResult:
    """Backward-compatible wrapper: composes select + confirm.
    The require_dsr_non_degradation, min_return_ratio, and pbo_limit
    parameters are propagated to select.
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

### `EventLog`
Structured JSON events logging history.
```python
class EventLog:
    def __init__(self, path: Path) -> None: ...
    def write(self, record: dict[str, Any]) -> None:
        """Append one JSON line with timestamp."""
    def close(self) -> None: ...
```

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
    dd_limit: float = 0.20  # max drawdown limit for selection gate
    turnover_limit: float = 2.0  # max turnover limit for selection gate
    min_return_ratio: float = 0.5  # min fraction of baseline return for selection gate
    last_iteration_failures: list[dict[str, Any]] | None = None  # failure details from previous iteration
    explored_config_summary: str = ""  # summary of parameter values already explored
    directive: str = ""  # optional high-level instruction passed to the LLM
    repair_request: dict[str, Any] | None = None  # details about a code repair attempt
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
    cached_tokens: int = 0  # tokens served from prompt cache
```

### `build_messages`
Builds the system and user message payload for the LLM completion API. Injects system prompt, program objective, lessons, strategy code, config YAML, evaluation report, attempt history, and mode-specific instructions. Supports Anthropic-style `cache_control` breakpoints when `cache_supported=True`.

```python
def build_messages(
    context: AgentContext,
    cache_supported: bool = False,
) -> list[dict[str, Any]]:
    """Build message payload. The stable prefix (SYSTEM_PROMPT + program_text) is
    byte-identical across iterations, enabling server-side prompt caching."""
```

### `parse_lessons` / `filter_lessons`
Parse and filter the lessons markdown content for context-specific injection.
```python
def parse_lessons(lessons_text: str) -> list[dict[str, str]]:
    """Parse lessons.md content into dicts with keys: title, type, body."""

def filter_lessons(lessons_text: str, context_stage: str | None) -> str:
    """Filter lessons by type matching active stage (BUG, DIVERSITY, GATE_REJECTION)."""
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
    early_stop_patience: int = settings.early_stop_patience,
    resume: str | None = None,
    quiet: bool = False,
) -> OrchestratorResult:
    """Run the LLM-driven strategy optimization loop.

    Generates N parallel LLM candidate edits per iteration, validates via
    preflight checks, diversity gates (config similarity + returns correlation),
    two-phase gate system (select + confirm), and commits winners to git.

    Early-stops when ``consecutive_no_accept >= early_stop_patience > 0``.
    Set ``AUTOBACKTEST_EARLY_STOP_PATIENCE=0`` to disable early stopping.

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
            Configurable via ``AUTOBACKTEST_EARLY_STOP_PATIENCE`` env var.
            Set to 0 to disable.
        resume: Run ID to resume a previously interrupted optimization.
        quiet: Suppress non-critical warnings and reduce terminal noise
            during the optimization loop (default: False).

    Returns:
        OrchestratorResult: Summary of the final optimization run outcomes.
        ``early_stopped`` is True when the loop exited early due to
        ``early_stop_patience`` consecutive rejections or the holdout-peek
        budget being exhausted.
    """
```

---

### `_LRUCache` (internal)
Thread-safe LRU cache used by the orchestrator to memoize expensive
evaluation results.  Evicts the least-recently-used entry when the cache
exceeds ``maxsize`` (default 36).  Provides the same ``__getitem__`` /
``__setitem__`` / ``__contains__`` / ``get`` interface as a standard
``dict`` so it can be passed transparently as ``_eval_cache`` to
``evaluate_strategy_detailed``.

### `OrchestratorResult`
Dataclass returned by ``run_optimization`` summarizing the full run outcome.
```python
@dataclass
class OrchestratorResult:
    run_id: str
    branch: str
    n_committed: int
    final_report: EvaluationReport
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost: float = 0.0
    baseline_report: EvaluationReport | None = None
    early_stopped: bool = False
    early_stop_iteration: int | None = None
```
- `run_id`: Unique identifier for the run (``{strategy}-{YYYYMMDD}-{HHMMSS}``).
- `branch`: Git branch name created for this run.
- `n_committed`: Number of candidate improvements committed to git.
- `final_report`: Evaluation report of the final (incumbent) strategy.
- `total_prompt_tokens`: Aggregate LLM prompt tokens consumed.
- `total_completion_tokens`: Aggregate LLM completion tokens consumed.
- `total_cost`: Total cost of all LLM calls in USD.
- `baseline_report`: Evaluation report of the pre-optimization baseline.
- `early_stopped`: ``True`` when the loop exited early due to ``early_stop_patience`` consecutive rejections or the holdout-peek budget being exhausted.
- `early_stop_iteration`: The 1-indexed iteration at which early-stop fired, or ``None`` if the run completed normally.

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
- `llm_prompt_cache`: Boolean flag to enable Anthropic-style prompt caching (default: `True`).
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
- `early_stop_patience`: Consecutive iteration rejections allowed before early-stop terminates the run (default: `10`). Set `AUTOBACKTEST_EARLY_STOP_PATIENCE=0` to disable.
- `max_file_size_kb`: Maximum allowed candidate code file length (default: `100`).
- `max_cyclomatic_complexity`: Maximum cyclomatic complexity allowed for functions (default: `25`).
- `max_function_lines`: Maximum physical lines allowed for functions (default: `100`).
- `safe_imports_whitelist`: Comma-separated allowed module imports.
- `sandbox_timeout`: Strategy signal execution limit integer (default: `15`).
- `enable_codemod_repair`: Boolean flag to enable automatic pandas deprecated API repair (default: `True`).
- `enable_llm_repair`: Boolean flag to enable multi-attempt LLM code repair on failed preflight (default: `True`).
- `max_repair_attempts`: Maximum LLM repair attempts per candidate (default: `2`).
- `enable_config_diversity_gate`: Boolean flag to enable the config similarity gate in explore mode (default: `True`).
- `enable_config_jitter`: Boolean flag to enable config jittering as diversity gate salvage (default: `True`).
- `config_jitter_max_attempts`: Maximum mutation attempts per jitter cycle (default: `12`).
- `config_jitter_rel_step`: Relative perturbation step size for jittering (default: `0.15`).
- `enable_json_salvage`: Boolean flag to enable JSON salvage from malformed LLM responses (default: `True`).
- `enable_candidate_directives`: Boolean flag to pass high-level directives to the LLM (default: `True`).
- `enable_explored_config_injection`: Boolean flag to inject explored config summaries into LLM context (default: `True`).
- `explored_config_max_configs`: Max configs included in explored config summary (default: `30`).
- `enable_identical_behavior_guard`: Boolean flag to guard against functionally identical signal outputs (default: `True`).
- `identical_behavior_epsilon`: Maximum absolute weight difference below which signals are considered identical (default: `1e-6`).
- `diversity_config_threshold`: Config similarity threshold for Tier 1 diversity gate (default: `0.95`).
- `diversity_returns_threshold`: Returns correlation threshold for Tier 2 diversity gate (default: `0.95`).
- `quiet`: Suppress non-critical warnings (default: `False`).
- `db_timeout`: Database block lock timeout limit (default: `15.0`).
- `parsed_safe_imports` (property): Resolved set of whitelisted import names.
- `ledger_db_path` (property): Resolved full Path to `ledger.db`.
- `lessons_db_path` (property): Resolved full Path to `lessons.db`.

---

## 9. Program Parser Module (`autobacktest.program`)

### `ProgramSpec`
Dataclass holding structured objectives and constraints from a program file.
```python
@dataclass(frozen=True)
class ProgramSpec:
    objective: str      # text under # Objective
    constraints: str    # text under # Constraints
    raw_text: str       # full file content (passed to LLM as-is)
```

### `parse_program`
Parses a `program.md` file with required `# Objective` and `# Constraints` headers.
```python
def parse_program(path: Path) -> ProgramSpec:
    """Parse program.md, skipping fenced code blocks to prevent false header matching.
    Raises ValueError if required headers are missing."""
```

The parsed `raw_text` is passed verbatim to the LLM so formatting is preserved.

---

## 10. Strategy Codemod Repair (`autobacktest.strategy.codemod`)

### `repair_pandas_code`
AST-based repair module for deprecated pandas API calls. Transforms pandas 1.x/2.x patterns to pandas 3.x-compatible equivalents.

Context-sensitive frequency alias rules:
- DatetimeIndex operations (`resample`, `date_range`, `bdate_range`, `Grouper`) require new aliases: `'ME'`, `'BME'`, `'QE'`, `'YE'`, `'h'`, `'min'`, `'s'`.
- Period operations (`to_period`, `period_range`) require original codes: `'M'`, `'Q'`, `'Y'`.

Also repairs:
- `.groupby(axis=...)` → drops `axis` keyword
- `.fillna(method='ffill'/'pad')` → `.ffill()`
- `.fillna(method='bfill'/'backfill')` → `.bfill()`
- `.mean/sum/std/min/max(level=L)` → `.groupby(level=L).func()`

```python
def repair_pandas_code(code: str) -> tuple[str, list[str]]:
    """Parse code, apply pandas deprecation fixes. Returns (repaired_code, fix_descriptions).
    If no fixes apply, returns the exact original string."""
```

### `repair_strategy_code`
Chains all AST-based repair passes into a single operation, called by the orchestrator after LLM code generation. Runs in order:

1. `_PandasDeprecationTransformer` — pandas 3.x API migration
2. `_MissingImportInjector` — injects ``from typing import Any`` when ``Any`` is used in type annotations but not imported. Places the import after the module docstring to preserve ``__doc__``.
3. `_WeightRenormalizer` — injects ``.clip(lower=0.0).div(…).fillna(0.0)`` before the final ``return`` in ``generate_signals`` to prevent float-accumulation drift above the ``1.0 + 1e-5`` tolerance.

```python
def repair_strategy_code(code: str) -> tuple[str, list[str]]:
    """Run all AST-based repair passes and return the fixed source.
    Returns (repaired_code, list_of_fix_descriptions).
    If no fixes apply or input has SyntaxError, returns (code, [])."""
```

---

## 11. Strategy Normalization (`autobacktest.strategy.normalization`)

### `normalize_python_code`
Normalizes Python source code by stripping comments and docstrings, producing a stable hash key for the eval cache.
```python
def normalize_python_code(code: str) -> str:
    """Remove comments, docstrings, and standardize whitespace."""
```

---

## 12. Parameter Importance Tracking (`autobacktest.strategy.parameter_importance`)

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

## 13. Lessons Memory Store (`autobacktest.lessons`)

### `LessonStore`
SQLite-backed deduplicated lesson store with per-strategy filtering. Replaces the flat `lessons.md` file.
- `migrate_from_file(lessons_md_path: Path, strategy: str) -> int`: Imports entries from legacy `lessons.md` into the database.
- `ingest_markdown(markdown_text: str, strategy: str)`: Parses markdown lessons and inserts deduplicated entries.
- `get_filtered_markdown(strategy: str) -> str`: Renders stored lessons as markdown for LLM context.
- `close()`: Closes the database connection.

---

## 14. Reporting Module (`autobacktest.reports`)

### `plot_equity_curves`
Generates a Matplotlib cumulative returns comparison chart.
```python
def plot_equity_curves(
    baseline_returns: pd.Series,
    final_returns: pd.Series,
    run_id: str,
    output_dir: Path,
    benchmark_returns: pd.Series | None = None,
    benchmark_ticker: str = "SPY",
) -> Path:
    """Generate a comparison chart of cumulative returns and save as equity_curves.png.

    Args:
        baseline_returns: Daily returns for the baseline strategy.
        final_returns: Daily returns for the optimized strategy.
        run_id: Unique run identifier (used in chart title).
        output_dir: Directory to save the PNG.
        benchmark_returns: Optional benchmark daily returns series.
        benchmark_ticker: Ticker label for the benchmark (default ``"SPY"``).

    Returns:
        Path to the saved ``equity_curves.png`` file.
    """
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

---

## 15. Combinatorially Symmetric Cross-Validation (`autobacktest.evaluator.cscv`)

### `calculate_pbo`
Calculates the Probability of Backtest Overfitting (PBO) using the CSCV methodology from Bailey et al.
```python
def calculate_pbo(
    returns_matrix: pd.DataFrame,
    n_blocks: int = 10,
    embargo_days: int = 0,
) -> float | None:
    """Calculate the Probability of Backtest Overfitting (PBO).

    Partitions the returns matrix into n_blocks contiguous blocks, generates
    all C(n_blocks, n_blocks/2) train/test split combinations, and measures
    how often the in-sample winner's relative rank in out-of-sample falls
    below 0.5 (indicating overfitting).

    Args:
        returns_matrix: DataFrame where each column is one trial's daily net
            returns, rows are trading dates.
        n_blocks: Number of blocks to split the data into (default: 10).
        embargo_days: Number of trailing days to drop from each block to
            avoid boundary autocorrelation (default: 0). Automatically
            falls back to 0 if embargo consumes too much data.

    Returns:
        float | None: PBO in [0, 1]. Higher values indicate greater
        overfitting risk. Returns None when uncomputable (insufficient
        trials or data).
    """
```

---

## 16. SPA Test (`autobacktest.evaluator.spa`)

### `calculate_hansen_spa`
Calculates Hansen's Superior Predictive Ability (SPA) test p-values to audit
whether the best-performing candidate significantly outperforms the baseline
after correcting for data-snooping bias.
```python
def calculate_hansen_spa(
    benchmark_returns: pd.Series,
    alternative_returns: pd.DataFrame,
    n_paths: int = 1000,
    block_size: int = 21,
    seed: int = 42,
) -> dict[str, float]:
    """Calculate Hansen's SPA test p-values.

    Returns three p-value bounds (consistent, upper conservative, lower
    liberal) and the observed test statistic T_SPA using the Politis-Romano
    stationary bootstrap.

    Args:
        benchmark_returns: Daily returns of the benchmark strategy (iteration 0).
        alternative_returns: Daily returns of alternative candidates (iterations > 0).
        n_paths: Number of stationary bootstrap paths (default: 1000).
        block_size: Expected block size for stationary bootstrap (default: 21).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        dict with keys:
        - ``p_consistent``: Standard SPA p-value (threshold-centered).
        - ``p_upper``: Conservative p-value (assumes all alternatives have zero mean).
        - ``p_lower``: Liberal p-value (assumes underperforming alternatives have mean <= 0).
        - ``t_spa``: Observed SPA test statistic.
    """
```

---

## 17. Launch Regime Timing Haircut (`autobacktest.evaluator.regime`)

### `calculate_regime_haircut`
Calculates the Liu timing haircut based on the benchmark's rolling 252-day
return z-score at the strategy launch date. When the benchmark enters the
holdout period at a cyclical peak (positive z-score), a proportional haircut
of ``0.05 * z_score`` is applied to the strategy's performance metrics to
penalise lucky launch timing.
```python
def calculate_regime_haircut(
    benchmark_prices: pd.Series,
    launch_date: pd.Timestamp,
) -> float:
    """Calculate the Liu timing haircut based on benchmark z-score at launch_date.

    Args:
        benchmark_prices: Historical daily prices of the benchmark (e.g. SPY).
        launch_date: Strategy launch date (start of holdout).

    Returns:
        float: Haircut fraction [0, inf). 0.0 when no peak is detected,
        insufficient history exists, or the z-score is non-positive.
    """
```

---

## 18. Identical-Behavior Guard (`autobacktest.strategy.validator`)

### `compare_signals_to_incumbent`
Compares a candidate strategy's output weights against the incumbent
strategy on synthetic prices to guard against functionally identical signal
outputs. Used by the orchestrator's pre-backtest guard.
```python
def compare_signals_to_incumbent(
    _strategy_name: str,
    candidate_code: str,
    candidate_config_yaml: str,
    incumbent_code: str,
    strategies_dir: Path,
    configs_dir: Path,
    epsilon: float = 1e-6,
) -> tuple[bool, float]:
    """Compare candidate signals against incumbent on synthetic prices.

    Args:
        _strategy_name: Strategy name (for validation context).
        candidate_code: Source code of the candidate strategy.
        candidate_config_yaml: YAML config of the candidate.
        incumbent_code: Source code of the incumbent strategy.
        strategies_dir: Path to strategies directory.
        configs_dir: Path to configs directory.
        epsilon: Maximum absolute weight difference below which signals
            are considered identical (default: 1e-6).

    Returns:
        tuple[bool, float]: (is_identical, max_abs_diff) where
        is_identical is True when max_abs_diff < epsilon.
    """
```

---

## 19. Explored Space Summary (`autobacktest.strategy.diversity`)

### `summarize_explored_space`
Generates a markdown summary of previously explored parameter values to
guide the LLM toward untried regions of the configuration space. Used with
``AUTOBACKTEST_ENABLE_EXPLORED_CONFIG_INJECTION``.
```python
def summarize_explored_space(
    historical_configs: list[str],
    max_configs: int = 30,
) -> str:
    """Summarize explored configuration space as markdown.

    Args:
        historical_configs: Previously attempted config YAML strings.
        max_configs: Maximum configs to include in the summary (default: 30).

    Returns:
        str: Markdown-formatted list of tried parameter values, or empty
        string if no configs are provided.
    """
```

---

## 20. Security Constants (`autobacktest.strategy.constants`)

### `FORBIDDEN_NAMES`
A ``frozenset`` of approximately 73 Python identifiers that are strictly
blocked during AST preflight scanning. Covers:
- Dangerous builtins: ``exec``, ``eval``, ``compile``, ``open``, ``__import__``
- Dunder attribute access: ``__builtins__``, ``__class__`` (detected via format strings and attribute chains)
- NumPy file I/O: ``load``, ``save``, ``savez``, ``memmap``, ``fromfile``, ``tofile``
- Pandas file I/O: ``read_csv``, ``to_csv``, ``read_parquet``, ``to_parquet``, ``read_sql``, ``to_pickle``,
  ``read_html``, ``to_html``, ``read_excel``, ``to_excel``, ``read_json``, ``to_json``, and 25+ other methods
- General sandbox escapes: ``io``, ``lib``, ``npyio``, ``HDFStore``, ``ExcelWriter``, ``ExcelFile``
- I/O pattern aliases: ``get_handle``, ``DataSource``

Any strategy code referencing these names — at import level, inside string
constants, or as attribute chains — is rejected during preflight.

---

## 21. Config Jitter System (`autobacktest.strategy.config_jitter`)

### `jitter_config`
Deterministically mutates numeric config parameters to satisfy the config diversity gate.
```python
def jitter_config(
    config_yaml: str,
    tried_configs: list[str],
    threshold: float,
    *,
    seed: int,
    max_attempts: int = 12,
    rel_step: float = 0.15,
    importance: dict[str, Any] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Mutate numeric config values until config similarity falls below threshold.

    Perturbs numeric (int/float) parameters via bounded random steps.
    Respects Pydantic schema constraints and ``KNOWN_RANGES`` from the
    diversity module.  Uses importance-weighted step sizes when parameter
    importance data is available.

    Args:
        config_yaml: Proposed strategy configuration YAML string.
        tried_configs: Historical config YAML strings to compare against.
        threshold: Config similarity threshold to beat (from ``AUTOBACKTEST_DIVERSITY_CONFIG_THRESHOLD``).
        seed: Random seed for deterministic perturbation.
        max_attempts: Maximum mutation attempts (default: 12).
        rel_step: Base relative step size for perturbation (default: 0.15).
        importance: Optional mapping of param names to correlation stats
            from ``compute_parameter_importance``.

    Returns:
        tuple[str | None, dict]: ``(mutated_yaml, metadata_dict)`` where
        metadata contains ``jitter_applied``, ``attempts``,
        ``final_similarity``, and ``changed_params``.
    """
```



