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
Primary coordinator running full training walk-forward windows and Out-of-Sample holdout checks.
```python
def evaluate_strategy(
    strategy_name: str,
    generate_signals_fn: Any,
    config: dict[str, Any],
    start_date: str = "2015-01-01",
    end_date: str = "2026-01-01",
) -> EvaluationReport:
    """Run full deterministic walk-forward & holdout evaluation lifecycle.

    Args:
        strategy_name: Identifier name of the strategy.
        generate_signals_fn: Dynamic weight generation method.
        config: Loaded strategy configuration dict.
        start_date: Backtesting starting boundary.
        end_date: Backtesting ending boundary.

    Returns:
        EvaluationReport: Structured dataclass enclosing all backtest, bootstrap,
                          regime, DSR metrics, and gate checklist decisions.
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

### `accept`
Evaluates target metrics and constraints against standard lexicographic criteria gates.
```python
def accept(
    report: EvaluationReport,
    baseline: EvaluationReport | None,
    target_metric: TargetMetric = TargetMetric.SHARPE,
    dd_limit: float | None = None,
    turnover_limit: float | None = None,
    min_improvement: float | None = None,
    config: Any = None,
) -> GateResult:
    """Evaluate candidate EvaluationReport against lexicographic gates.

    Hard constraints checked in sequence:
    1. Drawdown: holdout_metrics.max_drawdown <= dd_limit
    2. Regime tests: regime_passed is True
    3. Turnover: holdout_metrics.turnover <= turnover_limit

    If all pass, tie-breaker check:
    4. Target metric improvement over baseline by at least epsilon margin
       (if baseline is present)

    Note: Deflated Sharpe Ratio (DSR) is computed and stored on the report for
    overfitting insight but is NOT an active hard gate.

    Args:
        report: Candidate strategy EvaluationReport.
        baseline: Optional baseline comparison EvaluationReport.
        target_metric: Target metric to compare improvement.
        dd_limit: Maximum allowed holdout drawdown.
        turnover_limit: Maximum allowed holdout turnover rate.
        min_improvement: Epsilon required improvement margin.
        config: Optional configuration model or dict to resolve default gates limits.

    Returns:
        GateResult: Selection outcome specifying acceptance status and failure reasons.
    """
```

---

## 5. Storage Store & History Ledger (`autobacktest.ledger`)

### `LedgerStore`
SQLite relational database interface manager.
- `create_run(run_id: str, strategy_name: str, ...)`: Commits structured metadata for a new optimization session.
- `record_attempt(run_id: str, iteration: int, ...)`: Records attempt parameter values, performance metrics, gating decisions, and out-of-sample daily returns.
- `fetch_historical_returns(dataset_hash: str) -> tuple[pd.DataFrame, list[float]]`: Retrieves all historical return series and Sharpe ratios matching active dataset universe for DSR effective trials calculation.

---

## 6. Optimization Orchestrator (`autobacktest.orchestrator`)

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
    strategies_dir: Path = Path("strategies"),
    configs_dir: Path = Path("configs"),
    target_metric: TargetMetric = TargetMetric.SHARPE,
    repo_path: Path = Path(),
    start_date: str = "2015-01-01",
    end_date: str = "2026-01-01",
) -> OrchestratorResult:
    """Run the LLM-driven strategy optimization loop.

    Coordinates:
    1. Parsing target program.md guidelines.
    2. Evaluating the candidate baseline strategy as iteration 0.
    3. Generating structured code/config edits via LLM mutations.
    4. Executing static code analysis, dynamic signature, and lookahead preflight checks.
    5. Evaluates walk-forward window and holdout returns.
    6. Adjusts Sharpe ratio for multiple testing bias (DSR).
    7. Runs lexicographic gates and handles commits/rollbacks automatically.

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

    Returns:
        OrchestratorResult: Summary of the final optimization run outcomes.
    """
```


