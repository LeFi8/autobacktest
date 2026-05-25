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
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Execute a vectorized backtest with lookahead-bias protection.

    Args:
        prices: Daily close prices DataFrame (index=DatetimeIndex).
        weights: Portfolio weights DataFrame (index=DatetimeIndex).

    Returns:
        tuple containing:
            - Daily portfolio returns (pd.Series)
            - Portfolio cumulative equity curve (pd.Series)
            - Aligned weights daily DataFrame (pd.DataFrame)
    """
```

### `calculate_turnover_and_costs`
Computes daily rebalancing turnover and penalizes returns by transaction costs.
```python
def calculate_turnover_and_costs(
    portfolio_returns: pd.Series,
    daily_weights: pd.DataFrame,
    prices: pd.DataFrame,
    cost_bps: float = 10.0,
) -> tuple[pd.Series, pd.Series, float]:
    """Calculate portfolio turnover and returns adjusted for transaction costs.

    Args:
        portfolio_returns: Raw daily portfolio returns.
        daily_weights: Aligned daily asset weights.
        prices: Daily asset close prices.
        cost_bps: Transaction cost penalty in basis points (1 bp = 0.0001).

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

Generated: 2026-05-25
