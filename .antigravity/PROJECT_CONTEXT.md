# Project Context: AutoBacktest

## High-Level Mission
Autonomous AI-driven quant strategy optimization loop. LLM agent refines trading strategy code and configurations; deterministic evaluator runs walk-forward and holdout backtests, asserting lexicographic gating criteria; SQLite/git-backed tracking ledger commits improvements or rolls back regressions.

## Core Target Audience
- Quantitative developers and algorithmic traders.
- Autonomous agentic optimization systems running local loops.

## Core Business Operations
- **Dynamic Signal Generation**: Code execution of register-based strategy files under `strategies/`.
- **Rigorous Evaluation Lifecycle**: Backtesting daily return series, factoring transaction costs/turnover, computing Deflated Sharpe Ratio (DSR), running block bootstrapping (Monte Carlo), stress-testing historical regimes.
- **Gating Check Logic**: Multi-threshold validation determining strategy acceptance (drawdown, turnover, DSR, regime stress, etc.).
- **Ledger & Version Control Tracking**: Committing code states via Git; tracking historical iterations and metrics in SQLite.

## Directory Map
```
autobacktest/
├── .antigravity/                       # Machine-readable AI agent context manifests
│   ├── AGENT_GUIDELINES.md             # Coding conventions and routing guidelines
│   ├── PROJECT_CONTEXT.md              # Project mission and logical workflows
│   ├── STATE_MEMORY.md                 # Ledger of implementation milestones and debt
│   └── TECH_STACK.md                   # Approved dependencies and version boundaries
├── configs/                            # YAML parameter configurations for strategies
│   └── haa.yaml
├── data/                               # Daily price caches (Parquet)
├── src/
│   └── autobacktest/                   # Core system package
│       ├── cli.py                      # Typer CLI application entry point
│       ├── config.py                   # Central configuration management & Settings
│       ├── orchestrator.py             # Optimization loop orchestrator
│       ├── gate.py                     # Lexicographic optimization gate checking
│       ├── program.py                  # Objective markdown parser
│       ├── data/                       # Market data sourcing & caching
│       │   ├── base.py
│       │   ├── cache.py
│       │   └── yfinance_provider.py
│       ├── evaluator/                  # Vectorized backtester & risk calculation
│       │   ├── backtest.py             # Vectorized return generation
│       │   ├── costs.py                # Rebalancing costs and turnover
│       │   ├── deflated_sharpe.py      # DSR / PSR calculations
│       │   ├── evaluate.py             # Master walk-forward / holdout evaluation orchestrator
│       │   ├── holdout.py              # In-sample vs Out-of-sample data splitting
│       │   ├── monte_carlo.py          # Stationary block bootstrap simulation
│       │   ├── regime.py               # Stress regime tracking and limits
│       │   ├── report.py               # Dataclass schemas for outputs
│       │   └── walk_forward.py         # Rolling optimization training window gen
│       ├── ledger/                     # SQLite database and git state logger
│       │   ├── __init__.py
│       │   ├── event_log.py            # Event logging definitions
│       │   ├── git_ops.py              # Git branch, commit, and checkout utils
│       │   └── store.py                # SQLite database and DAO methods
│       ├── llm/                        # LiteLLM driver wrapper for strategy mutation
│       │   ├── __init__.py
│       │   ├── base.py                 # Abstract base classes and schemas
│       │   ├── litellm_provider.py     # LiteLLM structured output provider
│       │   ├── mock_provider.py        # Mock provider for offline testing
│       │   └── prompts.py              # Prompts and structured schema definitions
│       ├── lessons/                    # Structured lesson memory (SQLite)
│       │   ├── __init__.py
│       │   └── store.py                # LessonStore: dedup, import/export
│       ├── llm/                        # LLM driver wrapper for strategy mutation
│       │   ├── __init__.py
│       │   ├── base.py                 # Abstract base classes and schemas
│       │   ├── litellm_provider.py     # LiteLLM structured output provider
│       │   ├── mock_provider.py        # Mock provider for offline testing
│       │   └── prompts.py              # Prompts and structured schema definitions
│       ├── reports/                    # Report generation utilities
│       │   ├── __init__.py
│       │   └── generator.py            # Equity plots, failure summaries, strategy reports
│       └── strategy/                   # Schemas and strategy parsing utilities
│           ├── __init__.py
│           ├── config_schema.py        # Pydantic v2 strategy parameter schema
│           ├── contract.py             # Strategy function signature contract
│           ├── diversity.py            # Config fingerprint + returns correlation
│           ├── normalization.py        # Code normalization for eval cache keys
│           ├── parameter_importance.py # Spearman rank correlation tracking
│           └── validator.py            # Static AST check and pre-flight validation
├── strategies/                         # Quant strategy Python source files
│   └── haa.py                          # Historical Asset Allocation strategy signals
├── tests/                              # Pytest test suite
└── pyproject.toml                      # Project definition & strict configs
```

## Logical User Flows
1. **User Run Command**: User executes CLI specifying objective file and target strategy:
   ```bash
   uv run autobacktest run --program program.md --strategy haa --iterations 10
   ```
2. **Setup and baseline**: System runs standard evaluation on baseline strategy configuration; records scores in tracking ledger.
3. **Iterative optimization loop**:
   a. **LLM Mutation**: LiteLLM consumes current strategy source, `program.md` goals, previous execution failures, and generates strategy modifications.
   b. **Execution & Backtest**: System imports modified strategy module dynamically, executing `generate_signals()`.
   c. **Metric Evaluation**: Generates 5y-train/1y-test walk-forward reports plus 3y holdout report. Evaluates turnover, max drawdown, deflated Sharpe ratio (DSR >= 0.95), Monte Carlo 5th-95th percentile validation.
   d. **Gate Validation**: Evaluates gates. If gates fail, rolls back code changes. If gates pass, commits strategy code to git and writes run stats to SQLite.
4. **Leaderboard Report**: CLI report prints tabular runs leaderboard.


