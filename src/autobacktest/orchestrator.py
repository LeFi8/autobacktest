"""Top-level orchestration loop."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from autobacktest.evaluator.deflated_sharpe import (
    calculate_effective_trials,
    calculate_psr_dsr,
)
from autobacktest.evaluator.evaluate import evaluate_strategy_detailed
from autobacktest.evaluator.report import EvaluationReport
from autobacktest.gate import TargetMetric, accept
from autobacktest.ledger.event_log import EventLog
from autobacktest.ledger.git_ops import GitLedger
from autobacktest.ledger.store import LedgerStore
from autobacktest.llm.base import AgentContext, AgentEdit, LLMError, LLMProvider
from autobacktest.program import parse_program
from autobacktest.strategy.config_schema import StrategyConfig
from autobacktest.strategy.validator import preflight


@dataclass
class OrchestratorResult:
    run_id: str
    branch: str
    n_committed: int
    final_report: EvaluationReport


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
    """Run the LLM-driven strategy optimization loop."""
    # 1. Parse program
    spec = parse_program(program_path)

    # 2. Ensure strategy/config files exist
    strat_path = strategies_dir / f"{strategy_name}.py"
    cfg_path = configs_dir / f"{strategy_name}.yaml"
    if not strat_path.exists():
        raise FileNotFoundError(f"Strategy file not found: {strat_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    # 3. Setup git
    git_ledger = GitLedger(repo_path)
    git_ledger.ensure_clean(strategy_name)
    run_id = f"{strategy_name}-{datetime.now(tz=UTC):%Y%m%d-%H%M%S}"
    branch = git_ledger.create_run_branch(run_id)

    # 4. Setup ledger and event log — wrapped in try/finally so they are always
    # closed even if baseline evaluation or any loop iteration raises.
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger = LedgerStore(run_dir / "ledger.db")
    event_log = EventLog(run_dir / run_id / "events.jsonl")
    incumbent_returns = pd.Series(dtype=float)  # set before try so finally can read it
    incumbent: EvaluationReport  # assigned during baseline below
    try:
        # 5. Load baseline config
        config_obj = StrategyConfig.from_yaml(cfg_path)
        config = config_obj.model_dump()

        # 6. Record run metadata
        dataset_hash = _compute_dataset_hash(config)
        ledger.create_run(
            run_id=run_id,
            strategy_name=strategy_name,
            program_path=str(program_path),
            provider=provider.provider_name,
            model=getattr(provider, "model", "unknown"),
            branch=branch,
            dataset_hash=dataset_hash,
            iterations=iterations,
            started_at=datetime.now(tz=UTC).isoformat(),
        )

        # 7. Baseline evaluation (iteration 0 — not written to events.jsonl)
        baseline_fn = _load_signals(strat_path)
        baseline_report, baseline_returns = evaluate_strategy_detailed(
            strategy_name, baseline_fn, config, start_date=start_date, end_date=end_date
        )
        _deflate(baseline_report, baseline_returns, ledger)
        incumbent = baseline_report
        incumbent_returns = baseline_returns
        baseline_sha: str | None = git_ledger._repo.head.commit.hexsha
        ledger.record_attempt(
            run_id=run_id,
            iteration=0,
            strategy_name=strategy_name,
            dataset_hash=dataset_hash,
            config_yaml=cfg_path.read_text(encoding="utf-8"),
            observed_sharpe=baseline_report.observed_sharpe,
            deflated_sharpe=baseline_report.deflated_sharpe,
            target_metric=target_metric.value,
            target_metric_value=_get_metric_value(baseline_report, target_metric),
            holdout_max_drawdown=baseline_report.holdout_metrics.max_drawdown,
            holdout_turnover=baseline_report.holdout_metrics.turnover,
            regime_passed=baseline_report.regime_passed,
            accepted=True,
            committed=True,
            commit_sha=baseline_sha,
            rejection_reason=None,
            report_json=baseline_report.to_json(),
            holdout_returns=baseline_returns,
        )

        # 8. Optimization loop
        n_committed = 0
        for k in range(1, iterations + 1):
            event: dict[str, object] = {
                "iteration": k,
                "strategy": strategy_name,
            }

            # 8a. Build context
            current_code = strat_path.read_text(encoding="utf-8")
            current_yaml = cfg_path.read_text(encoding="utf-8")
            ctx = AgentContext(
                strategy_name=strategy_name,
                strategy_code=current_code,
                config_yaml=current_yaml,
                program_text=spec.raw_text,
                evaluation_report=incumbent,
                iteration=k,
            )

            # 8b. Get LLM edit
            try:
                edit = provider.generate_edit(ctx)
            except LLMError as e:
                event["llm_error"] = str(e)
                event["validation"] = None
                event["evaluation"] = None
                event["gate"] = None
                event["commit"] = None
                event_log.write(event)
                continue

            event["edit"] = {"reasoning": edit.reasoning}

            # 8c. Validate candidate via temp files (same pattern as llm-test command)
            ok, error_code, detail = _validate_candidate(
                strategy_name, edit, strategies_dir, configs_dir
            )
            if not ok:
                event["validation"] = {
                    "passed": False,
                    "error_code": error_code,
                    "detail": detail,
                }
                event["evaluation"] = None
                event["gate"] = None
                event["commit"] = None
                event_log.write(event)
                continue

            event["validation"] = {"passed": True}

            # 8d. Apply to real files and 8e. Evaluate — both inside the same
            # try/except so a write error also triggers a clean rollback.
            try:
                strat_path.write_text(edit.strategy_code, encoding="utf-8")
                cfg_path.write_text(edit.config_yaml, encoding="utf-8")
                new_config_obj = StrategyConfig.from_yaml(cfg_path)
                new_config = new_config_obj.model_dump()
                candidate_fn = _load_signals(strat_path)
                report_k, returns_k = evaluate_strategy_detailed(
                    strategy_name,
                    candidate_fn,
                    new_config,
                    start_date=start_date,
                    end_date=end_date,
                )
            except Exception as e:
                # Rollback and skip
                git_ledger.rollback_strategy(strategy_name)
                event["evaluation"] = {"error": str(e)}
                event["gate"] = None
                event["commit"] = None
                event_log.write(event)
                continue

            # 8f. DSR deflation
            _deflate(report_k, returns_k, ledger)

            # 8g. Gate against incumbent
            gate_res = accept(
                report_k,
                baseline=incumbent,
                target_metric=target_metric,
                config=new_config,
            )

            event["evaluation"] = {
                "observed_sharpe": report_k.observed_sharpe,
                "deflated_sharpe": report_k.deflated_sharpe,
                "effective_trials": report_k.effective_trials,
                "holdout_max_drawdown": report_k.holdout_metrics.max_drawdown,
                "holdout_turnover": report_k.holdout_metrics.turnover,
                "regime_passed": report_k.regime_passed,
            }

            # 8h. Commit or rollback
            if gate_res.accepted:
                sha = git_ledger.commit_strategy(
                    strategy_name,
                    f"iter {k}: {edit.reasoning[:72]}",
                )
                incumbent = report_k
                incumbent_returns = returns_k
                n_committed += 1
                ledger.record_attempt(
                    run_id=run_id,
                    iteration=k,
                    strategy_name=strategy_name,
                    dataset_hash=dataset_hash,
                    config_yaml=edit.config_yaml,
                    observed_sharpe=report_k.observed_sharpe,
                    deflated_sharpe=report_k.deflated_sharpe,
                    target_metric=target_metric.value,
                    target_metric_value=_get_metric_value(report_k, target_metric),
                    holdout_max_drawdown=report_k.holdout_metrics.max_drawdown,
                    holdout_turnover=report_k.holdout_metrics.turnover,
                    regime_passed=report_k.regime_passed,
                    accepted=True,
                    committed=True,
                    commit_sha=sha,
                    rejection_reason=None,
                    report_json=report_k.to_json(),
                    holdout_returns=returns_k,
                )
                event["gate"] = {"accepted": True, "reason": None}
                event["commit"] = {"sha": sha}
            else:
                git_ledger.rollback_strategy(strategy_name)
                ledger.record_attempt(
                    run_id=run_id,
                    iteration=k,
                    strategy_name=strategy_name,
                    dataset_hash=dataset_hash,
                    config_yaml=edit.config_yaml,
                    observed_sharpe=report_k.observed_sharpe,
                    deflated_sharpe=report_k.deflated_sharpe,
                    target_metric=target_metric.value,
                    target_metric_value=_get_metric_value(report_k, target_metric),
                    holdout_max_drawdown=report_k.holdout_metrics.max_drawdown,
                    holdout_turnover=report_k.holdout_metrics.turnover,
                    regime_passed=report_k.regime_passed,
                    accepted=False,
                    committed=False,
                    commit_sha=None,
                    rejection_reason=gate_res.reason,
                    report_json=report_k.to_json(),
                    holdout_returns=returns_k,
                )
                event["gate"] = {"accepted": False, "reason": gate_res.reason}
                event["commit"] = None

            event_log.write(event)

    finally:
        # Refresh final report's DSR using complete session history so the
        # returned incumbent reflects the true multiple-testing penalty.
        try:
            hist_matrix, hist_sharpes = ledger.fetch_historical_returns(
                incumbent.dataset_hash
            )
            if not hist_matrix.empty and len(hist_sharpes) > 1:
                n = max(1, calculate_effective_trials(hist_matrix))
                incumbent.effective_trials = n
                incumbent.deflated_sharpe = calculate_psr_dsr(
                    incumbent_returns, hist_sharpes, n
                )
        except Exception:
            pass  # best-effort; do not mask the loop exception
        # 9. Cleanup
        event_log.close()
        ledger.close()

    return OrchestratorResult(
        run_id=run_id,
        branch=branch,
        n_committed=n_committed,
        final_report=incumbent,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_dataset_hash(config: dict[str, Any]) -> str:
    """Compute stable dataset hash from sorted universe tickers."""
    tickers = sorted(config.get("universe", []))
    return hashlib.sha256(",".join(tickers).encode()).hexdigest()[:16]


def _load_signals(path: Path) -> Any:
    """Dynamically import generate_signals from a strategy .py file."""
    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load strategy module from {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        # Only evict if the module registered itself — avoids clobbering a
        # legitimate stdlib/third-party module that shares the same name.
        if sys.modules.get(module_name) is module:
            sys.modules.pop(module_name)
    if not hasattr(module, "generate_signals"):
        raise AttributeError(f"Strategy module {path} has no generate_signals function")
    return module.generate_signals


def _validate_candidate(
    strategy_name: str,
    edit: AgentEdit,
    strategies_dir: Path,
    configs_dir: Path,
) -> tuple[bool, str | None, str | None]:
    """Validate candidate edit via temp files (mirrors cli.py llm-test pattern)."""
    temp_name = f"{strategy_name}_candidate_{uuid.uuid4().hex}"
    temp_py = strategies_dir / f"{temp_name}.py"
    temp_yaml = configs_dir / f"{temp_name}.yaml"
    try:
        temp_py.write_text(edit.strategy_code, encoding="utf-8")
        temp_yaml.write_text(edit.config_yaml, encoding="utf-8")
        result = preflight(temp_name, strategies_dir, configs_dir)
        return (
            result.passed,
            str(result.error_code) if result.error_code else None,
            str(result.detail) if result.detail else None,
        )
    finally:
        if temp_py.exists():
            temp_py.unlink()
        if temp_yaml.exists():
            temp_yaml.unlink()


def _deflate(
    report: EvaluationReport,
    holdout_returns: pd.Series[Any],
    ledger: LedgerStore,
) -> None:
    """Overwrite report's deflated_sharpe and effective_trials using ledger history."""
    hist_matrix, hist_sharpes = ledger.fetch_historical_returns(report.dataset_hash)

    if hist_matrix.empty:
        # No prior history; current is the only trial → PSR (N=1) — report unchanged
        return

    # Build matrix including current trial
    current_col = holdout_returns.rename("current")
    matrix = pd.concat([hist_matrix, current_col], axis=1).dropna(how="all")

    n = max(1, calculate_effective_trials(matrix))
    sharpes = [*hist_sharpes, report.observed_sharpe]

    report.effective_trials = n
    report.deflated_sharpe = calculate_psr_dsr(holdout_returns, sharpes, n)


def _get_metric_value(report: EvaluationReport, metric: TargetMetric) -> float:
    """Extract the target metric value from a report."""
    if metric == TargetMetric.SHARPE:
        return report.holdout_metrics.sharpe_ratio
    elif metric == TargetMetric.SORTINO:
        return report.holdout_metrics.sortino_ratio
    else:  # INFORMATION_RATIO
        return report.holdout_metrics.information_ratio
