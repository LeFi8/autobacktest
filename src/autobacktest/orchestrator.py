"""Top-level orchestration loop."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import logging
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Any

import pandas as pd
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from autobacktest.config import settings
from autobacktest.evaluator.deflated_sharpe import (
    calculate_effective_trials,
    calculate_psr_dsr,
)
from autobacktest.evaluator.evaluate import evaluate_strategy_detailed
from autobacktest.evaluator.report import EvaluationReport
from autobacktest.gate import TargetMetric, confirm, select
from autobacktest.ledger.event_log import EventLog
from autobacktest.ledger.git_ops import GitLedger
from autobacktest.ledger.store import LedgerStore
from autobacktest.llm.base import AgentContext, AgentEdit, LLMError, LLMProvider
from autobacktest.program import parse_program
from autobacktest.strategy.config_schema import StrategyConfig
from autobacktest.strategy.diversity import (
    check_returns_correlation,
    max_config_similarity,
)
from autobacktest.strategy.validator import preflight

logger = logging.getLogger(__name__)

DIVERSITY_CONFIG_THRESHOLD = 0.95
DIVERSITY_RETURNS_THRESHOLD = 0.90
STUCK_THRESHOLD = 5
STUCK_ESCALATION_FACTOR = 0.8
MAX_DIVERSITY_RETRIES = 2
EARLY_STOP_PATIENCE = 10  # counts all non-acceptance outcomes (validation, diversity, gate)
EXPLOIT_PATIENCE = 3  # consecutive non-improvements in EXPLOIT before returning to EXPLORE


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

    Args:
        program_path: Path to the markdown program objective file.
        strategy_name: The name of the target strategy to optimize.
        iterations: Total optimization runs to execute.
        provider: Enclosing LiteLLM/Mock provider to generate candidate mutations.
        run_dir: Directory where the run database and events log are written.
        strategies_dir: Directory enclosing target strategy modules.
            Defaults to "strategies".
        configs_dir: Directory enclosing YAML parameters files.
            Defaults to "configs".
        target_metric: Metric choice to target during gate checks.
            Defaults to TargetMetric.SHARPE.
        repo_path: Root repository path for Git workspace operations.
            Defaults to root path.
        start_date: Starting date boundary for evaluation.
            Defaults to "2015-01-01".
        end_date: Ending date boundary for evaluation.
            Defaults to "2026-01-01".

    Returns:
        OrchestratorResult: Summary of the final optimization run outcomes.

    Raises:
        FileNotFoundError: If the target strategy or configuration files are missing.
    """
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
    if resume:
        run_id = resume
        branch = f"autobacktest/{run_id}"
        try:
            git_ledger._repo.heads[branch].checkout()
        except Exception as e:
            try:
                git_ledger._repo.git.checkout(branch)
            except Exception:
                raise ValueError(f"Could not checkout branch '{branch}': {e}") from e
    else:
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
        start_temp = getattr(provider, "temperature", None)
        min_temp = 0.1
        config_obj = StrategyConfig.from_yaml(cfg_path)
        config = config_obj.model_dump()
        _eval_cache: dict[int, tuple[EvaluationReport, pd.Series]] = {}

        # 6. Record run metadata
        dataset_hash = _compute_dataset_hash(config)
        if not resume:
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

        # Initialize lessons
        lessons_path = git_ledger.repo_root / "lessons.md"
        lessons_text = ""
        if lessons_path.exists():
            lessons_text = lessons_path.read_text(encoding="utf-8")

        # 7. Baseline evaluation (iteration 0 — not written to events.jsonl)
        baseline_exists = False
        if resume:
            check_baseline = ledger._conn.execute(
                "SELECT COUNT(*) FROM attempts WHERE run_id = ? AND iteration = 0", (run_id,)
            ).fetchone()
            if check_baseline and check_baseline[0] > 0:
                baseline_exists = True

        if not baseline_exists:
            baseline_fn = _load_signals(strat_path)
            _baseline_code = strat_path.read_text(encoding="utf-8")
            baseline_report, baseline_returns = evaluate_strategy_detailed(
                strategy_name,
                baseline_fn,
                config,
                start_date=start_date,
                end_date=end_date,
                _eval_cache=_eval_cache,
                _strategy_code=_baseline_code,
            )
            _deflate(baseline_report, baseline_returns, ledger)
            _deflate_holdout(baseline_report, ledger)
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
                in_sample_max_drawdown=baseline_report.in_sample_metrics.max_drawdown,
                in_sample_turnover=baseline_report.in_sample_metrics.turnover,
                regime_passed=baseline_report.regime_passed,
                accepted=True,
                committed=True,
                commit_sha=baseline_sha,
                rejection_reason=None,
                report_json=baseline_report.to_json(),
                selection_returns=baseline_returns,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost=0.0,
                holdout_evaluated=True,
                holdout_observed_sharpe=baseline_report.holdout_metrics.sharpe_ratio,
                holdout_returns=baseline_report.holdout_net_returns,
            )
        else:
            # Reconstruct incumbent from the latest accepted/committed attempt
            rows = ledger._conn.execute(
                "SELECT iteration, accepted, committed, report_json "
                "FROM attempts WHERE run_id = ? ORDER BY iteration ASC",
                (run_id,),
            ).fetchall()

            latest_accepted = None
            if rows:
                accepted_rows = [row for row in rows if row[1] or row[2]]
                if accepted_rows:
                    latest_accepted = max(accepted_rows, key=lambda row: row[0])
                else:
                    latest_accepted = next(r for r in rows if r[0] == 0)

            if latest_accepted:
                from autobacktest.evaluator.report import EvaluationReport
                from autobacktest.ledger.store import _deserialize_returns

                incumbent = EvaluationReport.from_json(latest_accepted[3])

                ret_row = ledger._conn.execute(
                    "SELECT returns_blob FROM attempts WHERE run_id = ? AND iteration = ?", (run_id, latest_accepted[0])
                ).fetchone()
                incumbent_returns = _deserialize_returns(bytes(ret_row[0])) if ret_row else pd.Series(dtype=float)
            else:
                from autobacktest.evaluator.report import WindowReport

                _zero = WindowReport(
                    start_date="",
                    end_date="",
                    annualized_return=0.0,
                    annualized_volatility=0.0,
                    sharpe_ratio=0.0,
                    sortino_ratio=0.0,
                    max_drawdown=0.0,
                    turnover=0.0,
                )
                incumbent = EvaluationReport(
                    strategy_name=strategy_name,
                    dataset_hash=dataset_hash,
                    gates_passed={},
                    is_accepted=False,
                    rejection_reason=None,
                    holdout_metrics=_zero,
                    in_sample_metrics=_zero,
                    walk_forward_metrics=[],
                    regime_drawdowns={},
                    regime_passed=False,
                    mc_sharpe_5th=0.0,
                    mc_sharpe_50th=0.0,
                    mc_sharpe_95th=0.0,
                    observed_sharpe=0.0,
                    effective_trials=1,
                    deflated_sharpe=0.0,
                )
                incumbent_returns = pd.Series(dtype=float)

        # 8. Optimization loop
        start_k = 1
        if resume:
            rows = ledger._conn.execute("SELECT iteration FROM attempts WHERE run_id = ?", (run_id,)).fetchall()
            if rows:
                start_k = max(row[0] for row in rows) + 1

        n_committed = 0
        n_llm_ok = 0
        last_attempt: dict[str, Any] | None = None
        consecutive_no_accept: int = 0
        rolling_history: list[bool] = []
        _early_stop = False
        mode: str = "explore"
        exploit_stall: int = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
        ) as progress:
            task = progress.add_task(
                f"[cyan]Optimizing {strategy_name}... (Incumbent Sharpe: {incumbent.observed_sharpe:.3f})",
                total=iterations,
            )
            for k in range(start_k, iterations + 1):
                prev_n_committed = n_committed
                # Stuck detection: force EXPLORE and reset exploit_stall before temperature block
                if consecutive_no_accept >= STUCK_THRESHOLD:
                    if mode != "explore":
                        logger.info("Stuck threshold reached, forcing EXPLORE mode.")
                    mode = "explore"
                    exploit_stall = 0

                if start_temp is not None:
                    if mode == "exploit":
                        # In exploit mode: always use low temperature for focused refinement
                        provider.temperature = min_temp
                    else:
                        # Explore mode: scale temperature based on rolling failure history
                        if rolling_history:
                            failures = rolling_history.count(False)
                            failure_rate = failures / len(rolling_history)
                        else:
                            # Start with a neutral-to-high exploration temperature in explore mode
                            failure_rate = 0.6
                        provider.temperature = min_temp + (start_temp - min_temp) * failure_rate

                event: dict[str, object] = {
                    "iteration": k,
                    "strategy": strategy_name,
                }
                event["mode"] = mode
                if start_temp is not None:
                    event["temperature"] = provider.temperature

                try:
                    # 8a. Build context
                    current_code = strat_path.read_text(encoding="utf-8")
                    current_yaml = cfg_path.read_text(encoding="utf-8")
                    historical_configs = ledger.fetch_configs(dataset_hash)
                    attempt_summaries = ledger.fetch_attempt_summaries(dataset_hash)
                    ctx = AgentContext(
                        strategy_name=strategy_name,
                        strategy_code=current_code,
                        config_yaml=current_yaml,
                        program_text=spec.raw_text,
                        evaluation_report=incumbent,
                        iteration=k,
                        lessons_text=lessons_text,
                        n_historical_configs=len(historical_configs),
                        last_attempt=last_attempt,
                        attempt_history=attempt_summaries,
                        mode=mode,
                    )

                    # 8b. Get LLM edit (with auto-retry on length limit cutoff and transient error backoff)
                    retry_count = 0
                    max_retries = 1
                    transient_retry_count = 0
                    max_transient_retries = 2
                    orig_max_tokens = getattr(provider, "max_tokens", None)
                    edit = None
                    while True:
                        try:
                            edit = provider.generate_edit(ctx)
                            n_llm_ok += 1
                            break
                        except LLMError as e:
                            # 1. Length cutoff retry
                            if getattr(e, "finish_reason", None) == "length" and retry_count < max_retries:
                                retry_count += 1
                                if hasattr(provider, "max_tokens") and provider.max_tokens is not None:
                                    old_tokens = provider.max_tokens
                                    new_tokens = int(old_tokens * 1.5)
                                    provider.max_tokens = new_tokens
                                    logger.warning(
                                        f"LLM cut off on length limit in iteration {k}. "
                                        f"Retrying with max_tokens scaled from {old_tokens} to {new_tokens}."
                                    )
                                    continue

                            # 2. Bounded backoff retry for transient retryable errors
                            if e.retryable and transient_retry_count < max_transient_retries:
                                transient_retry_count += 1
                                backoff_sec = 2.0**transient_retry_count
                                logger.warning(
                                    f"LLM transient error in iteration {k}: {e.detail}. "
                                    f"Retrying in {backoff_sec}s "
                                    f"(retry {transient_retry_count}/{max_transient_retries})..."
                                )
                                sleep(backoff_sec)
                                continue

                            # 3. Exhausted retries or non-retryable error
                            logger.warning(f"LLMError: {e.detail}")
                            event["llm_error"] = str(e)
                            event["validation"] = None
                            event["evaluation"] = None
                            event["gate"] = None
                            event["commit"] = None
                            event_log.write(event)
                            if not e.retryable:
                                from rich.console import Console

                                console = Console(stderr=True)
                                console.print(f"[bold red]LLM is misconfigured (non-retryable error):[/] {e.detail}")
                                raise e
                            edit = None
                            break

                    if orig_max_tokens is not None and hasattr(provider, "max_tokens"):
                        provider.max_tokens = orig_max_tokens

                    if edit is None:
                        consecutive_no_accept += 1
                        if mode == "exploit":
                            exploit_stall += 1
                            if exploit_stall >= EXPLOIT_PATIENCE:
                                logger.info(f"Exploit stall reached ({EXPLOIT_PATIENCE}), switching to EXPLORE mode.")
                                mode = "explore"
                                exploit_stall = 0
                        continue

                    event["edit"] = {
                        "reasoning": edit.reasoning,
                        "prompt_tokens": edit.prompt_tokens,
                        "completion_tokens": edit.completion_tokens,
                        "total_tokens": edit.total_tokens,
                        "cost": edit.cost,
                    }

                    # Immediately persist non-empty lessons_text from the edit to disk
                    # and memory.
                    if edit.lessons_text is not None and edit.lessons_text.strip():
                        lessons_text = edit.lessons_text
                        lessons_path.write_text(lessons_text, encoding="utf-8")

                    # 8c. Validate candidate via temp files (same pattern as llm-test command)
                    ok, error_code, detail = _validate_candidate(strategy_name, edit, strategies_dir, configs_dir)
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
                        last_attempt = {
                            "stage": "validation",
                            "error_code": error_code,
                            "detail": detail,
                            "candidate_strategy_code": edit.strategy_code,
                            "candidate_config_yaml": edit.config_yaml,
                        }
                        consecutive_no_accept += 1
                        if mode == "exploit":
                            exploit_stall += 1
                            if exploit_stall >= EXPLOIT_PATIENCE:
                                logger.info(f"Exploit stall reached ({EXPLOIT_PATIENCE}), switching to EXPLORE mode.")
                                mode = "explore"
                                exploit_stall = 0
                        continue

                    event["validation"] = {"passed": True}

                    # 8c.5 Tier 1 — Config diversity gate (pre-backtest)
                    _diversity_exhausted = False
                    if mode == "explore" and historical_configs:
                        max_sim = max_config_similarity(edit.config_yaml, historical_configs)
                        if max_sim > DIVERSITY_CONFIG_THRESHOLD:
                            # Diversity rejected: consume this iteration immediately with no retries
                            last_attempt = {
                                "stage": "diversity_config",
                                "detail": (
                                    f"Config similarity {max_sim:.3f} exceeded threshold "
                                    f"{DIVERSITY_CONFIG_THRESHOLD}. Your config was too similar "
                                    f"to a past attempt."
                                ),
                                "candidate_config_yaml": edit.config_yaml,
                            }
                            event["diversity"] = {
                                "tier": "config",
                                "passed": False,
                                "max_similarity": max_sim,
                                "retries_exhausted": True,
                            }
                            event["evaluation"] = None
                            event["gate"] = None
                            event["commit"] = None
                            event_log.write(event)
                            consecutive_no_accept += 1
                            _diversity_exhausted = True

                    if _diversity_exhausted:
                        continue

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
                            _eval_cache=_eval_cache,
                            _strategy_code=edit.strategy_code,
                        )
                    except Exception as e:
                        # Rollback and skip
                        git_ledger.rollback_strategy(strategy_name)
                        event["evaluation"] = {"error": str(e)}
                        event["gate"] = None
                        event["commit"] = None
                        event_log.write(event)
                        last_attempt = {
                            "stage": "eval_error",
                            "detail": str(e),
                            "candidate_strategy_code": edit.strategy_code,
                            "candidate_config_yaml": edit.config_yaml,
                        }
                        consecutive_no_accept += 1
                        if mode == "exploit":
                            exploit_stall += 1
                            if exploit_stall >= EXPLOIT_PATIENCE:
                                logger.info(f"Exploit stall reached ({EXPLOIT_PATIENCE}), switching to EXPLORE mode.")
                                mode = "explore"
                                exploit_stall = 0
                        continue

                    # 8e.5 Tier 2 — Returns correlation diversity gate (post-backtest)
                    if mode == "explore":
                        hist_matrix, _ = ledger.fetch_historical_returns(dataset_hash)
                        if not hist_matrix.empty:
                            corr_passed, max_corr = check_returns_correlation(
                                returns_k, hist_matrix, DIVERSITY_RETURNS_THRESHOLD
                            )
                            if not corr_passed:
                                git_ledger.rollback_strategy(strategy_name)
                                event["diversity"] = {
                                    "tier": "returns",
                                    "passed": False,
                                    "max_correlation": max_corr,
                                }
                                event["evaluation"] = None
                                event["gate"] = None
                                event["commit"] = None
                                event_log.write(event)
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
                                    in_sample_max_drawdown=report_k.in_sample_metrics.max_drawdown,
                                    in_sample_turnover=report_k.in_sample_metrics.turnover,
                                    regime_passed=report_k.regime_passed,
                                    accepted=False,
                                    committed=False,
                                    commit_sha=None,
                                    rejection_reason="diversity_tier2_returns",
                                    report_json=report_k.to_json(),
                                    selection_returns=returns_k,
                                    prompt_tokens=edit.prompt_tokens,
                                    completion_tokens=edit.completion_tokens,
                                    total_tokens=edit.total_tokens,
                                    cost=edit.cost,
                                )
                                last_attempt = {
                                    "stage": "diversity_returns",
                                    "detail": (
                                        f"Return correlation {max_corr:.3f} exceeded threshold "
                                        f"{DIVERSITY_RETURNS_THRESHOLD}. The strategy produces "
                                        f"returns too similar to past attempts."
                                    ),
                                    "candidate_config_yaml": edit.config_yaml,
                                    "candidate_metrics": {
                                        "observed_sharpe": report_k.observed_sharpe,
                                        "in_sample_max_drawdown": report_k.in_sample_metrics.max_drawdown,
                                    },
                                }
                                consecutive_no_accept += 1
                                continue

                    # 8f. DSR deflation (in-sample selection basis)
                    _deflate(report_k, returns_k, ledger)
                    if incumbent is not None and not incumbent_returns.empty:
                        _deflate(incumbent, incumbent_returns, ledger)

                    # 8g1. Selection gate (in-sample walk-forward aggregate)
                    sel = select(
                        report_k,
                        baseline=incumbent,
                        target_metric=target_metric,
                        config=new_config,
                    )

                    event["evaluation"] = {
                        "stage": "selection",
                        "observed_sharpe": report_k.observed_sharpe,
                        "deflated_sharpe": report_k.deflated_sharpe,
                        "effective_trials": report_k.effective_trials,
                        "in_sample_max_drawdown": report_k.in_sample_metrics.max_drawdown,
                        "in_sample_turnover": report_k.in_sample_metrics.turnover,
                        "regime_passed": report_k.regime_passed,
                    }

                    # Common attempt kwargs (in-sample selection basis)
                    attempt_kwargs: dict[str, object] = {
                        "run_id": run_id,
                        "iteration": k,
                        "strategy_name": strategy_name,
                        "dataset_hash": dataset_hash,
                        "config_yaml": edit.config_yaml,
                        "observed_sharpe": report_k.observed_sharpe,
                        "deflated_sharpe": report_k.deflated_sharpe,
                        "target_metric": target_metric.value,
                        "target_metric_value": _get_metric_value(report_k, target_metric),
                        "in_sample_max_drawdown": report_k.in_sample_metrics.max_drawdown,
                        "in_sample_turnover": report_k.in_sample_metrics.turnover,
                        "regime_passed": report_k.regime_passed,
                        "report_json": report_k.to_json(),
                        "selection_returns": returns_k,
                        "prompt_tokens": edit.prompt_tokens,
                        "completion_tokens": edit.completion_tokens,
                        "total_tokens": edit.total_tokens,
                        "cost": edit.cost,
                    }

                    if sel.accepted:
                        # Check holdout peek budget cap
                        hist_matrix, _ = ledger.fetch_holdout_history(report_k.dataset_hash)
                        current_peeks = len(hist_matrix.columns) if not hist_matrix.empty else 0
                        if current_peeks >= holdout_peek_limit:
                            logger.warning(
                                f"Holdout peek limit reached: {current_peeks} >= {holdout_peek_limit}. "
                                "Aborting optimization loop immediately to prevent further out-of-sample data leakage."
                            )
                            git_ledger.rollback_strategy(strategy_name)
                            ledger.record_attempt(
                                **attempt_kwargs,  # type: ignore[arg-type]
                                accepted=False,
                                committed=False,
                                commit_sha=None,
                                rejection_reason=(
                                    f"holdout_peek_limit_exceeded ({current_peeks} >= {holdout_peek_limit})"
                                ),
                                holdout_evaluated=False,
                                holdout_observed_sharpe=None,
                                holdout_returns=None,
                            )
                            event["gate"] = {
                                "stage": "select",
                                "accepted": False,
                                "reason": f"Holdout peek limit exceeded ({current_peeks} >= {holdout_peek_limit})",
                            }
                            event["commit"] = None
                            event_log.write(event)
                            consecutive_no_accept += 1
                            _early_stop = True
                            continue

                        # 8g2. Holdout DSR deflation (only when select passes)
                        _deflate_holdout(report_k, ledger)
                        if incumbent is not None:
                            _deflate_holdout(incumbent, ledger)

                        # 8g3. Confirmation gate (holdout — budgeted peek)
                        cnf = confirm(
                            report_k,
                            baseline=incumbent,
                            config=new_config,
                        )

                        if cnf.accepted:
                            # --- Commit (both gates pass) ---
                            sha = git_ledger.commit_strategy(
                                strategy_name,
                                f"iter {k}: {edit.reasoning[:72]}",
                            )
                            incumbent = report_k
                            incumbent_returns = returns_k
                            n_committed += 1
                            ledger.record_attempt(
                                **attempt_kwargs,  # type: ignore[arg-type]
                                accepted=True,
                                committed=True,
                                commit_sha=sha,
                                rejection_reason=None,
                                holdout_evaluated=True,
                                holdout_observed_sharpe=report_k.holdout_metrics.sharpe_ratio,
                                holdout_returns=report_k.holdout_net_returns,
                            )
                            event["gate"] = {"stage": "select", "accepted": True}
                            event["commit"] = {"sha": sha}
                            last_attempt = None
                            consecutive_no_accept = 0
                            mode = "exploit"
                            exploit_stall = 0
                        else:
                            # --- Confirm rejected (holdout violation) ---
                            git_ledger.rollback_strategy(strategy_name)
                            ledger.record_attempt(
                                **attempt_kwargs,  # type: ignore[arg-type]
                                accepted=False,
                                committed=False,
                                commit_sha=None,
                                rejection_reason=cnf.reason,
                                holdout_evaluated=True,
                                holdout_observed_sharpe=report_k.holdout_metrics.sharpe_ratio,
                                holdout_returns=report_k.holdout_net_returns,
                            )
                            event["gate"] = {
                                "stage": "confirm",
                                "accepted": False,
                                "reason": cnf.reason,
                                "failed_gate": cnf.failed_gate,
                            }
                            event["commit"] = None
                            last_attempt = {
                                "stage": "gate",
                                "rejection_reason": cnf.reason,
                                "failed_gate": cnf.failed_gate,
                                "candidate_strategy_code": edit.strategy_code,
                                "candidate_config_yaml": edit.config_yaml,
                                "candidate_metrics": {
                                    "observed_sharpe": report_k.observed_sharpe,
                                    "in_sample_max_drawdown": report_k.in_sample_metrics.max_drawdown,
                                    "in_sample_turnover": report_k.in_sample_metrics.turnover,
                                    "regime_passed": report_k.regime_passed,
                                },
                            }
                            consecutive_no_accept += 1
                            if mode == "exploit":
                                exploit_stall += 1
                                if exploit_stall >= EXPLOIT_PATIENCE:
                                    logger.info(
                                        f"Exploit stall reached ({EXPLOIT_PATIENCE}), switching to EXPLORE mode."
                                    )
                                    mode = "explore"
                                    exploit_stall = 0
                    else:
                        # --- Select rejected (in-sample violation, no holdout peek) ---
                        git_ledger.rollback_strategy(strategy_name)
                        ledger.record_attempt(
                            **attempt_kwargs,  # type: ignore[arg-type]
                            accepted=False,
                            committed=False,
                            commit_sha=None,
                            rejection_reason=sel.reason,
                            holdout_evaluated=False,
                            holdout_observed_sharpe=None,
                            holdout_returns=None,
                        )
                        event["gate"] = {
                            "stage": "select",
                            "accepted": False,
                            "reason": sel.reason,
                            "failed_gate": sel.failed_gate,
                        }
                        event["commit"] = None
                        last_attempt = {
                            "stage": "gate",
                            "rejection_reason": sel.reason,
                            "failed_gate": sel.failed_gate,
                            "candidate_strategy_code": edit.strategy_code,
                            "candidate_config_yaml": edit.config_yaml,
                            "candidate_metrics": {
                                "observed_sharpe": report_k.observed_sharpe,
                                "in_sample_max_drawdown": report_k.in_sample_metrics.max_drawdown,
                                "in_sample_turnover": report_k.in_sample_metrics.turnover,
                                "regime_passed": report_k.regime_passed,
                            },
                        }
                        consecutive_no_accept += 1
                        if mode == "exploit":
                            exploit_stall += 1
                            if exploit_stall >= EXPLOIT_PATIENCE:
                                logger.info(f"Exploit stall reached ({EXPLOIT_PATIENCE}), switching to EXPLORE mode.")
                                mode = "explore"
                                exploit_stall = 0

                    event_log.write(event)
                finally:
                    # Record iteration outcome in rolling history
                    rolling_history.append(n_committed > prev_n_committed)
                    if len(rolling_history) > 5:
                        rolling_history.pop(0)

                    # Single progress.update per iteration — runs regardless of skip/error/success.
                    progress.update(
                        task,
                        advance=1,
                        description=(
                            f"[cyan]Optimizing {strategy_name}... (Incumbent Sharpe: {incumbent.observed_sharpe:.3f})"
                        ),
                    )
                    if consecutive_no_accept >= early_stop_patience:
                        logger.info(
                            f"Early stop: no acceptance in {consecutive_no_accept} consecutive iterations. "
                            f"Stopping at iteration {k}/{iterations}."
                        )
                        _early_stop = True
                if _early_stop:
                    break

        if n_llm_ok == 0:
            raise RuntimeError("Zero successful LLM calls during optimization run. All iterations failed.")

    finally:
        if start_temp is not None:
            provider.temperature = start_temp
        # Refresh final report's selection DSR using complete session history.
        try:
            hist_matrix, hist_sharpes = ledger.fetch_historical_returns(incumbent.dataset_hash)
            if not hist_matrix.empty and len(hist_sharpes) > 1:
                n = max(1, calculate_effective_trials(hist_matrix))
                incumbent.effective_trials = n
                incumbent.deflated_sharpe = calculate_psr_dsr(incumbent_returns, hist_sharpes, n)
        except Exception:
            pass  # best-effort; do not mask the loop exception
        # Also re-deflate the holdout DSR.
        with contextlib.suppress(Exception):
            _deflate_holdout(incumbent, ledger)
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
    selection_returns: pd.Series[Any],
    ledger: LedgerStore,
) -> None:
    """Deflate the in-sample selection DSR using the ledger's multi-trial history.

    This operates on **in-sample** returns (``selection_returns``) and
    ``fetch_historical_returns`` (which now returns in-sample streams after
    the holdout-separation refactor).  The holdout DSR is deflated separately
    in ``_deflate_holdout``.
    """
    hist_matrix, hist_sharpes = ledger.fetch_historical_returns(report.dataset_hash)

    if hist_matrix.empty:
        return

    current_col = selection_returns.rename("current")
    matrix = pd.concat([hist_matrix, current_col], axis=1).dropna(how="all")

    n = max(1, calculate_effective_trials(matrix))
    sharpes = [*hist_sharpes, report.observed_sharpe]

    report.effective_trials = n
    report.deflated_sharpe = calculate_psr_dsr(selection_returns, sharpes, n)


def _deflate_holdout(
    report: EvaluationReport,
    ledger: LedgerStore,
) -> None:
    """Deflate ``report.holdout_deflated_sharpe`` by the holdout-peek count.

    Each time a candidate passes the in-sample selection gate and the
    holdout is consulted, that peek is recorded in the ledger
    (``holdout_evaluated = 1``).  The effective-trial count is derived
    from the clustered correlation of *holdout* return streams, so the
    multiple-testing penalty correctly reflects how many times the holdout
    has actually been used.
    """
    hist_matrix, hist_sharpes = ledger.fetch_holdout_history(report.dataset_hash)

    if report.holdout_net_returns is None or report.holdout_net_returns.empty:
        return

    if hist_matrix.empty:
        report.holdout_deflated_sharpe = calculate_psr_dsr(
            report.holdout_net_returns,
            effective_trials=1,
        )
        return

    current_col = report.holdout_net_returns.rename("current")
    matrix = pd.concat([hist_matrix, current_col], axis=1).dropna(how="all")

    n = max(1, calculate_effective_trials(matrix))
    sharpes = [*hist_sharpes, report.holdout_metrics.sharpe_ratio]

    report.holdout_deflated_sharpe = calculate_psr_dsr(
        report.holdout_net_returns,
        sharpes,
        n,
    )


def _get_metric_value(report: EvaluationReport, metric: TargetMetric) -> float:
    """Extract the target metric value from the in-sample walk-forward aggregate."""
    if metric == TargetMetric.SHARPE:
        return report.in_sample_metrics.sharpe_ratio
    elif metric == TargetMetric.SORTINO:
        return report.in_sample_metrics.sortino_ratio
    else:  # INFORMATION_RATIO
        return report.in_sample_metrics.information_ratio
