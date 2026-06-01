"""Top-level orchestration loop.

Coordinates the iterative LLM-driven strategy optimization pipeline:
generates parallel candidate edits, validates via preflight checks, enforces
config/returns diversity gates, evaluates walk-forward + holdout performance,
applies the two-phase select/confirm gate system, and commits winners to git.
Tracks explore/exploit mode with dynamic temperature tuning and performs
parameter importance analysis after each iteration.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import logging
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
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
from autobacktest.lessons import LessonStore
from autobacktest.llm.base import AgentContext, AgentEdit, LLMError, LLMProvider
from autobacktest.program import parse_program
from autobacktest.strategy.config_schema import StrategyConfig
from autobacktest.strategy.diversity import (
    check_returns_correlation,
    max_config_similarity,
)
from autobacktest.strategy.parameter_importance import (
    compute_parameter_importance,
    format_importance_lessons,
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


class _ThreadSafeDict:
    """Thread-safe dict wrapper using composition (not inheritance).

    Wraps a plain ``dict`` with a ``threading.Lock`` so that concurrent
    ``.get()`` / ``__setitem__`` / ``__contains__`` calls from the
    ``ThreadPoolExecutor`` workers do not race on dict resize.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[int, tuple[EvaluationReport, pd.Series]] = {}

    def __getitem__(self, key: int) -> tuple[EvaluationReport, pd.Series]:
        with self._lock:
            return self._data[key]

    def __setitem__(self, key: int, value: tuple[EvaluationReport, pd.Series]) -> None:
        with self._lock:
            self._data[key] = value

    def __contains__(self, key: int) -> bool:
        with self._lock:
            return key in self._data

    def get(
        self, key: int, default: tuple[EvaluationReport, pd.Series] | None = None
    ) -> tuple[EvaluationReport, pd.Series] | None:
        with self._lock:
            return self._data.get(key, default)


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
    lesson_store = LessonStore(run_dir / "lessons.db")
    event_log = EventLog(run_dir / run_id / "events.jsonl")
    incumbent_returns = pd.Series(dtype=float)  # set before try so finally can read it
    incumbent: EvaluationReport  # assigned during baseline below
    try:
        # 5. Load baseline config
        start_temp = getattr(provider, "temperature", None)
        min_temp = 0.1
        config_obj = StrategyConfig.from_yaml(cfg_path)
        config = config_obj.model_dump()
        _eval_cache = _ThreadSafeDict()

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

        # 6b. Initialize lessons — migrate any existing lessons.md, then use the DB
        lessons_md_path = git_ledger.repo_root / "lessons.md"
        if lessons_md_path.exists():
            n = lesson_store.migrate_from_file(lessons_md_path, strategy_name)
            if n > 0:
                logger.info("Migrated %d lessons from lessons.md to lessons.db", n)
        lessons_text = lesson_store.get_filtered_markdown(strategy_name)

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
                _eval_cache=_eval_cache,  # type: ignore[arg-type]
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
                    "SELECT returns_blob, holdout_returns_blob FROM attempts WHERE run_id = ? AND iteration = ?",
                    (run_id, latest_accepted[0]),
                ).fetchone()
                incumbent_returns = _deserialize_returns(bytes(ret_row[0])) if ret_row else pd.Series(dtype=float)
                if ret_row and ret_row[1] is not None:
                    incumbent.holdout_net_returns = _deserialize_returns(bytes(ret_row[1]))
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
        baseline_at_start = incumbent
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost = 0.0
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
                    # Snapshot pre-iteration state for progress display
                    _iter_prev_cost = total_cost
                    _iter_incumbent_sharpe = incumbent.observed_sharpe

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

                    # 8b. Generate N candidates in parallel
                    n = getattr(settings, "n_candidates", 3)
                    raw_edits = _generate_candidates(provider, ctx, n)
                    n_gen = sum(1 for e in raw_edits if e is not None)
                    progress.update(
                        task,
                        description=(
                            f"[cyan]Iter {k}/{iterations} | mode={mode} | "
                            f"LLM generated {n_gen} candidates, validating..."
                        ),
                    )

                    # Filter out None (LLM failures) and process each candidate
                    candidate_results: list[dict[str, Any]] = []
                    for edit in raw_edits:
                        ev: dict[str, Any] = {"edit": edit}
                        if edit is None:
                            ev["llm_error"] = True
                        else:
                            n_llm_ok += 1
                            total_prompt_tokens += edit.prompt_tokens
                            total_completion_tokens += edit.completion_tokens
                            total_cost += edit.cost

                            # Immediately persist lessons from the edit
                            if edit.lessons_text is not None and edit.lessons_text.strip():
                                lesson_store.ingest_markdown(edit.lessons_text, strategy_name)
                                lessons_text = lesson_store.get_filtered_markdown(strategy_name)

                            # Validate
                            ok, err_code, err_detail = _validate_candidate(
                                strategy_name, edit, strategies_dir, configs_dir
                            )
                            if ok:
                                ev["valid"] = True
                                ev["strategy_code"] = edit.strategy_code
                                ev["config_yaml"] = edit.config_yaml
                                ev["prompt_tokens"] = edit.prompt_tokens
                                ev["completion_tokens"] = edit.completion_tokens
                                ev["total_tokens"] = edit.total_tokens
                                ev["cost"] = edit.cost
                            else:
                                ev["valid"] = False
                                ev["validation_stage"] = "validation"
                                ev["detail"] = err_detail
                                ev["error_code"] = err_code
                                ev["strategy_code"] = edit.strategy_code
                                ev["config_yaml"] = edit.config_yaml

                        candidate_results.append(ev)

                    # Config diversity gate (pre-backtest, main thread)
                    valid_candidates = []
                    for ev in candidate_results:
                        if not ev.get("valid"):
                            valid_candidates.append(ev)
                            continue
                        e_config_yaml = ev["config_yaml"]
                        if mode == "explore" and historical_configs:
                            max_sim = max_config_similarity(e_config_yaml, historical_configs)
                            if max_sim > DIVERSITY_CONFIG_THRESHOLD:
                                ev["valid"] = False
                                ev["validation_stage"] = "diversity_config"
                                ev["detail"] = (
                                    f"Config similarity {max_sim:.3f} exceeded threshold {DIVERSITY_CONFIG_THRESHOLD}."
                                )
                        valid_candidates.append(ev)

                    # Collect valid candidates for backtest evaluation
                    to_eval = [ev for ev in valid_candidates if ev.get("valid")]
                    _iter_n_val = len(to_eval)
                    progress.update(
                        task, description=(f"[cyan]Iter {k}/{iterations} | Backtesting {_iter_n_val} candidates...")
                    )

                    # Phase: evaluate all valid candidates in parallel on temp files
                    if to_eval:
                        # Ensure single-threaded eval cache creation
                        def _eval_one(ev: dict[str, Any]) -> dict[str, Any]:
                            r, ret, cfg, err = _eval_single_candidate(
                                strategy_name,
                                ev["strategy_code"],
                                ev["config_yaml"],
                                strategies_dir,
                                configs_dir,
                                start_date,
                                end_date,
                                _eval_cache,  # type: ignore[arg-type]
                            )
                            if err:
                                ev["valid"] = False
                                ev["validation_stage"] = "eval_error"
                                ev["detail"] = err
                            else:
                                ev["_report"] = r
                                ev["_returns"] = ret
                                ev["_new_config"] = cfg
                            return ev

                        with ThreadPoolExecutor(max_workers=min(4, len(to_eval))) as pool:
                            futures = {pool.submit(_eval_one, ev): i for i, ev in enumerate(to_eval)}
                            for future in as_completed(futures):
                                future.result()
                    if to_eval:
                        progress.update(task, description=(f"[cyan]Iter {k}/{iterations} | Running gates..."))

                    # Gate phase + winner selection (main thread, sequential)
                    winner: dict[str, Any] | None = None
                    sha: str | None = None
                    best_metric: float = -float("inf")

                    for ev in candidate_results:
                        if not ev.get("valid") or ev.get("_report") is None:
                            continue

                        report_k = ev["_report"]
                        returns_k = ev["_returns"]
                        new_config = ev["_new_config"]

                        # Returns diversity gate (post-backtest)
                        if mode == "explore":
                            hist_matrix, _ = ledger.fetch_historical_returns(dataset_hash)
                            if not hist_matrix.empty:
                                corr_passed, max_corr = check_returns_correlation(
                                    returns_k, hist_matrix, DIVERSITY_RETURNS_THRESHOLD
                                )
                                if not corr_passed:
                                    ev["valid"] = False
                                    ev["validation_stage"] = "diversity_returns"
                                    ev["detail"] = (
                                        f"Return correlation {max_corr:.3f} exceeded threshold "
                                        f"{DIVERSITY_RETURNS_THRESHOLD}."
                                    )
                                    ev["_report_json"] = report_k.to_json()
                                    ev["_observed_sharpe"] = report_k.observed_sharpe
                                    continue

                        # DSR deflation (in-sample)
                        _deflate(report_k, returns_k, ledger)
                        if incumbent is not None and not incumbent_returns.empty:
                            _deflate(incumbent, incumbent_returns, ledger)

                        # Selection gate
                        sel = select(report_k, baseline=incumbent, target_metric=target_metric, config=new_config)
                        ev["_sel"] = sel

                        if sel.accepted:
                            # Holdout peek budget check
                            hist_matrix, _ = ledger.fetch_holdout_history(report_k.dataset_hash)
                            current_peeks = len(hist_matrix.columns) if not hist_matrix.empty else 0
                            if current_peeks >= holdout_peek_limit:
                                logger.warning(
                                    f"Holdout peek limit reached: {current_peeks} >= {holdout_peek_limit}. "
                                    "Aborting optimization loop immediately."
                                )
                                _early_stop = True
                                ev["valid"] = False
                                ev["validation_stage"] = "holdout_peek_limit"
                                ev["_peek_fail"] = True
                                continue

                            _deflate_holdout(report_k, ledger)
                            if incumbent is not None:
                                _deflate_holdout(incumbent, ledger)

                            cnf = confirm(report_k, baseline=incumbent, config=new_config)
                            ev["_cnf"] = cnf

                            if cnf.accepted:
                                metric_val = _get_metric_value(report_k, target_metric)
                                if metric_val > best_metric:
                                    best_metric = metric_val
                                    winner = ev
                                ev["_accepted"] = True
                            else:
                                ev["valid"] = False
                                ev["validation_stage"] = "gate"
                                ev["detail"] = cnf.reason
                                ev["_failed_gate"] = cnf.failed_gate
                        else:
                            ev["valid"] = False
                            ev["validation_stage"] = "gate"
                            ev["detail"] = sel.reason
                            ev["_failed_gate"] = sel.failed_gate

                    # --- Commit winner (if any) ---
                    if winner is not None:
                        w_edit = winner["edit"]
                        w_report = winner["_report"]
                        w_returns = winner["_returns"]
                        strat_path.write_text(w_edit.strategy_code, encoding="utf-8")
                        cfg_path.write_text(w_edit.config_yaml, encoding="utf-8")
                        sha = git_ledger.commit_strategy(
                            strategy_name,
                            f"iter {k}: {w_edit.reasoning[:72] if w_edit.reasoning else 'multi-candidate'}",
                        )
                        incumbent = w_report
                        incumbent_returns = w_returns
                        n_committed += 1
                        last_attempt = None
                        consecutive_no_accept = 0
                        mode = "exploit"
                        exploit_stall = 0

                    # --- Record ALL candidates in ledger ---
                    rejection_reason_map = {
                        "diversity_config": "diversity_tier1_config",
                        "diversity_returns": "diversity_tier2_returns",
                        "validation": None,
                        "eval_error": None,
                        "gate": None,
                    }

                    def _rejection_reason(
                        ev: dict[str, Any],
                        _map: dict[str, str | None] = rejection_reason_map,
                    ) -> str | None:
                        stage = ev.get("validation_stage")
                        if stage in _map:
                            return _map[stage] or ev.get("detail")
                        if stage == "holdout_peek_limit":
                            return "holdout_peek_limit_exceeded"
                        if stage == "gate":
                            return ev.get("detail") or ev.get("_failed_gate")
                        return ev.get("detail") or stage

                    def _record(
                        ev: dict[str, Any],
                        accepted: bool,
                        committed: bool,
                        _k: int = k,
                        _sha: str | None = sha,
                    ) -> None:
                        rp = ev.get("_report")
                        rt = ev.get("_returns")
                        edit = ev.get("edit")
                        if rp is None or edit is None:
                            return
                        ho_evaluated = ev.get("_cnf") is not None
                        ledger.record_attempt(
                            run_id=run_id,
                            iteration=_k,
                            strategy_name=strategy_name,
                            dataset_hash=dataset_hash,
                            config_yaml=ev.get("config_yaml", ""),
                            observed_sharpe=rp.observed_sharpe,
                            deflated_sharpe=rp.deflated_sharpe,
                            target_metric=target_metric.value,
                            target_metric_value=_get_metric_value(rp, target_metric),
                            in_sample_max_drawdown=rp.in_sample_metrics.max_drawdown,
                            in_sample_turnover=rp.in_sample_metrics.turnover,
                            regime_passed=rp.regime_passed,
                            accepted=accepted,
                            committed=committed,
                            commit_sha=_sha if committed else None,
                            rejection_reason=None if accepted else _rejection_reason(ev),
                            report_json=rp.to_json(),
                            selection_returns=rt,
                            prompt_tokens=edit.prompt_tokens,
                            completion_tokens=edit.completion_tokens,
                            total_tokens=edit.total_tokens,
                            cost=edit.cost,
                            holdout_evaluated=ho_evaluated,
                            holdout_observed_sharpe=rp.holdout_metrics.sharpe_ratio if ho_evaluated else None,
                            holdout_returns=rp.holdout_net_returns if ho_evaluated else None,
                        )

                    for ev in candidate_results:
                        if ev.get("_report") is not None:
                            is_winner = ev is winner
                            _record(ev, accepted=is_winner, committed=is_winner)
                        elif ev.get("llm_error"):
                            pass  # No report to record for LLM failures

                    # --- Compute parameter importance ---
                    try:
                        imp_configs, imp_metrics = ledger.fetch_param_importance_data(dataset_hash)
                        importance = compute_parameter_importance(
                            imp_configs,
                            imp_metrics,
                            min_attempts=settings.importance_min_attempts,
                            p_threshold=settings.importance_p_threshold,
                        )
                        if importance:
                            imp_text = format_importance_lessons(importance)
                            if imp_text:
                                lesson_store.ingest_markdown(imp_text, strategy_name)
                                lessons_text = lesson_store.get_filtered_markdown(strategy_name)
                            event["parameter_importance"] = importance
                    except Exception:
                        logger.warning("Parameter importance computation failed", exc_info=True)

                    # --- Build consolidated event ---
                    candidates_summary = []
                    for ev in candidate_results:
                        if ev.get("llm_error"):
                            candidates_summary.append({"llm_error": True})
                        elif not ev.get("valid"):
                            fail_item: dict[str, Any] = {
                                "passed": False,
                                "stage": ev.get("validation_stage"),
                                "detail": ev.get("detail"),
                            }
                            candidates_summary.append(fail_item)
                        else:
                            rp = ev.get("_report")
                            pass_item: dict[str, Any] = {
                                "passed": True,
                                "accepted": ev is winner,
                                "observed_sharpe": rp.observed_sharpe if rp else None,
                            }
                            candidates_summary.append(pass_item)
                    event["candidates"] = candidates_summary
                    if winner is not None:
                        winner_idx = next(i for i, e in enumerate(candidate_results) if e is winner)
                        event["winner"] = {"candidate_idx": winner_idx}
                        event["gate"] = {"stage": "select", "accepted": True}
                        event["commit"] = {"sha": sha}
                    else:
                        event["gate"] = {
                            "stage": "select",
                            "accepted": False,
                            "reason": "No candidate passed all gates",
                        }
                        event["commit"] = None
                        consecutive_no_accept += 1
                        if mode == "exploit":
                            exploit_stall += 1
                            if exploit_stall >= EXPLOIT_PATIENCE:
                                logger.info(f"Exploit stall reached ({EXPLOIT_PATIENCE}), switching to EXPLORE mode.")
                                mode = "explore"
                                exploit_stall = 0
                        last_attempt = {"stage": "all_candidates_failed", "detail": "No candidate passed all gates"}

                    # --- Per-iteration summary line ---
                    _iter_cost = total_cost - _iter_prev_cost
                    if winner is not None:
                        w_report = winner["_report"]
                        delta = w_report.observed_sharpe - _iter_incumbent_sharpe
                        summary = (
                            f"[green]✓[/] Iter {k:>3}/{iterations}  "
                            f"[cyan]mode={mode:<7}[/]"
                            f"{f'  t={provider.temperature:.2f}' if start_temp is not None else ''}  "
                            f"gen={n_gen}  val={_iter_n_val}  "
                            f"→  Sharpe {_iter_incumbent_sharpe:.3f}→{w_report.observed_sharpe:.3f}  "
                            f"({delta:+.3f})  "
                            f"dd={w_report.in_sample_metrics.max_drawdown * 100:.1f}%  "
                            f"to={w_report.in_sample_metrics.turnover:.2f}x  "
                            f"${_iter_cost:.4f}"
                        )
                    else:
                        reasons: list[str] = []
                        for ev in candidate_results:
                            stage = ev.get("validation_stage")
                            if stage == "gate":
                                fg = ev.get("_failed_gate")
                                reasons.append(f"gate({fg})" if fg else "gate")
                            elif stage == "validation":
                                reasons.append("preflight")
                            elif stage == "diversity_config":
                                reasons.append("config_diversity")
                            elif stage == "diversity_returns":
                                reasons.append("returns_diversity")
                            elif stage == "eval_error":
                                reasons.append("backtest_error")
                            elif stage == "holdout_peek_limit":
                                reasons.append("peek_limit")
                            elif ev.get("llm_error"):
                                reasons.append("llm_error")
                            elif stage:
                                reasons.append(stage)
                        reasons_str = ",".join(dict.fromkeys(reasons))[:80] if reasons else "all_failed"
                        summary = (
                            f"[red]✗[/] Iter {k:>3}/{iterations}  "
                            f"[cyan]mode={mode:<7}[/]"
                            f"{f'  t={provider.temperature:.2f}' if start_temp is not None else ''}  "
                            f"gen={n_gen}  val={_iter_n_val}  "
                            f"→  FAIL  {reasons_str}  "
                            f"${_iter_cost:.4f}"
                        )
                    progress.console.print(summary)

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
        lesson_store.close()
        ledger.close()

    return OrchestratorResult(
        run_id=run_id,
        branch=branch,
        n_committed=n_committed,
        final_report=incumbent,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_cost=total_cost,
        baseline_report=baseline_at_start,
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


def _generate_candidates(
    provider: LLMProvider,
    ctx: AgentContext,
    n: int,
) -> list[AgentEdit | None]:
    """Generate N candidate edits in parallel, returning None for transient failures.

    Non-retryable errors (e.g. auth failures) are raised immediately.
    """

    def _try() -> AgentEdit | None:
        try:
            return provider.generate_edit(ctx)
        except LLMError as e:
            if not e.retryable:
                raise
            return None

    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(_try) for _ in range(n)]
        return [f.result() for f in futures]


def _eval_single_candidate(
    strategy_name: str,
    strategy_code: str,
    config_yaml: str,
    strategies_dir: Path,
    configs_dir: Path,
    start_date: str,
    end_date: str,
    _eval_cache: dict[int, tuple[EvaluationReport, pd.Series]],
) -> tuple[EvaluationReport | None, pd.Series[Any] | None, dict[str, Any] | None, str | None]:
    """Evaluate one candidate via temp files.

    Returns ``(report, returns, new_config, error_str)``.  When evaluation
    fails all four values are ``None``.  Temp files are cleaned up.
    """
    temp_name = f"eval_{uuid.uuid4().hex}"
    temp_py = strategies_dir / f"{temp_name}.py"
    temp_yaml = configs_dir / f"{temp_name}.yaml"
    try:
        temp_py.write_text(strategy_code, encoding="utf-8")
        temp_yaml.write_text(config_yaml, encoding="utf-8")
        candidate_fn = _load_signals(temp_py)
        new_config_obj = StrategyConfig.from_yaml(temp_yaml)
        new_config = new_config_obj.model_dump()
        report, returns = evaluate_strategy_detailed(
            strategy_name,
            candidate_fn,
            new_config,
            start_date=start_date,
            end_date=end_date,
            _eval_cache=_eval_cache,
            _strategy_code=strategy_code,
        )
        return report, returns, new_config, None
    except Exception as e:
        return None, None, None, str(e)
    finally:
        for p in [temp_py, temp_yaml]:
            if p.exists():
                p.unlink()


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
