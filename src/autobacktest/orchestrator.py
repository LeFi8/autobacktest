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
import logging
import math
import threading
import uuid
from collections import OrderedDict
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
from autobacktest.evaluator.evaluate import _CacheProtocol, compute_dataset_hash, evaluate_strategy_detailed
from autobacktest.evaluator.report import EvaluationReport
from autobacktest.gate import TargetMetric, confirm, select
from autobacktest.ledger.event_log import EventLog
from autobacktest.ledger.git_ops import GitLedger
from autobacktest.ledger.store import LedgerStore
from autobacktest.lessons import LessonStore
from autobacktest.llm.base import AgentContext, AgentEdit, LLMProvider
from autobacktest.optimization.candidate import (
    generate_candidates,
    process_and_repair_candidate,
)
from autobacktest.optimization.eval_manager import (
    load_signals,
)
from autobacktest.optimization.persistence import (
    deflate_holdout,
    deflate_selection,
)
from autobacktest.program import parse_program
from autobacktest.strategy.config_schema import StrategyConfig
from autobacktest.strategy.diversity import (
    check_returns_correlation,
    extract_config_fingerprint,
    max_config_similarity,
    summarize_explored_space,
)
from autobacktest.strategy.parameter_importance import (
    compute_parameter_importance,
    format_importance_lessons,
)
from autobacktest.strategy.validator import preflight

logger = logging.getLogger(__name__)

STUCK_THRESHOLD = 5
STUCK_ESCALATION_FACTOR = 0.8
MAX_DIVERSITY_RETRIES = 2
EXPLOIT_PATIENCE = 3  # consecutive non-improvements in EXPLOIT before returning to EXPLORE

CANDIDATE_DIRECTIVES = [
    "structurally change the signal-generation logic",
    "explore an untried parameter region far from explored values",
    "change the risk-management/leverage mechanism",
]


class _LRUCache:
    """Thread-safe LRU cache with bounded maxsize for eval results.

    Evicts the least-recently-used entry when the cache exceeds ``maxsize``.
    Maintains the same interface as the previous ``_ThreadSafeDict`` so that
    all callers (``orchestrator.py``, ``evaluate.py``) work without changes.
    """

    def __init__(self, maxsize: int = 36) -> None:
        self._lock = threading.Lock()
        self._data: OrderedDict[int, tuple[EvaluationReport, pd.Series]] = OrderedDict()
        self._maxsize = maxsize

    def __getitem__(self, key: int) -> tuple[EvaluationReport, pd.Series]:
        with self._lock:
            self._data.move_to_end(key)
            return self._data[key]

    def __setitem__(self, key: int, value: tuple[EvaluationReport, pd.Series]) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            if len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def __contains__(self, key: int) -> bool:
        with self._lock:
            return key in self._data

    def get(
        self, key: int, default: tuple[EvaluationReport, pd.Series] | None = None
    ) -> tuple[EvaluationReport, pd.Series] | None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
            return default


@dataclass
class OrchestratorResult:
    """Summary of an optimization run returned by ``run_optimization``.

    Attributes:
        run_id: Unique run identifier ``{strategy}-{YYYYMMDD}-{HHMMSS}``.
        branch: Git branch created for this run.
        n_committed: Number of successful candidate commits.
        final_report: Evaluation report of the final incumbent strategy.
        total_prompt_tokens: Aggregate prompt tokens consumed across all LLM calls.
        total_completion_tokens: Aggregate completion tokens consumed.
        total_cost: Total cost of all LLM calls in USD.
        baseline_report: Evaluation report of the pre-optimization baseline.
        early_stopped: True when the loop exited early due to consecutive
            rejections reaching ``early_stop_patience`` or the holdout-peek
            budget being exhausted.
        early_stop_iteration: The 1-indexed iteration at which early-stop
            fired, or None if the run completed all iterations.
    """

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


class _OptimizationState:
    def __init__(
        self,
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
        quiet: bool = False,
    ):
        self.program_path = program_path
        self.strategy_name = strategy_name
        self.iterations = iterations
        self.provider = provider
        self.run_dir = run_dir
        self.strategies_dir = strategies_dir
        self.configs_dir = configs_dir
        self.target_metric = target_metric
        self.repo_path = repo_path
        self.start_date = start_date
        self.end_date = end_date
        self.holdout_peek_limit = holdout_peek_limit
        self.early_stop_patience = early_stop_patience
        self.quiet = quiet

        # Loop variables and state
        self.incumbent_returns = pd.Series(dtype=float)
        self.incumbent_attempt_id: int | None = None
        self.incumbent: EvaluationReport | None = None
        self.baseline_at_start: EvaluationReport | None = None
        self.start_temp = getattr(provider, "temperature", None)
        self.min_temp = 0.1
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self._early_stop_iteration = 0
        self.last_importance: dict[str, Any] = {}
        self.lessons_text = ""
        self.consecutive_no_accept = 0
        self.consecutive_no_backtest = 0
        self.rolling_history: list[bool] = []
        self.mode = "explore"
        self.exploit_stall = 0
        self.n_committed = 0
        self.n_llm_ok = 0
        self.last_attempt: dict[str, Any] | None = None
        self.last_iteration_failures: list[dict[str, Any]] | None = None
        self._early_stop = False

        # Resources initialized in setup
        self.spec: Any = None
        self.strat_path: Any = None
        self.cfg_path: Any = None
        self.git_ledger: Any = None
        self.run_id: Any = None
        self.branch: Any = None
        self.ledger: Any = None
        self.lesson_store: Any = None
        self.event_log: Any = None
        self.config_obj: Any = None
        self.config: Any = None
        self._eval_cache: Any = None
        self.dataset_hash: Any = None

    def setup(self, resume: str | None) -> None:
        self.spec = parse_program(self.program_path)

        self.strat_path = self.strategies_dir / f"{self.strategy_name}.py"
        self.cfg_path = self.configs_dir / f"{self.strategy_name}.yaml"
        if not self.strat_path.exists():
            raise FileNotFoundError(f"Strategy file not found: {self.strat_path}")
        if not self.cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.cfg_path}")

        self.git_ledger = GitLedger(self.repo_path)
        if resume:
            self.run_id = resume
            branch = f"autobacktest/{self.run_id}"
            try:
                self.git_ledger._repo.heads[branch].checkout()
            except Exception as e:
                try:
                    self.git_ledger._repo.git.checkout(branch)
                except Exception:
                    raise ValueError(f"Could not checkout branch '{branch}': {e}") from e
            self.branch = branch
        else:
            self.git_ledger.ensure_clean(self.strategy_name)
            self.run_id = f"{self.strategy_name}-{datetime.now(tz=UTC):%Y%m%d-%H%M%S}"
            self.branch = self.git_ledger.create_run_branch(self.run_id)

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = LedgerStore(self.run_dir / "ledger.db")
        self.lesson_store = LessonStore(self.run_dir / "lessons.db")
        self.event_log = EventLog(self.run_dir / self.run_id / "events.jsonl")

        self.config_obj = StrategyConfig.from_yaml(self.cfg_path)
        self.config = self.config_obj.model_dump()
        self._eval_cache = _LRUCache(maxsize=36)

        self.dataset_hash = compute_dataset_hash(
            self.config.get("universe", []),
            start_date=self.start_date,
            end_date=self.end_date,
            holdout_years=settings.default_holdout_years,
        )
        if not resume:
            self.ledger.create_run(
                run_id=self.run_id,
                strategy_name=self.strategy_name,
                program_path=str(self.program_path),
                provider=self.provider.provider_name,
                model=getattr(self.provider, "model", "unknown"),
                branch=self.branch,
                dataset_hash=self.dataset_hash,
                iterations=self.iterations,
                started_at=datetime.now(tz=UTC).isoformat(),
            )

        self.lessons_text = self.lesson_store.get_filtered_markdown(self.strategy_name)

        self._setup_incumbent(resume)

    def _setup_baseline(self) -> None:
        baseline_fn = _load_signals(self.strat_path)
        _baseline_code = self.strat_path.read_text(encoding="utf-8")
        baseline_report, baseline_returns = evaluate_strategy_detailed(
            self.strategy_name,
            baseline_fn,
            self.config,
            start_date=self.start_date,
            end_date=self.end_date,
            _eval_cache=self._eval_cache,
            _strategy_code=_baseline_code,
        )
        _deflate(
            baseline_report,
            baseline_returns,
            self.ledger,
            cscv_blocks=self.config.get("cscv_blocks", 10),
            embargo_days=self.config.get("cscv_embargo_days", 5),
        )
        _deflate_holdout(baseline_report, self.ledger)
        self.incumbent = baseline_report
        self.incumbent_returns = baseline_returns
        baseline_sha: str | None = self.git_ledger._repo.head.commit.hexsha
        self.incumbent_attempt_id = self.ledger.record_attempt(
            run_id=self.run_id,
            iteration=0,
            strategy_name=self.strategy_name,
            dataset_hash=self.dataset_hash,
            config_yaml=self.cfg_path.read_text(encoding="utf-8"),
            observed_sharpe=baseline_report.observed_sharpe,
            deflated_sharpe=baseline_report.deflated_sharpe,
            target_metric=self.target_metric.value,
            target_metric_value=_get_metric_value(baseline_report, self.target_metric),
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

        baseline_warnings = _audit_baseline(self.incumbent, self.config_obj)
        for w in baseline_warnings:
            logger.warning("Baseline gate check: %s", w)
        if baseline_warnings and not self.quiet:
            from rich.console import Console

            Console().print(
                "[yellow]⚠ Baseline fails gate constraints. "
                "Candidates must pass these constraints AND improve over baseline "
                f"(Sharpe {self.incumbent.observed_sharpe:.3f}).[/]"
            )

        if self.config_obj.metric_floor is not None:
            base_val = _get_metric_value(self.incumbent, self.target_metric)
            if base_val < self.config_obj.metric_floor:
                logger.warning(
                    "metric_floor (%.2f) exceeds baseline %s (%.2f). "
                    "All candidates with %s below %.2f will be rejected.",
                    self.config_obj.metric_floor,
                    self.target_metric.value,
                    base_val,
                    self.target_metric.value,
                    self.config_obj.metric_floor,
                )

    def _resume_incumbent(self) -> None:
        rows = (
            self.ledger._conn()
            .execute(
                "SELECT id, iteration, accepted, committed, report_json "
                "FROM attempts WHERE run_id = ? ORDER BY iteration ASC",
                (self.run_id,),
            )
            .fetchall()
        )

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

            self.incumbent = EvaluationReport.from_json(latest_accepted[4])
            self.incumbent_attempt_id = int(latest_accepted[0])

            ret_row = (
                self.ledger._conn()
                .execute(
                    "SELECT returns_blob, holdout_returns_blob FROM attempts WHERE run_id = ? AND iteration = ?",
                    (self.run_id, latest_accepted[1]),
                )
                .fetchone()
            )
            self.incumbent_returns = _deserialize_returns(bytes(ret_row[0])) if ret_row else pd.Series(dtype=float)
            if ret_row and ret_row[1] is not None:
                self.incumbent.holdout_net_returns = _deserialize_returns(bytes(ret_row[1]))
        else:
            from autobacktest.evaluator.report import EvaluationReport, WindowReport

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
            self.incumbent = EvaluationReport(
                strategy_name=self.strategy_name,
                dataset_hash=self.dataset_hash,
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
            self.incumbent_returns = pd.Series(dtype=float)

    def _setup_incumbent(self, resume: str | None) -> None:
        baseline_exists = False
        if resume:
            check_baseline = (
                self.ledger._conn()
                .execute("SELECT COUNT(*) FROM attempts WHERE run_id = ? AND iteration = 0", (self.run_id,))
                .fetchone()
            )
            if check_baseline and check_baseline[0] > 0:
                baseline_exists = True

        if not baseline_exists:
            self._setup_baseline()
        else:
            self._resume_incumbent()

        self.baseline_at_start = self.incumbent

    def prepare_iteration(self) -> None:
        if self.consecutive_no_accept >= STUCK_THRESHOLD:
            if self.mode != "explore":
                logger.info("Stuck threshold reached, forcing EXPLORE mode.")
            self.mode = "explore"
            self.exploit_stall = 0

        if self.start_temp is not None:
            if self.mode == "exploit":
                self.provider.temperature = self.min_temp
            else:
                if self.rolling_history:
                    failures = self.rolling_history.count(False)
                    failure_rate = failures / len(self.rolling_history)
                else:
                    failure_rate = 0.6
                self.provider.temperature = self.min_temp + (self.start_temp - self.min_temp) * failure_rate

    def build_context(self, k: int) -> AgentContext:
        current_code = self.strat_path.read_text(encoding="utf-8")
        current_yaml = self.cfg_path.read_text(encoding="utf-8")
        historical_configs = self.ledger.fetch_configs(self.dataset_hash)
        attempt_summaries = self.ledger.fetch_attempt_summaries(self.dataset_hash)

        explored_config_summary = ""
        if settings.enable_explored_config_injection and historical_configs:
            explored_config_summary = summarize_explored_space(
                historical_configs,
                max_configs=settings.explored_config_max_configs,
            )

        return AgentContext(
            strategy_name=self.strategy_name,
            strategy_code=current_code,
            config_yaml=current_yaml,
            program_text=self.spec.raw_text,
            evaluation_report=self.incumbent,
            iteration=k,
            lessons_text=self.lessons_text,
            n_historical_configs=len(historical_configs),
            last_attempt=self.last_attempt,
            last_iteration_failures=self.last_iteration_failures,
            explored_config_summary=explored_config_summary,
            attempt_history=attempt_summaries,
            mode=self.mode,
            dd_limit=self.config_obj.max_drawdown_limit,
            turnover_limit=self.config_obj.turnover_limit,
            min_return_ratio=self.config_obj.select_min_return_ratio,
        )

    def _apply_jitter(self, k: int, i: int, ev: dict[str, Any], all_tried: list[str]) -> None:
        import hashlib

        seed_bytes = f"{ev['config_yaml']}_{k}_{i}".encode()
        seed = int(hashlib.sha256(seed_bytes).hexdigest()[:8], 16) & 0xFFFFFFFF

        from autobacktest.strategy.config_jitter import jitter_config

        new_yaml, jitter_meta = jitter_config(
            ev["config_yaml"],
            all_tried,
            settings.diversity_config_threshold,
            seed=seed,
            max_attempts=settings.config_jitter_max_attempts,
            rel_step=settings.config_jitter_rel_step,
            importance=self.last_importance,
        )
        if new_yaml is not None:
            import dataclasses as _dc

            edit_jittered = _dc.replace(ev["edit"], config_yaml=new_yaml)
            rep_ok, _rep_err_code, _rep_err_detail = _validate_candidate(
                self.strategy_name, edit_jittered, self.strategies_dir, self.configs_dir
            )
            ev["config_yaml"] = new_yaml
            if rep_ok:
                ev["edit"] = edit_jittered
                ev["jitter_applied"] = True
                ev["jitter_meta"] = jitter_meta
                ev["config_similarity"] = jitter_meta["final_similarity"]
            else:
                # Jitter produced invalid code — keep original candidate, just log attempt
                ev["jitter_applied"] = False
                ev["jitter_meta"] = jitter_meta
                ev["jitter_attempted"] = True
        else:
            # Jitter couldn't find a diverse config — keep original candidate, just log attempt
            ev["jitter_applied"] = False
            ev["jitter_attempted"] = True

    def _process_raw_edit(self, k: int, edit: AgentEdit, directive: str, ctx: AgentContext) -> dict[str, Any]:
        self.n_llm_ok += 1
        if edit.lessons_text is not None and edit.lessons_text.strip():
            self.lesson_store.ingest_markdown(edit.lessons_text, self.strategy_name)
            self.lessons_text = self.lesson_store.get_filtered_markdown(self.strategy_name)

        ev = process_and_repair_candidate(
            strategy_name=self.strategy_name,
            edit=edit,
            ctx=ctx,
            directive=directive,
            provider=self.provider,
            strategies_dir=self.strategies_dir,
            configs_dir=self.configs_dir,
            lessons_text=self.lessons_text,
            k=k,
            validate_fn=_validate_candidate,
        )
        final_edit = ev["edit"]
        self.total_prompt_tokens += final_edit.prompt_tokens
        self.total_completion_tokens += final_edit.completion_tokens
        self.total_cost += final_edit.cost
        return ev

    def _diversity_config_pool(
        self, historical_configs: list[str], batch_configs: list[str], current_yaml: str
    ) -> list[str]:
        mode = settings.diversity_compare_mode
        if mode == "incumbent":
            base = [current_yaml]
        elif mode == "recent":
            n = max(0, settings.diversity_recent_n)
            base = historical_configs[-n:] if n else []
        else:
            if mode != "all":
                logger.warning("Unknown diversity_compare_mode %r; using 'all'.", mode)
            base = list(historical_configs)
        return base + batch_configs

    def _check_diversity_gate(
        self,
        k: int,
        i: int,
        ev: dict[str, Any],
        historical_configs: list[str],
        batch_configs: list[str],
        current_yaml: str,
    ) -> None:
        if not settings.enable_config_diversity_gate:
            ev["config_similarity"] = 0.0
            return
        pool = self._diversity_config_pool(historical_configs, batch_configs, current_yaml)
        e_config_yaml = ev["config_yaml"]
        max_sim = max_config_similarity(e_config_yaml, pool) if pool else 0.0
        ev["config_similarity"] = max_sim

        if settings.enable_config_jitter and max_sim > settings.diversity_config_threshold:
            self._apply_jitter(k, i, ev, pool)
        # NO hard reject here — candidate always proceeds to backtest

    def _check_exploit_gate(self, ev: dict[str, Any], current_yaml: str) -> None:
        e_config_yaml = ev["config_yaml"]
        max_sim = max_config_similarity(e_config_yaml, [current_yaml])
        if max_sim >= 0.999:
            ev["valid"] = False
            ev["validation_stage"] = "exploit_no_change"
            ev["detail"] = "Exploit candidate is identical to the incumbent config; no change to evaluate."

    def _check_identical_behavior(self, ev: dict[str, Any], current_code: str) -> None:
        from autobacktest.strategy.validator import compare_signals_to_incumbent

        is_identical, diff = compare_signals_to_incumbent(
            self.strategy_name,
            ev["strategy_code"],
            ev["config_yaml"],
            current_code,
            self.strategies_dir,
            self.configs_dir,
            epsilon=settings.identical_behavior_epsilon,
        )
        if is_identical:
            ev["valid"] = False
            ev["validation_stage"] = "identical_behavior"
            ev["detail"] = (
                f"Candidate weights are identical to incumbent (max abs weight diff {diff:.2e} "
                f"< epsilon {settings.identical_behavior_epsilon:.2e})."
            )

    def _validate_diversity_and_guards(self, k: int, candidate_results: list[dict[str, Any]]) -> None:
        historical_configs = self.ledger.fetch_configs(self.dataset_hash)
        current_yaml = self.cfg_path.read_text(encoding="utf-8")
        current_code = self.strat_path.read_text(encoding="utf-8")

        batch_configs: list[str] = []
        for i, ev in enumerate(candidate_results):
            if not ev.get("valid"):
                continue

            if self.mode == "explore":
                self._check_diversity_gate(k, i, ev, historical_configs, batch_configs, current_yaml)
            elif self.mode == "exploit":
                self._check_exploit_gate(ev, current_yaml)

            if ev.get("valid"):
                batch_configs.append(ev["config_yaml"])

        if settings.enable_identical_behavior_guard and self.mode == "explore" and current_code:
            for ev in candidate_results:
                if ev.get("valid"):
                    self._check_identical_behavior(ev, current_code)

    def generate_and_pre_validate_candidates(
        self,
        k: int,
        ctx: AgentContext,
        progress_task: Any,
        progress: Any,
    ) -> list[dict[str, Any]]:
        n = getattr(settings, "n_candidates", 3)
        raw_edits = _generate_candidates(self.provider, ctx, n)
        n_gen = sum(1 for e in raw_edits if e is not None)
        progress.update(
            progress_task,
            description=(
                f"[cyan]Iter {k}/{self.iterations} | mode={self.mode} | LLM generated {n_gen} candidates, validating..."
            ),
        )

        candidate_results: list[dict[str, Any]] = []
        for i, edit in enumerate(raw_edits):
            directive = (
                CANDIDATE_DIRECTIVES[i % len(CANDIDATE_DIRECTIVES)]
                if settings.enable_candidate_directives and ctx.mode == "explore"
                else ""
            )
            if edit is None:
                candidate_results.append({"edit": None, "directive": directive, "llm_error": True})
            else:
                ev = self._process_raw_edit(k, edit, directive, ctx)
                candidate_results.append(ev)

        self._validate_diversity_and_guards(k, candidate_results)
        return candidate_results

    def evaluate_candidates(
        self,
        k: int,
        candidate_results: list[dict[str, Any]],
        progress_task: Any,
        progress: Any,
    ) -> list[dict[str, Any]]:
        to_eval = [ev for ev in candidate_results if ev.get("valid")]
        progress.update(
            progress_task,
            description=f"[cyan]Iter {k}/{self.iterations} | Backtesting {len(to_eval)} candidates...",
        )

        if to_eval:

            def _eval_one(ev: dict[str, Any]) -> dict[str, Any]:
                r, ret, cfg, err = _eval_single_candidate(
                    self.strategy_name,
                    ev["strategy_code"],
                    ev["config_yaml"],
                    self.strategies_dir,
                    self.configs_dir,
                    self.start_date,
                    self.end_date,
                    self._eval_cache,
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

            with ThreadPoolExecutor(max_workers=min(settings.eval_max_workers, len(to_eval))) as pool:
                futures = {pool.submit(_eval_one, ev): i for i, ev in enumerate(to_eval)}
                for future in as_completed(futures):
                    future.result()

        return to_eval

    def _evaluate_candidate_gates(
        self,
        k: int,
        ev: dict[str, Any],
        peeks_this_iteration: int,
    ) -> tuple[bool, int]:
        report_k = ev["_report"]
        returns_k = ev["_returns"]
        new_config = ev["_new_config"]

        # DSR deflation
        _deflate(
            report_k,
            returns_k,
            self.ledger,
            cscv_blocks=new_config.get("cscv_blocks", 10),
            embargo_days=new_config.get("cscv_embargo_days", 5),
        )
        if self.incumbent is not None and not self.incumbent_returns.empty:
            _deflate(
                self.incumbent,
                self.incumbent_returns,
                self.ledger,
                exclude_id=self.incumbent_attempt_id,
                cscv_blocks=new_config.get("cscv_blocks", 10),
                embargo_days=new_config.get("cscv_embargo_days", 5),
            )

        # Selection gate
        sel = select(report_k, baseline=self.incumbent, target_metric=self.target_metric, config=new_config)
        ev["_sel"] = sel

        if not sel.accepted:
            ev["valid"] = False
            ev["validation_stage"] = "gate"
            ev["detail"] = sel.reason
            ev["_failed_gate"] = sel.failed_gate
            return False, peeks_this_iteration

        # Holdout peek budget check
        hist_matrix, _ = self.ledger.fetch_holdout_history(report_k.dataset_hash)
        current_peeks = len(hist_matrix.columns) if not hist_matrix.empty else 0
        if current_peeks + peeks_this_iteration >= self.holdout_peek_limit:
            total_peeks = current_peeks + peeks_this_iteration
            logger.warning(
                f"Holdout peek limit reached: {total_peeks} >= {self.holdout_peek_limit}. "
                "Aborting optimization loop immediately."
            )
            self._early_stop = True
            self._early_stop_iteration = k
            ev["valid"] = False
            ev["validation_stage"] = "holdout_peek_limit"
            ev["_peek_fail"] = True
            ev["detail"] = f"Holdout peek budget exhausted ({total_peeks} >= {self.holdout_peek_limit} peeks)."
            return False, peeks_this_iteration

        peeks_this_iteration += 1
        _deflate_holdout(report_k, self.ledger)
        if self.incumbent is not None:
            _deflate_holdout(self.incumbent, self.ledger, exclude_id=self.incumbent_attempt_id)

        cnf = confirm(report_k, baseline=self.incumbent, config=new_config)
        ev["_cnf"] = cnf

        if cnf.accepted:
            ev["_accepted"] = True
            # Post-quality returns-diversity check (after select+confirm pass)
            # Only hard-reject exact duplicates (≥ diversity_hard_threshold)
            if self.mode == "explore" and settings.enable_config_diversity_gate:
                hist_matrix, _ = self.ledger.fetch_historical_returns(self.dataset_hash)
                max_corr = 0.0
                if not hist_matrix.empty:
                    _corr_passed, max_corr = check_returns_correlation(
                        returns_k, hist_matrix, settings.diversity_hard_threshold
                    )
                    if not _corr_passed:
                        ev["valid"] = False
                        ev["validation_stage"] = "diversity_returns"
                        ev["detail"] = (
                            f"Return correlation {max_corr:.3f} exceeded hard duplicate "
                            f"threshold {settings.diversity_hard_threshold}."
                        )
                        ev["returns_correlation"] = max_corr
                        return False, peeks_this_iteration
                ev["returns_correlation"] = max_corr
            return True, peeks_this_iteration
        else:
            ev["valid"] = False
            ev["validation_stage"] = "gate"
            ev["detail"] = cnf.reason
            ev["_failed_gate"] = cnf.failed_gate
            return False, peeks_this_iteration

    def run_gates_and_select_winner(
        self,
        k: int,
        candidate_results: list[dict[str, Any]],
        progress_task: Any,
        progress: Any,
    ) -> tuple[dict[str, Any] | None, int]:
        progress.update(progress_task, description=f"[cyan]Iter {k}/{self.iterations} | Running gates...")
        winner: dict[str, Any] | None = None
        best_metric: float = -float("inf")
        peeks_this_iteration: int = 0

        for ev in candidate_results:
            if not ev.get("valid") or ev.get("_report") is None:
                continue

            accepted, peeks_this_iteration = self._evaluate_candidate_gates(k, ev, peeks_this_iteration)
            if accepted:
                metric_val = _get_metric_value(ev["_report"], self.target_metric)
                penalty = settings.diversity_returns_penalty * ev.get("returns_correlation", 0.0)
                adjusted_val = metric_val - penalty
                if adjusted_val > best_metric:
                    best_metric = adjusted_val
                    winner = ev

        return winner, peeks_this_iteration

    def commit_winner(self, k: int, winner: dict[str, Any]) -> str | None:
        w_edit = winner["edit"]
        w_report = winner["_report"]
        w_returns = winner["_returns"]
        attempt_id: int | None = None
        sha: str | None = None
        try:
            self.strat_path.write_text(w_edit.strategy_code, encoding="utf-8")
            self.cfg_path.write_text(w_edit.config_yaml, encoding="utf-8")

            attempt_id = self.ledger.record_attempt(
                run_id=self.run_id,
                iteration=k,
                strategy_name=self.strategy_name,
                dataset_hash=self.dataset_hash,
                config_yaml=w_edit.config_yaml,
                observed_sharpe=w_report.observed_sharpe,
                deflated_sharpe=w_report.deflated_sharpe,
                target_metric=self.target_metric.value,
                target_metric_value=_get_metric_value(w_report, self.target_metric),
                in_sample_max_drawdown=w_report.in_sample_metrics.max_drawdown,
                in_sample_turnover=w_report.in_sample_metrics.turnover,
                regime_passed=w_report.regime_passed,
                accepted=True,
                committed=False,
                commit_sha=None,
                rejection_reason=None,
                report_json=w_report.to_json(),
                selection_returns=w_returns,
                prompt_tokens=w_edit.prompt_tokens,
                completion_tokens=w_edit.completion_tokens,
                total_tokens=w_edit.total_tokens,
                cost=w_edit.cost,
                holdout_evaluated=True,
                holdout_observed_sharpe=w_report.holdout_metrics.sharpe_ratio,
                holdout_returns=w_report.holdout_net_returns,
            )

            sha = self.git_ledger.commit_strategy(
                self.strategy_name,
                f"iter {k}: {w_edit.reasoning[:72] if w_edit.reasoning else 'multi-candidate'}",
            )

            self.ledger.mark_committed(attempt_id, sha)

            self.incumbent = w_report
            self.incumbent_returns = w_returns
            self.incumbent_attempt_id = attempt_id
            self.n_committed += 1
            self.last_attempt = None
            self.last_iteration_failures = None
            self.consecutive_no_accept = 0
            self.mode = "exploit"
            self.exploit_stall = 0
            winner["_recorded"] = True
            return sha
        except Exception:
            self.git_ledger.rollback_strategy(self.strategy_name)
            aid = attempt_id if attempt_id is not None else "unknown"
            logger.exception(
                "Atomic commit failed for iteration %d — rolled back. "
                "Ledger row id=%s remains as committed=0 for resume recovery.",
                k,
                aid,
            )
            raise

    def record_candidates(
        self, k: int, candidate_results: list[dict[str, Any]], winner: dict[str, Any] | None, sha: str | None
    ) -> None:
        rejection_reason_map = {
            "diversity_config": "diversity_tier1_config",
            "diversity_returns": "diversity_tier2_returns",
            "validation": None,
            "eval_error": None,
            "gate": None,
        }

        def _rejection_reason(ev: dict[str, Any]) -> str | None:
            stage = ev.get("validation_stage")
            if stage in rejection_reason_map:
                return rejection_reason_map[stage] or ev.get("detail")
            if stage == "holdout_peek_limit":
                return "holdout_peek_limit_exceeded"
            if stage == "gate":
                return ev.get("detail") or ev.get("_failed_gate")
            return ev.get("detail") or stage

        def _record(ev: dict[str, Any], accepted: bool, committed: bool) -> None:
            rp = ev.get("_report")
            rt = ev.get("_returns")
            edit = ev.get("edit")
            if rp is None or edit is None:
                return
            ho_evaluated = ev.get("_cnf") is not None
            self.ledger.record_attempt(
                run_id=self.run_id,
                iteration=k,
                strategy_name=self.strategy_name,
                dataset_hash=self.dataset_hash,
                config_yaml=ev.get("config_yaml", ""),
                observed_sharpe=rp.observed_sharpe,
                deflated_sharpe=rp.deflated_sharpe,
                target_metric=self.target_metric.value,
                target_metric_value=_get_metric_value(rp, self.target_metric),
                in_sample_max_drawdown=rp.in_sample_metrics.max_drawdown,
                in_sample_turnover=rp.in_sample_metrics.turnover,
                regime_passed=rp.regime_passed,
                accepted=accepted,
                committed=committed,
                commit_sha=sha if committed else None,
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
                if ev.get("_recorded"):
                    continue
                is_winner = ev is winner
                _record(ev, accepted=is_winner, committed=is_winner)

    def update_parameter_importance(self, event: dict[str, Any]) -> None:
        try:
            imp_configs, imp_metrics = self.ledger.fetch_param_importance_data(self.dataset_hash)
            importance = compute_parameter_importance(
                imp_configs,
                imp_metrics,
                min_attempts=settings.importance_min_attempts,
                p_threshold=settings.importance_p_threshold,
            )
            if importance:
                imp_text = format_importance_lessons(importance)
                if imp_text:
                    self.lesson_store.ingest_markdown(imp_text, self.strategy_name)
                    self.lessons_text = self.lesson_store.get_filtered_markdown(self.strategy_name)
                event["parameter_importance"] = importance
                self.last_importance.clear()
                self.last_importance.update(importance)
        except Exception:
            logger.warning("Parameter importance computation failed", exc_info=True)

    def _build_candidate_failure_item(self, ev: dict[str, Any]) -> dict[str, Any]:
        edit = ev.get("edit")
        rp = ev.get("_report")
        fail_item: dict[str, Any] = {
            "passed": False,
            "stage": ev.get("validation_stage"),
            "detail": ev.get("detail"),
            "failed_gate": ev.get("_failed_gate"),
            "repair_applied": ev.get("repair_applied", False),
            "directive": ev.get("directive", ""),
            "prompt_tokens": edit.prompt_tokens if edit else 0,
            "completion_tokens": edit.completion_tokens if edit else 0,
            "total_tokens": edit.total_tokens if edit else 0,
            "cost": edit.cost if edit else 0.0,
            "deflated_sharpe": rp.deflated_sharpe if rp else None,
            "holdout_deflated_sharpe": rp.holdout_deflated_sharpe if rp else None,
            "in_sample_max_drawdown": rp.in_sample_metrics.max_drawdown if rp else None,
            "in_sample_turnover": rp.in_sample_metrics.turnover if rp else None,
            "in_sample_sharpe": rp.in_sample_metrics.sharpe_ratio if rp else None,
            "pbo": rp.pbo if rp else None,
            "returns_correlation": ev.get("returns_correlation"),
        }
        config_yaml = ev.get("config_yaml") or (edit.config_yaml if edit else None)
        if config_yaml:
            fail_item["candidate_config_yaml"] = config_yaml
        if "jitter_applied" in ev:
            fail_item["jitter_applied"] = ev["jitter_applied"]
        if "jitter_meta" in ev:
            fail_item["jitter_meta"] = ev["jitter_meta"]
        if "jitter_attempted" in ev:
            fail_item["jitter_attempted"] = ev["jitter_attempted"]
        return fail_item

    def _build_candidate_success_item(self, ev: dict[str, Any], is_winner: bool) -> dict[str, Any]:
        rp = ev.get("_report")
        edit = ev.get("edit")
        pass_item: dict[str, Any] = {
            "passed": True,
            "accepted": is_winner,
            "observed_sharpe": rp.observed_sharpe if rp else None,
            "repair_applied": ev.get("repair_applied", False),
            "directive": ev.get("directive", ""),
            "prompt_tokens": edit.prompt_tokens if edit else 0,
            "completion_tokens": edit.completion_tokens if edit else 0,
            "total_tokens": edit.total_tokens if edit else 0,
            "cost": edit.cost if edit else 0.0,
            "deflated_sharpe": rp.deflated_sharpe if rp else None,
            "holdout_deflated_sharpe": rp.holdout_deflated_sharpe if rp else None,
            "in_sample_max_drawdown": rp.in_sample_metrics.max_drawdown if rp else None,
            "in_sample_turnover": rp.in_sample_metrics.turnover if rp else None,
            "in_sample_sharpe": rp.in_sample_metrics.sharpe_ratio if rp else None,
            "pbo": rp.pbo if rp else None,
            "returns_correlation": ev.get("returns_correlation"),
        }
        config_yaml = ev.get("config_yaml") or (edit.config_yaml if edit else None)
        if config_yaml:
            pass_item["candidate_config_yaml"] = config_yaml
        if "jitter_applied" in ev:
            pass_item["jitter_applied"] = ev["jitter_applied"]
        if "jitter_meta" in ev:
            pass_item["jitter_meta"] = ev["jitter_meta"]
        return pass_item

    def _build_candidates_summary(
        self, candidate_results: list[dict[str, Any]], winner: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        summary_list = []
        for ev in candidate_results:
            if ev.get("llm_error"):
                summary_list.append({"llm_error": True})
            elif not ev.get("valid"):
                summary_list.append(self._build_candidate_failure_item(ev))
            else:
                summary_list.append(self._build_candidate_success_item(ev, ev is winner))
        return summary_list

    def _format_failure_summary(
        self,
        k: int,
        n_gen: int,
        _iter_n_val: int,
        _iter_cost: float,
        candidate_results: list[dict[str, Any]],
    ) -> str:
        self.consecutive_no_accept += 1
        if self.mode == "exploit":
            self.exploit_stall += 1
            if self.exploit_stall >= EXPLOIT_PATIENCE:
                logger.info(f"Exploit stall reached ({EXPLOIT_PATIENCE}), switching to EXPLORE mode.")
                self.mode = "explore"
                self.exploit_stall = 0
        self.last_attempt = _extract_best_failure(candidate_results)
        self.last_iteration_failures = _summarize_all_failures(candidate_results)

        stage_map = {
            "validation": "preflight",
            "diversity_config": "config_diversity",
            "diversity_returns": "returns_diversity",
            "eval_error": "backtest_error",
            "holdout_peek_limit": "peek_limit",
        }

        reasons: list[str] = []
        for ev in candidate_results:
            stage = ev.get("validation_stage")
            if stage == "gate":
                fg = ev.get("_failed_gate")
                reasons.append(f"gate({fg})" if fg else "gate")
            elif ev.get("llm_error"):
                reasons.append("llm_error")
            elif stage in stage_map:
                reasons.append(stage_map[stage])
            elif stage:
                reasons.append(stage)
        reasons_raw = ",".join(dict.fromkeys(reasons)) if reasons else "all_failed"
        reasons_str = (reasons_raw[:77] + "...") if len(reasons_raw) > 80 else reasons_raw
        return (
            f"[red]✗[/] Iter {k:>3}/{self.iterations}  "
            f"[cyan]mode={self.mode:<7}[/]"
            f"{f'  t={self.provider.temperature:.2f}' if self.start_temp is not None else ''}  "
            f"gen={n_gen}  val={_iter_n_val}  "
            f"→  FAIL  {reasons_str}  "
            f"${_iter_cost:.4f}"
        )

    def log_and_summarize_iteration(
        self,
        k: int,
        candidate_results: list[dict[str, Any]],
        winner: dict[str, Any] | None,
        sha: str | None,
        _iter_prev_cost: float,
        _iter_incumbent_sharpe: float,
        n_gen: int,
        _iter_n_val: int,
        progress: Any,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "iteration": k,
            "strategy": self.strategy_name,
            "mode": self.mode,
        }
        if self.start_temp is not None:
            event["temperature"] = self.provider.temperature

        event["candidates"] = self._build_candidates_summary(candidate_results, winner)
        _iter_cost = max(0.0, self.total_cost - _iter_prev_cost)

        if winner is not None:
            winner_idx = next(i for i, e in enumerate(candidate_results) if e is winner)
            event["winner"] = {"candidate_idx": winner_idx}
            event["gate"] = {"stage": "select", "accepted": True}
            event["commit"] = {"sha": sha}

            w_report = winner["_report"]
            delta = w_report.observed_sharpe - _iter_incumbent_sharpe
            summary = (
                f"[green]✓[/] Iter {k:>3}/{self.iterations}  "
                f"[cyan]mode={self.mode:<7}[/]"
                f"{f'  t={self.provider.temperature:.2f}' if self.start_temp is not None else ''}  "
                f"gen={n_gen}  val={_iter_n_val}  "
                f"→  Sharpe {_iter_incumbent_sharpe:.3f}→{w_report.observed_sharpe:.3f}  "
                f"({delta:+.3f})  "
                f"dd={w_report.in_sample_metrics.max_drawdown * 100:.1f}%  "
                f"to={w_report.in_sample_metrics.turnover:.2f}x  "
                f"${_iter_cost:.4f}"
            )
        else:
            event["gate"] = {
                "stage": "select",
                "accepted": False,
                "reason": "No candidate passed all gates",
            }
            event["commit"] = None
            summary = self._format_failure_summary(k, n_gen, _iter_n_val, _iter_cost, candidate_results)

        progress.console.print(summary)
        return event

    def run_iteration(self, k: int, progress: Any, progress_task: Any) -> bool:
        """Execute one optimization iteration.

        Args:
            k: 1-indexed iteration number.
            progress: Rich ``Progress`` instance for updating the bar.
            progress_task: Task ID returned by ``progress.add_task``.

        Returns:
            bool: True if the loop should break (early stop triggered).
        """
        _iter_prev_cost = self.total_cost
        _iter_incumbent_sharpe = self.incumbent.observed_sharpe if self.incumbent else 0.0
        _prev_n_committed = self.n_committed

        self.prepare_iteration()
        ctx = self.build_context(k)
        candidate_results = self.generate_and_pre_validate_candidates(k, ctx, progress_task, progress)
        n_gen = sum(1 for ev in candidate_results if not ev.get("llm_error"))
        evaluated = self.evaluate_candidates(k, candidate_results, progress_task, progress)
        _iter_n_val = len(evaluated)

        winner, _peeks = self.run_gates_and_select_winner(k, candidate_results, progress_task, progress)

        sha: str | None = None
        if winner is not None:
            sha = self.commit_winner(k, winner)

        self.record_candidates(k, candidate_results, winner, sha)

        event = self.log_and_summarize_iteration(
            k,
            candidate_results,
            winner,
            sha,
            _iter_prev_cost,
            _iter_incumbent_sharpe,
            n_gen,
            _iter_n_val,
            progress,
        )

        self.update_parameter_importance(event)
        self.event_log.write(event)

        # Rolling history (used for temperature decay)
        self.rolling_history.append(self.n_committed > _prev_n_committed)
        if len(self.rolling_history) > 5:
            self.rolling_history.pop(0)

        # Advance progress bar
        progress.update(
            progress_task,
            advance=1,
            description=(
                f"[cyan]Optimizing {self.strategy_name}... (Incumbent Sharpe: {self.incumbent.observed_sharpe:.3f})"
                if self.incumbent
                else ""
            ),
        )

        # Early stop check
        if self._early_stop:
            return True
        if self.early_stop_patience > 0 and self.consecutive_no_accept >= self.early_stop_patience:
            logger.info(
                f"Early stop: no acceptance in {self.consecutive_no_accept} consecutive iterations. "
                f"Stopping at iteration {k}/{self.iterations}."
            )
            self._early_stop = True
            self._early_stop_iteration = k
            return True

        return False

    def cleanup(self) -> None:
        if self.start_temp is not None:
            self.provider.temperature = self.start_temp
        if self.incumbent is not None:
            try:
                hist_matrix, hist_sharpes = self.ledger.fetch_historical_returns(self.incumbent.dataset_hash)
                if not hist_matrix.empty and len(hist_sharpes) > 1:
                    n = max(1, calculate_effective_trials(hist_matrix))
                    self.incumbent.effective_trials = n
                    self.incumbent.deflated_sharpe = calculate_psr_dsr(self.incumbent_returns, hist_sharpes, n)
            except Exception as exc:
                logger.warning("Failed to refresh final report DSR: %s", exc)
            with contextlib.suppress(Exception):
                _deflate_holdout(self.incumbent, self.ledger, exclude_id=self.incumbent_attempt_id)
        with contextlib.suppress(Exception):
            self.event_log.close()
        with contextlib.suppress(Exception):
            self.lesson_store.close()
        with contextlib.suppress(Exception):
            self.ledger.close()


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
        holdout_peek_limit: Maximum holdout peeks before early termination.
            Defaults to 20.
        early_stop_patience: Consecutive rejections before early stopping.
            Configurable via ``AUTOBACKTEST_EARLY_STOP_PATIENCE`` env var.
            Set to 0 to disable. Defaults to ``settings.early_stop_patience``.
        resume: Run ID to resume a previously interrupted optimization.
            When provided the loop recovers the incumbent state from the
            ledger and continues from the next unprocessed iteration.
        quiet: Suppress non-critical warnings and reduce terminal noise
            during the optimization loop. Defaults to False.

    Returns:
        OrchestratorResult: Summary of the final optimization run outcomes.
        ``early_stopped`` is True when the loop exited early due to
        ``early_stop_patience`` consecutive rejections or the holdout-peek
        budget being exhausted.

    Raises:
        FileNotFoundError: If the target strategy or configuration files are missing.
    """
    state = _OptimizationState(
        program_path=program_path,
        strategy_name=strategy_name,
        iterations=iterations,
        provider=provider,
        run_dir=run_dir,
        strategies_dir=strategies_dir,
        configs_dir=configs_dir,
        target_metric=target_metric,
        repo_path=repo_path,
        start_date=start_date,
        end_date=end_date,
        holdout_peek_limit=holdout_peek_limit,
        early_stop_patience=early_stop_patience,
        quiet=quiet,
    )
    state.setup(resume)

    try:
        start_k = 1
        if resume:
            rows = (
                state.ledger._conn()
                .execute("SELECT iteration FROM attempts WHERE run_id = ?", (state.run_id,))
                .fetchall()
            )
            if rows:
                start_k = max(row[0] for row in rows) + 1

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            disable=False,
        ) as progress:
            incumbent_sharpe = state.incumbent.observed_sharpe if state.incumbent else 0.0
            task = progress.add_task(
                f"[cyan]Optimizing {strategy_name}... (Incumbent Sharpe: {incumbent_sharpe:.3f})",
                total=iterations,
            )
            if start_k > 1:
                progress.advance(task, start_k - 1)

            for k in range(start_k, iterations + 1):
                should_break = state.run_iteration(k, progress, task)
                if should_break:
                    break

        if state.n_llm_ok == 0:
            raise RuntimeError("Zero successful LLM calls during optimization run. All iterations failed.")
    except KeyboardInterrupt:
        if state.incumbent is None:
            raise
        if not quiet:
            from rich.console import Console

            Console().print(
                "\n[bold yellow]⚠ Optimization loop interrupted by user. Performing graceful shutdown...[/]"
            )
    finally:
        state.cleanup()

    assert state.incumbent is not None
    return OrchestratorResult(
        run_id=state.run_id,
        branch=state.branch,
        n_committed=state.n_committed,
        final_report=state.incumbent,
        total_prompt_tokens=state.total_prompt_tokens,
        total_completion_tokens=state.total_completion_tokens,
        total_cost=state.total_cost,
        baseline_report=state.baseline_at_start,
        early_stopped=state._early_stop_iteration > 0,
        early_stop_iteration=state._early_stop_iteration or None,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _audit_baseline(report: EvaluationReport, config: StrategyConfig) -> list[str]:
    """Check whether the baseline strategy passes hard gate constraints.

    Returns a list of human-readable warning messages for every constraint
    the baseline violates.  An empty list means the baseline is clean.
    """
    warnings: list[str] = []
    dd = report.in_sample_metrics.max_drawdown
    if not math.isnan(dd) and dd > config.max_drawdown_limit:
        warnings.append(
            f"Baseline max drawdown ({dd * 100:.1f}%) exceeds config limit ({config.max_drawdown_limit * 100:.0f}%)."
        )
    to = report.in_sample_metrics.turnover
    if not math.isnan(to) and to > config.turnover_limit:
        warnings.append(f"Baseline turnover ({to:.2f}x) exceeds config limit ({config.turnover_limit:.1f}x).")
    if not report.regime_passed:
        warnings.append("Baseline fails regime stress tests.")
    return warnings


def _get_metric_value(report: EvaluationReport, metric: TargetMetric) -> float:
    """Extract the target metric value from the in-sample walk-forward aggregate.

    Args:
        report: Evaluation report containing in-sample metrics.
        metric: The target metric to extract (Sharpe, Sortino, or IR).

    Returns:
        float: The metric value, or 0.0 for IR when None.
    """
    if metric == TargetMetric.SHARPE:
        return report.in_sample_metrics.sharpe_ratio
    elif metric == TargetMetric.SORTINO:
        return report.in_sample_metrics.sortino_ratio
    else:  # INFORMATION_RATIO
        return report.in_sample_metrics.information_ratio or 0.0


def _populate_failure_details(failure: dict[str, Any], stage: str, ev: dict[str, Any]) -> None:
    if stage == "validation":
        failure["error_code"] = ev.get("error_code")
        failure["detail"] = ev.get("detail")
    elif stage == "eval_error":
        failure["detail"] = ev.get("detail")
    elif stage == "gate":
        failure["rejection_reason"] = ev.get("detail")
        failure["failed_gate"] = ev.get("_failed_gate")
        rp = ev.get("_report")
        if rp:
            failure["candidate_metrics"] = {
                "Sharpe": rp.in_sample_metrics.sharpe_ratio,
                "Sortino": rp.in_sample_metrics.sortino_ratio,
                "Information Ratio": rp.in_sample_metrics.information_ratio,
                "Max Drawdown": rp.in_sample_metrics.max_drawdown,
                "Turnover": rp.in_sample_metrics.turnover,
            }
    elif stage in ("diversity_returns", "diversity_config", "holdout_peek_limit", "identical_behavior"):
        failure["detail"] = ev.get("detail")


def _extract_best_failure(candidate_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the most descriptive failure from parallel candidate results to present as feedback.

    Prioritises failures by stage precedence (validation > eval_error >
    identical_behavior > gate > diversity_returns > diversity_config).
    Falls back to ``"all_candidates_failed"`` when no failure is found.

    Args:
        candidate_results: List of candidate evaluation result dicts.

    Returns:
        dict: The best failure with keys ``stage``, ``detail``, and
        optionally ``error_code``, ``candidate_metrics``, etc.
    """
    stages_priority = [
        "validation",
        "eval_error",
        "identical_behavior",
        "gate",
        "diversity_returns",
        "diversity_config",
        "holdout_peek_limit",
    ]

    target_ev = None
    target_stage = None
    for stage in stages_priority:
        for ev in candidate_results:
            if ev.get("validation_stage") == stage:
                target_ev = ev
                target_stage = stage
                break
        if target_ev:
            break

    if target_ev:
        edit = target_ev.get("edit")
        failure = {
            "stage": target_stage,
            "candidate_strategy_code": target_ev.get("strategy_code") or (edit.strategy_code if edit else ""),
            "candidate_config_yaml": target_ev.get("config_yaml") or (edit.config_yaml if edit else ""),
        }
        assert target_stage is not None
        _populate_failure_details(failure, target_stage, target_ev)
        return failure

    for ev in candidate_results:
        if ev.get("llm_error"):
            return {
                "stage": "llm_error",
                "detail": "LLM failed to return a valid candidate or parsing failed.",
            }

    return {
        "stage": "all_candidates_failed",
        "detail": "No candidate passed all gates.",
    }


def _summarize_all_failures(candidate_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize all failures observed across parallel candidates in an iteration.

    Extracts stage, error code, detail (truncated to 120 chars), and
    numeric parameter fingerprints for each failed candidate.

    Args:
        candidate_results: List of candidate evaluation result dicts.

    Returns:
        list[dict]: Each with keys ``stage``, ``error_code``, ``detail``, ``params``.
    """
    failures = []
    for ev in candidate_results:
        if not ev.get("valid") or ev.get("llm_error"):
            stage = ev.get("validation_stage") or ("llm_error" if ev.get("llm_error") else "unknown")
            error_code = ev.get("error_code")
            detail = ev.get("detail") or ""
            if len(detail) > 120:
                detail = detail[:117] + "..."

            params = {}
            config_yaml = ev.get("config_yaml")
            if not config_yaml and ev.get("edit"):
                config_yaml = ev["edit"].config_yaml
            if config_yaml:
                try:
                    fp = extract_config_fingerprint(config_yaml)
                    params = fp.numeric_params
                except Exception:
                    pass

            failures.append(
                {
                    "stage": stage,
                    "error_code": error_code,
                    "detail": detail,
                    "params": params,
                }
            )
    return failures


def _load_signals(path: Path) -> Any:
    """Dynamically import generate_signals from a strategy .py file.

    Delegates to ``optimization.eval_manager.load_signals``.

    Args:
        path: Path to the strategy ``.py`` file.

    Returns:
        Any: The ``generate_signals`` function.
    """
    return load_signals(path)


def _validate_candidate(
    strategy_name: str,
    edit: AgentEdit,
    strategies_dir: Path,
    configs_dir: Path,
) -> tuple[bool, str | None, str | None]:
    """Validate candidate edit via temp files.

    Writes the candidate's code and config to temporary files, runs
    ``preflight()``, and cleans up.  Duplicates ``candidate.validate_candidate``
    for local reasoning.

    Args:
        strategy_name: The target strategy name.
        edit: ``AgentEdit`` containing strategy code and config YAML.
        strategies_dir: Directory for temporary strategy files.
        configs_dir: Directory for temporary config files.

    Returns:
        tuple[bool, str | None, str | None]: (passed, error_code, error_detail).
    """
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
    """Generate N candidate edits in parallel.

    Delegates to ``optimization.candidate.generate_candidates``.

    Args:
        provider: LLM provider.
        ctx: Shared ``AgentContext``.
        n: Number of parallel candidates.

    Returns:
        list[AgentEdit | None]: One entry per slot.
    """
    return generate_candidates(provider, ctx, n)


def _eval_single_candidate(
    strategy_name: str,
    strategy_code: str,
    config_yaml: str,
    strategies_dir: Path,
    configs_dir: Path,
    start_date: str,
    end_date: str,
    _eval_cache: _CacheProtocol,
) -> tuple[EvaluationReport | None, pd.Series[Any] | None, dict[str, Any] | None, str | None]:
    """Evaluate one candidate via temp files.

    Duplicates ``optimization.eval_manager.eval_single_candidate`` for local
    reasoning.  Writes temp files, loads signals, evaluates, cleans up.

    Args:
        strategy_name: The target strategy name.
        strategy_code: Source code of the candidate.
        config_yaml: YAML config of the candidate.
        strategies_dir: Path to strategies directory.
        configs_dir: Path to configs directory.
        start_date: Start of backtest period.
        end_date: End of backtest period.
        _eval_cache: Memoisation cache for evaluation results.

    Returns:
        tuple: ``(report, in_sample_returns, flat_config, error_str)``.
        All ``None`` when evaluation fails.
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
    exclude_id: int | None = None,
    cscv_blocks: int = 10,
    embargo_days: int = 5,
) -> None:
    """Deflate selection metrics via the ledger's multi-trial history.

    Delegates to ``optimization.persistence.deflate_selection``.

    Args:
        report: EvaluationReport to mutate in-place.
        selection_returns: In-sample walk-forward net returns.
        ledger: Ledger store.
        exclude_id: Optional attempt ID to exclude.
        cscv_blocks: CSCV block count for PBO computation.
        embargo_days: Block embargo days.
    """
    deflate_selection(
        report,
        selection_returns,
        ledger,
        exclude_id=exclude_id,
        cscv_blocks=cscv_blocks,
        embargo_days=embargo_days,
    )


def _deflate_holdout(
    report: EvaluationReport,
    ledger: LedgerStore,
    exclude_id: int | None = None,
) -> None:
    """Deflate holdout DSR by the holdout-peek count.

    Delegates to ``optimization.persistence.deflate_holdout``.

    Args:
        report: EvaluationReport to mutate in-place.
        ledger: Ledger store.
        exclude_id: Optional attempt ID to exclude.
    """
    deflate_holdout(report, ledger, exclude_id=exclude_id)
