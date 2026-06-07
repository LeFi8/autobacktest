"""Candidate generation, validation, and repair helpers."""

from __future__ import annotations

import dataclasses
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from autobacktest.config import settings
from autobacktest.llm.base import AgentContext, AgentEdit, LLMError, LLMProvider
from autobacktest.strategy.codemod import repair_strategy_code
from autobacktest.strategy.validator import preflight

logger = logging.getLogger(__name__)

CANDIDATE_DIRECTIVES = [
    "structurally change the signal-generation logic",
    "explore an untried parameter region far from explored values",
    "change the risk-management/leverage mechanism",
]


def validate_candidate(
    strategy_name: str,
    edit: AgentEdit,
    strategies_dir: Path,
    configs_dir: Path,
) -> tuple[bool, str | None, str | None]:
    """Validate candidate edit via temp files.

    Writes the candidate's code and config to temporary files, runs the
    full preflight validation suite, and cleans up the temp files.

    Args:
        strategy_name: The target strategy name.
        edit: ``AgentEdit`` containing strategy code and config YAML.
        strategies_dir: Directory for temporary strategy ``.py`` files.
        configs_dir: Directory for temporary config ``.yaml`` files.

    Returns:
        tuple[bool, str | None, str | None]: ``(passed, error_code, error_detail)``.
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


def generate_candidates(
    provider: LLMProvider,
    ctx: AgentContext,
    n: int,
) -> list[AgentEdit | None]:
    """Generate N candidate edits in parallel, returning None for transient failures.

    In explore mode, each candidate receives a unique diversity directive
    to encourage structurally different mutations.  Non-retryable errors
    (e.g. auth failures) are raised immediately; transient LLM errors
    return ``None`` for that slot.

    Args:
        provider: LLM provider used to generate edits.
        ctx: Shared ``AgentContext`` passed to each candidate.
        n: Number of parallel candidates to generate.

    Returns:
        list[AgentEdit | None]: One entry per slot. ``None`` indicates a
        transient failure in that slot.
    """

    def _try(c: AgentContext) -> AgentEdit | None:
        try:
            return provider.generate_edit(c)
        except LLMError as e:
            if not e.retryable:
                raise
            return None

    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = []
        for i in range(n):
            if settings.enable_candidate_directives and ctx.mode == "explore":
                dir_str = CANDIDATE_DIRECTIVES[i % len(CANDIDATE_DIRECTIVES)]
                c = dataclasses.replace(ctx, directive=dir_str)
            else:
                c = ctx
            futures.append(pool.submit(_try, c))
        return [f.result() for f in futures]


def process_and_repair_candidate(
    strategy_name: str,
    edit: AgentEdit,
    ctx: AgentContext,
    directive: str,
    provider: LLMProvider,
    strategies_dir: Path,
    configs_dir: Path,
    lessons_text: str,
    k: int,
    validate_fn: Any = None,
) -> dict[str, Any]:
    """Process, validate, and optionally repair a generated candidate edit.

    Applies deterministic codemod repair first (pandas deprecation fixes),
    then runs preflight validation.  When validation fails and LLM repair is
    enabled, launches up to ``max_repair_attempts`` LLM-based repair rounds.

    Args:
        strategy_name: The target strategy name.
        edit: The initial ``AgentEdit`` from the LLM.
        ctx: The ``AgentContext`` used for repair rounds.
        directive: The diversity directive assigned to this candidate.
        provider: LLM provider for repair calls.
        strategies_dir: Path to strategies directory.
        configs_dir: Path to configs directory.
        lessons_text: Curated lessons text for the repair context.
        k: Iteration number (for log correlation).
        validate_fn: Override for ``validate_candidate`` (test injection).

    Returns:
        dict with keys:
        - ``valid`` (bool): Whether the candidate passed all checks.
        - ``strategy_code`` (str): Final strategy source code.
        - ``config_yaml`` (str): Final config YAML.
        - ``prompt_tokens``, ``completion_tokens``, ``total_tokens``, ``cost``
          (int/int/int/float): Accumulated token/cost usage.
        - ``edit`` (AgentEdit): The final edit object.
        - ``repair_applied`` (bool): Whether an LLM repair was applied.
        When invalid, additionally:
        - ``validation_stage`` (str): ``"validation"`` or ``"codemod"``.
        - ``detail`` (str | None): Error details.
        - ``error_code`` (str | None): Error code string.
    """
    if validate_fn is None:
        validate_fn = validate_candidate
    ev: dict[str, Any] = {"edit": edit, "directive": directive}

    # Apply deterministic pandas codemod (repairs deprecated API calls)
    if settings.enable_codemod_repair:
        repaired_code, applied_fixes = repair_strategy_code(edit.strategy_code)
        if applied_fixes:
            edit = dataclasses.replace(edit, strategy_code=repaired_code)
            logger.info("codemod repaired candidate in iter %s: %s", k, applied_fixes)

    # Validate
    orig_ok, orig_err_code, orig_err_detail = validate_fn(strategy_name, edit, strategies_dir, configs_dir)

    ok, err_code, err_detail = orig_ok, orig_err_code, orig_err_detail
    repair_applied = False

    if not ok and settings.enable_llm_repair:
        current_edit = edit
        for _attempt_idx in range(settings.max_repair_attempts):
            repair_request = {
                "failed_code": current_edit.strategy_code,
                "failed_config_yaml": current_edit.config_yaml,
                "error_code": err_code,
                "error_detail": err_detail,
            }
            repair_ctx = dataclasses.replace(
                ctx,
                strategy_code=current_edit.strategy_code,
                config_yaml=current_edit.config_yaml,
                lessons_text=lessons_text,
                repair_request=repair_request,
                directive=ev.get("directive", ""),
            )
            try:
                repair_edit = provider.generate_edit(repair_ctx)
            except LLMError as e:
                if not e.retryable:
                    raise
                break

            if repair_edit is not None:
                edit = dataclasses.replace(
                    repair_edit,
                    prompt_tokens=edit.prompt_tokens + repair_edit.prompt_tokens,
                    completion_tokens=edit.completion_tokens + repair_edit.completion_tokens,
                    total_tokens=edit.total_tokens + repair_edit.total_tokens,
                    cost=edit.cost + repair_edit.cost,
                )

                if settings.enable_codemod_repair:
                    repaired_code, applied_fixes = repair_strategy_code(edit.strategy_code)
                    if applied_fixes:
                        edit = dataclasses.replace(edit, strategy_code=repaired_code)

                rep_ok, rep_err_code, rep_err_detail = validate_fn(strategy_name, edit, strategies_dir, configs_dir)
                if rep_ok:
                    ok, err_code, err_detail = rep_ok, rep_err_code, rep_err_detail
                    repair_applied = True
                    break
                else:
                    current_edit = edit
                    err_code, err_detail = rep_err_code, rep_err_detail

        if not repair_applied:
            ok, err_code, err_detail = orig_ok, orig_err_code, orig_err_detail

    ev["repair_applied"] = repair_applied
    if ok:
        ev["valid"] = True
        ev["strategy_code"] = edit.strategy_code
        ev["config_yaml"] = edit.config_yaml
        ev["prompt_tokens"] = edit.prompt_tokens
        ev["completion_tokens"] = edit.completion_tokens
        ev["total_tokens"] = edit.total_tokens
        ev["cost"] = edit.cost
        ev["edit"] = edit
    else:
        ev["valid"] = False
        ev["validation_stage"] = "validation"
        ev["detail"] = err_detail
        ev["error_code"] = err_code
        ev["strategy_code"] = edit.strategy_code
        ev["config_yaml"] = edit.config_yaml

    return ev
