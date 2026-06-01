"""Unit and E2E tests for the lessons memory system in the orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import git
import pytest

from autobacktest.gate import TargetMetric
from autobacktest.lessons import LessonStore
from autobacktest.llm.base import AgentContext, AgentEdit
from autobacktest.llm.mock_provider import MockProvider
from autobacktest.orchestrator import run_optimization
from tests.test_orchestrator_e2e import (
    IMPROVED_CONFIG,
    IMPROVED_STRATEGY,
    STRATEGY_CONFIG,
    _make_fake_provider,
    _make_synthetic_prices,
)


def test_lessons_roundtrip_and_rollback(project_root_with_lessons: Path) -> None:
    """Verify lessons are stored/accumulated in the DB regardless of git rollback."""
    synthetic_prices = _make_synthetic_prices()
    fake_instance = _make_fake_provider(synthetic_prices)

    # 1st iteration: Accepted edit (improves strategy) with updated lessons_text
    improved_edit = AgentEdit(
        strategy_code=IMPROVED_STRATEGY,
        config_yaml=IMPROVED_CONFIG,
        reasoning="Switch to HIGH asset.",
        raw_response="{}",
        lessons_text=(
            "### Switched to HIGH asset\n- **Type:** PERFORMANCE_INSIGHT\n- Switched to HIGH asset and succeeded.\n"
        ),
    )

    # 2nd iteration: Rejected edit (syntax-invalid triggers validation failure)
    bad_edit = AgentEdit(
        strategy_code="def generate_signals(): syntax error here",
        config_yaml=STRATEGY_CONFIG,
        reasoning="Bad edit.",
        raw_response="{}",
        lessons_text=("### Validation failure example\n- **Type:** BUG\n- This failed validation.\n"),
    )

    class ScriptedMockProvider(MockProvider):
        def generate_edit(self, context: AgentContext) -> AgentEdit:
            self.calls.append(context)
            if context.iteration == 1:
                assert "Baseline strategy loaded" in context.lessons_text
                return improved_edit
            else:
                assert "Switched to HIGH asset" in context.lessons_text
                return bad_edit

    mock_provider = ScriptedMockProvider()

    with patch(
        "autobacktest.evaluator.evaluate.CachedDataProvider",
        return_value=fake_instance,
    ):
        result = run_optimization(
            program_path=project_root_with_lessons / "program.md",
            strategy_name="toy",
            iterations=2,
            provider=mock_provider,
            run_dir=project_root_with_lessons / "runs",
            strategies_dir=project_root_with_lessons / "strategies",
            configs_dir=project_root_with_lessons / "configs",
            target_metric=TargetMetric.SHARPE,
            repo_path=project_root_with_lessons,
            start_date="2013-01-01",
            end_date="2025-01-01",
        )

    # Iteration 1 succeeded, Iteration 2 failed (syntax error triggers rollback)
    assert result.n_committed == 1

    # Verify both lessons are stored in the DB (lessons persist regardless of rollback)
    store = LessonStore(project_root_with_lessons / "runs" / "lessons.db")
    all_lessons = store.all_lessons(strategy="toy")
    bodies = [lesson["body"] for lesson in all_lessons]

    assert any("Switched to HIGH asset" in b for b in bodies)
    assert any("This failed validation" in b for b in bodies)
    store.close()

    # Verify git rollback: the committed HEAD does NOT have the bad strategy code
    repo = git.Repo(project_root_with_lessons)
    committed_code = repo.git.show(f"{result.branch}:strategies/toy.py")
    assert 'weights["HIGH"] = 1.0' in committed_code
    assert "syntax error" not in committed_code


@pytest.mark.parametrize("lessons_text", [None, "", "   \n"])
def test_orchestrator_preserves_lessons_when_update_missing_or_blank(
    project_root_with_lessons: Path,
    lessons_text: str | None,
) -> None:
    synthetic_prices = _make_synthetic_prices()
    fake_instance = _make_fake_provider(synthetic_prices)

    blank_edit = AgentEdit(
        strategy_code="import os\n",
        config_yaml=STRATEGY_CONFIG,
        reasoning="Invalid edit without a lessons update.",
        raw_response="{}",
        lessons_text=lessons_text,
    )
    mock_provider = MockProvider(response=blank_edit)

    with patch(
        "autobacktest.evaluator.evaluate.CachedDataProvider",
        return_value=fake_instance,
    ):
        run_optimization(
            program_path=project_root_with_lessons / "program.md",
            strategy_name="toy",
            iterations=1,
            provider=mock_provider,
            run_dir=project_root_with_lessons / "runs",
            strategies_dir=project_root_with_lessons / "strategies",
            configs_dir=project_root_with_lessons / "configs",
            target_metric=TargetMetric.SHARPE,
            repo_path=project_root_with_lessons,
            start_date="2013-01-01",
            end_date="2025-01-01",
        )

    # Verify the originally migrated lessons are still intact
    store = LessonStore(project_root_with_lessons / "runs" / "lessons.db")
    all_lessons = store.all_lessons(strategy="toy")
    bodies = [lesson["body"] for lesson in all_lessons]
    assert any("Baseline strategy loaded" in b for b in bodies)
    blank_body = "Invalid edit without a lessons update."
    assert not any(blank_body in b for b in bodies)
    store.close()
