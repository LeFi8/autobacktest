"""Unit and E2E tests for the lessons memory system in the orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import git
import pytest

from autobacktest.gate import TargetMetric
from autobacktest.llm.base import AgentEdit
from autobacktest.llm.mock_provider import MockProvider
from autobacktest.orchestrator import run_optimization
from tests.test_orchestrator_e2e import (
    BASELINE_STRATEGY,
    IMPROVED_STRATEGY,
    STRATEGY_CONFIG,
    _make_fake_provider,
    _make_synthetic_prices,
)

PROGRAM_MD = """\
# Objective
Maximize Sharpe.

# Constraints
None.
"""


@pytest.fixture
def project_root_with_lessons(tmp_path: Path) -> Path:
    """Set up a project directory with git repo, strategy files, and lessons.md."""
    strat_dir = tmp_path / "strategies"
    cfg_dir = tmp_path / "configs"
    run_dir = tmp_path / "runs"
    strat_dir.mkdir()
    cfg_dir.mkdir()
    run_dir.mkdir()

    (strat_dir / "toy.py").write_text(BASELINE_STRATEGY, encoding="utf-8")
    (cfg_dir / "toy.yaml").write_text(STRATEGY_CONFIG, encoding="utf-8")
    (tmp_path / "program.md").write_text(PROGRAM_MD, encoding="utf-8")

    # Create lessons.md
    initial_lessons = "# Lessons\n\n- Baseline strategy loaded.\n"
    (tmp_path / "lessons.md").write_text(initial_lessons, encoding="utf-8")

    repo = git.Repo.init(tmp_path)
    repo.config_writer().set_value("user", "name", "Test User").release()
    repo.config_writer().set_value("user", "email", "test@test.com").release()
    repo.index.add(["strategies/toy.py", "configs/toy.yaml", "lessons.md"])
    repo.index.commit("initial commit with lessons")

    return tmp_path


def test_lessons_roundtrip_and_rollback(project_root_with_lessons: Path) -> None:
    """Verify lessons.md is updated/committed or rolled back as expected."""
    synthetic_prices = _make_synthetic_prices()
    fake_instance = _make_fake_provider(synthetic_prices)

    # 1st iteration: Accepted edit (improves strategy) with updated lessons_text
    improved_edit = AgentEdit(
        strategy_code=IMPROVED_STRATEGY,
        config_yaml=STRATEGY_CONFIG,
        reasoning="Switch to HIGH asset.",
        raw_response="{}",
        lessons_text="# Lessons\n\n- Switched to HIGH asset and succeeded.\n",
    )

    # 2nd iteration: Rejected edit (e.g. low Sharpe or invalid config/exception)
    # We trigger validation exception by making the strategy code syntax-invalid
    bad_edit = AgentEdit(
        strategy_code="def generate_signals(): syntax error here",
        config_yaml=STRATEGY_CONFIG,
        reasoning="Bad edit.",
        raw_response="{}",
        lessons_text="# Lessons\n\n- This should be rolled back and not persisted.\n",
    )

    class ScriptedMockProvider(MockProvider):
        def generate_edit(self, context):
            self.calls.append(context)
            if context.iteration == 1:
                # Verify initial lessons were passed to LLM
                assert "Baseline strategy loaded" in context.lessons_text
                return improved_edit
            else:
                # Verify updated lessons from iteration 1 were passed to LLM
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

    # Check that lessons.md on disk has Iteration 2 failure content
    # preserved (cumulative learning)
    lessons_path = project_root_with_lessons / "lessons.md"
    lessons_content = lessons_path.read_text(encoding="utf-8")
    assert "This should be rolled back and not persisted" in lessons_content

    # Verify lessons.md git commits: the committed HEAD does NOT have the bad edit
    repo = git.Repo(project_root_with_lessons)
    committed_lessons = repo.git.show(f"{result.branch}:lessons.md")
    assert "Switched to HIGH asset and succeeded" in committed_lessons
    assert "This should be rolled back" not in committed_lessons


@pytest.mark.parametrize("lessons_text", [None, "", "   \n"])
def test_orchestrator_preserves_lessons_when_update_missing_or_blank(
    project_root_with_lessons: Path,
    lessons_text: str | None,
) -> None:
    synthetic_prices = _make_synthetic_prices()
    fake_instance = _make_fake_provider(synthetic_prices)
    initial_lessons = (project_root_with_lessons / "lessons.md").read_text(
        encoding="utf-8"
    )

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

    assert (project_root_with_lessons / "lessons.md").read_text(
        encoding="utf-8"
    ) == initial_lessons
