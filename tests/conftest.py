from pathlib import Path
import warnings

import git
import pytest

from autobacktest.config import settings
from autobacktest.lessons import LessonStore
from tests.test_orchestrator_e2e import BASELINE_STRATEGY, STRATEGY_CONFIG

PROGRAM_MD = """\
# Objective
Maximize Sharpe.

# Constraints
None.
"""


@pytest.fixture(autouse=True, scope="session")
def setup_test_environment() -> None:
    """Configure low-latency settings overrides for unit testing."""
    settings.sandbox_timeout = 2
    settings.enable_llm_repair = False
    settings.enable_candidate_directives = False
    settings.enable_explored_config_injection = False
    settings.enable_identical_behavior_guard = False
    warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)


@pytest.fixture
def project_root_with_lessons(tmp_path: Path) -> Path:
    """Set up a project directory with git repo and strategy files."""
    strat_dir = tmp_path / "strategies"
    cfg_dir = tmp_path / "configs"
    run_dir = tmp_path / "runs"
    strat_dir.mkdir()
    cfg_dir.mkdir()
    run_dir.mkdir()

    (strat_dir / "toy.py").write_text(BASELINE_STRATEGY, encoding="utf-8")
    (cfg_dir / "toy.yaml").write_text(STRATEGY_CONFIG, encoding="utf-8")
    (tmp_path / "program.md").write_text(PROGRAM_MD, encoding="utf-8")

    repo = git.Repo.init(tmp_path)
    repo.config_writer().set_value("user", "name", "Test User").release()
    repo.config_writer().set_value("user", "email", "test@test.com").release()
    repo.index.add(["strategies/toy.py", "configs/toy.yaml"])
    repo.index.commit("initial commit")

    # Seed an initial lesson directly into the DB
    store = LessonStore(tmp_path / "runs" / "lessons.db")
    store.store_lesson(
        strategy="toy",
        title="Baseline strategy loaded",
        body="- Baseline strategy loaded.",
        lesson_type="STRUCTURAL",
    )
    store.close()

    return tmp_path
