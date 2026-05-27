from __future__ import annotations

from pathlib import Path

import git
import pytest

from autobacktest.ledger.git_ops import GitLedger


@pytest.fixture
def repo_setup(tmp_path: Path) -> tuple[git.Repo, Path]:
    # Create strategies/ and configs/ dirs
    strat_dir = tmp_path / "strategies"
    cfg_dir = tmp_path / "configs"
    strat_dir.mkdir()
    cfg_dir.mkdir()

    # Write initial strategy and config
    (strat_dir / "toy.py").write_text(
        "def generate_signals(p, c):\n    return p.head(1)\n"
    )
    (cfg_dir / "toy.yaml").write_text("universe: [SPY]\n")

    # Init repo, configure user, add files, make initial commit
    repo = git.Repo.init(tmp_path)
    repo.config_writer().set_value("user", "name", "Test").release()
    repo.config_writer().set_value("user", "email", "test@test.com").release()
    repo.index.add(["strategies/toy.py", "configs/toy.yaml"])
    repo.index.commit("initial commit")

    return repo, tmp_path


def test_create_run_branch(repo_setup: tuple[git.Repo, Path]) -> None:
    repo, tmp_path = repo_setup
    ledger = GitLedger(tmp_path)
    branch_name = ledger.create_run_branch("test-123")
    assert branch_name == "autobacktest/test-123"
    assert ledger.current_branch == "autobacktest/test-123"
    branch_names = [b.name for b in repo.branches]
    assert "autobacktest/test-123" in branch_names


def test_commit_strategy_stages_only_two_files(
    repo_setup: tuple[git.Repo, Path],
) -> None:
    repo, tmp_path = repo_setup
    ledger = GitLedger(tmp_path)
    ledger.create_run_branch("run-001")

    # Modify strategy and config
    (tmp_path / "strategies" / "toy.py").write_text(
        "def generate_signals(p, c):\n    return p.tail(1)\n"
    )
    (tmp_path / "configs" / "toy.yaml").write_text("universe: [QQQ]\n")

    # Create unrelated file that must NOT appear in the commit
    (tmp_path / "README.md").write_text("# readme\n")

    hexsha = ledger.commit_strategy("toy", "test commit")
    commit = repo.commit(hexsha)

    assert "strategies/toy.py" in commit.stats.files
    assert "configs/toy.yaml" in commit.stats.files
    assert "README.md" not in commit.stats.files


def test_commit_strategy_returns_hexsha(repo_setup: tuple[git.Repo, Path]) -> None:
    _, tmp_path = repo_setup
    ledger = GitLedger(tmp_path)
    ledger.create_run_branch("run-002")

    (tmp_path / "strategies" / "toy.py").write_text(
        "def generate_signals(p, c):\n    return p.tail(2)\n"
    )
    (tmp_path / "configs" / "toy.yaml").write_text("universe: [IWM]\n")

    hexsha = ledger.commit_strategy("toy", "hexsha test")
    assert len(hexsha) == 40
    assert all(c in "0123456789abcdef" for c in hexsha)


def test_rollback_strategy(repo_setup: tuple[git.Repo, Path]) -> None:
    _, tmp_path = repo_setup
    ledger = GitLedger(tmp_path)

    original_content = (tmp_path / "strategies" / "toy.py").read_text()
    (tmp_path / "strategies" / "toy.py").write_text("# modified\n")
    assert (tmp_path / "strategies" / "toy.py").read_text() != original_content

    ledger.rollback_strategy("toy")
    assert (tmp_path / "strategies" / "toy.py").read_text() == original_content


def test_ensure_clean_raises_on_dirty(repo_setup: tuple[git.Repo, Path]) -> None:
    _, tmp_path = repo_setup
    ledger = GitLedger(tmp_path)

    (tmp_path / "strategies" / "toy.py").write_text("# dirty change\n")

    with pytest.raises(ValueError, match="uncommitted changes"):
        ledger.ensure_clean("toy")


def test_ensure_clean_passes_on_clean(repo_setup: tuple[git.Repo, Path]) -> None:
    _, tmp_path = repo_setup
    ledger = GitLedger(tmp_path)
    # Should not raise
    ledger.ensure_clean("toy")


def test_reset_to_main(repo_setup: tuple[git.Repo, Path]) -> None:
    _repo, tmp_path = repo_setup
    ledger = GitLedger(tmp_path)

    # Move to a run branch and modify the files
    ledger.create_run_branch("run-xyz")
    (tmp_path / "strategies" / "toy.py").write_text("# candidate changes\n")
    (tmp_path / "configs" / "toy.yaml").write_text("universe: [FAIL]\n")
    ledger.commit_strategy("toy", "modified strategy")

    # Verify modified state
    assert "# candidate changes" in (tmp_path / "strategies" / "toy.py").read_text()

    # Reset
    ledger.reset_to_main("toy")

    # Verify we are back on main and files are restored
    assert ledger.current_branch == "main"
    assert "generate_signals" in (tmp_path / "strategies" / "toy.py").read_text()
    assert "universe: [SPY]" in (tmp_path / "configs" / "toy.yaml").read_text()
