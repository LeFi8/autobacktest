from __future__ import annotations

from pathlib import Path

import git


class GitLedger:
    def __init__(self, repo_path: Path) -> None:
        self._repo = git.Repo(repo_path)
        self._strategies_dir = "strategies"
        self._configs_dir = "configs"

    def ensure_clean(self, strategy_name: str) -> None:
        """Raise ValueError if target strategy/config files have uncommitted changes."""
        strat_rel = f"{self._strategies_dir}/{strategy_name}.py"
        cfg_rel = f"{self._configs_dir}/{strategy_name}.yaml"
        changed = self._repo.git.status("--porcelain", strat_rel, cfg_rel)
        if changed.strip():
            raise ValueError(
                f"Strategy files for '{strategy_name}' have uncommitted changes. "
                f"Commit or stash them before running optimization."
            )

    def create_run_branch(self, run_id: str) -> str:
        """Create and checkout branch 'autobacktest/<run_id>'. Return branch name."""
        branch_name = f"autobacktest/{run_id}"
        new_branch = self._repo.create_head(branch_name)
        new_branch.checkout()
        return branch_name

    def commit_strategy(self, strategy_name: str, message: str) -> str:
        """Stage strategy, config, and lessons files, commit, return hexsha."""
        strat_rel = f"{self._strategies_dir}/{strategy_name}.py"
        cfg_rel = f"{self._configs_dir}/{strategy_name}.yaml"
        files_to_add = [strat_rel, cfg_rel]
        lessons_path = Path(self._repo.working_tree_dir or "") / "lessons.md"
        if lessons_path.exists():
            files_to_add.append("lessons.md")
        self._repo.index.add(files_to_add)
        commit = self._repo.index.commit(message)
        return commit.hexsha

    def rollback_strategy(self, strategy_name: str) -> None:
        """Restore strategies/{name}.py, configs/{name}.yaml, and lessons.md to HEAD."""
        strat_rel = f"{self._strategies_dir}/{strategy_name}.py"
        cfg_rel = f"{self._configs_dir}/{strategy_name}.yaml"
        files_to_checkout = [strat_rel, cfg_rel]
        lessons_path = Path(self._repo.working_tree_dir or "") / "lessons.md"
        if lessons_path.exists():
            files_to_checkout.append("lessons.md")
        self._repo.git.checkout("--", *files_to_checkout)

    @property
    def current_branch(self) -> str:
        """Return the name of the currently active branch."""
        return str(self._repo.active_branch.name)

    def reset_to_main(self, strategy_name: str | None = None) -> None:
        """Checkout main branch and restore strategy/config files to baseline."""
        if self._repo.active_branch.name != "main":
            self._repo.heads.main.checkout()

        if strategy_name is not None:
            strat_rel = f"{self._strategies_dir}/{strategy_name}.py"
            cfg_rel = f"{self._configs_dir}/{strategy_name}.yaml"
            self._repo.git.checkout("HEAD", "--", strat_rel, cfg_rel)
        else:
            self._repo.git.checkout(
                "HEAD", "--", self._strategies_dir, self._configs_dir
            )
