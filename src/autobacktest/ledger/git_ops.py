"""Git integration for strategy version control and rollback.

Provides a dedicated git branch per optimization run, atomic two-file
commits for accepted strategies, and clean rollback to HEAD for
rejected candidates.  Uses ``gitpython`` for all operations.
"""

from __future__ import annotations

from pathlib import Path

import git


class GitLedger:
    """Manages git operations for strategy optimization lifecycle.

    Creates an isolated branch per run, stages only the strategy code
    and config YAML files, and supports clean rollback on rejection.
    """

    def __init__(self, repo_path: Path) -> None:
        """Open the git repository and configure strategy/config paths.

        Args:
            repo_path: Path inside the git repository (searches parent dirs).
        """
        self._repo = git.Repo(repo_path, search_parent_directories=True)
        self._strategies_dir = "strategies"
        self._configs_dir = "configs"

    @property
    def repo_root(self) -> Path:
        """Expose the absolute path of the git working tree root."""
        return Path(self._repo.working_tree_dir or "")

    def _resolve_rel_paths(self, strategy_name: str) -> tuple[str, str]:
        """Resolve relative paths for strategy and config based on active layout."""
        new_strat = f"{self._strategies_dir}/{strategy_name}/strategy.py"
        new_cfg = f"{self._strategies_dir}/{strategy_name}/config.yaml"
        if (self.repo_root / new_strat).exists():
            return new_strat, new_cfg
        return f"{self._strategies_dir}/{strategy_name}.py", f"{self._configs_dir}/{strategy_name}.yaml"

    def ensure_clean(self, strategy_name: str) -> None:
        """Raise ValueError if target strategy/config files have uncommitted changes."""
        strat_rel, cfg_rel = self._resolve_rel_paths(strategy_name)
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
        """Stage and commit only strategy and config files.

        Uses ``git add`` to stage only the two targeted files, then
        ``git commit --allow-empty`` to create the commit.  The
        ``--allow-empty`` flag is required because GitPython's stash or
        resume flow may produce a commit with identical content to HEAD
        (in which case ``git commit`` without the flag would abort with
        *"nothing to commit"*).

        Because ``git add`` stages only the two targeted paths, any
        other pre-existing staged changes are still present in the
        index after this call.  In practice this is safe — the
        orchestrator runs in an isolated branch/clean checkout.
        Returns the commit hexsha.
        """
        strat_rel, cfg_rel = self._resolve_rel_paths(strategy_name)
        self._repo.git.add(strat_rel, cfg_rel)
        self._repo.git.commit("-m", message, "--allow-empty")
        return self._repo.head.commit.hexsha

    def rollback_strategy(self, strategy_name: str) -> None:
        """Restore strategies/{name}.py, configs/{name}.yaml to HEAD."""
        strat_rel, cfg_rel = self._resolve_rel_paths(strategy_name)
        self._repo.git.checkout("--", strat_rel, cfg_rel)

    @property
    def current_branch(self) -> str:
        """Return the name of the currently active branch."""
        return str(self._repo.active_branch.name)

    def reset_to_main(self, strategy_name: str | None = None) -> None:
        """Checkout primary branch (main or master) and restore strategy/config files
        to baseline.
        """
        primary_branch = "main"
        if "main" not in self._repo.heads:
            if "master" in self._repo.heads:
                primary_branch = "master"
            else:
                raise ValueError(
                    "Could not determine primary branch: expected 'main' or 'master'. "
                    "Rename your primary branch to 'main' or 'master'."
                )

        if self._repo.active_branch.name != primary_branch:
            self._repo.heads[primary_branch].checkout()

        if strategy_name is not None:
            strat_rel, cfg_rel = self._resolve_rel_paths(strategy_name)
            self._repo.git.checkout("HEAD", "--", strat_rel, cfg_rel)
        else:
            paths = [self._strategies_dir]
            if (self.repo_root / self._configs_dir).exists():
                paths.append(self._configs_dir)
            self._repo.git.checkout("HEAD", "--", *paths)
