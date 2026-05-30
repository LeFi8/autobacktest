"""Unit tests verifying the 15 code review fixes and sandbox upgrades."""

import time
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from autobacktest.evaluator.report import EvaluationReport, WindowReport
from autobacktest.gate import accept as gate_accept
from autobacktest.strategy.config_schema import StrategyConfig
from autobacktest.strategy.contract import validate_output, validate_signature
from autobacktest.strategy.validator import ValidationError, _check_ast, preflight


@pytest.fixture
def mock_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Fixture creating temp directories for strategy and config files."""
    strat_dir = tmp_path / "strategies"
    conf_dir = tmp_path / "configs"
    strat_dir.mkdir()
    conf_dir.mkdir()
    return strat_dir, conf_dir


def test_path_traversal_rejection(mock_dirs: tuple[Path, Path]) -> None:
    """Verifies that strategy names with traversal elements are rejected."""
    strat_dir, conf_dir = mock_dirs
    res = preflight("../simple", strat_dir, conf_dir)
    assert not res.passed
    assert res.error_code == ValidationError.IMPORT_FAILED
    assert "path traversal" in res.detail


def test_forbidden_names_and_submodules_gaps() -> None:
    """Verify AST blocks escapes (pd.read_table, np.loadtxt, npyio, etc.)."""
    codes = [
        "import pandas as pd\ndef generate_signals(p, c):\n    pd.read_table('file.txt')\n",
        "import numpy as np\ndef generate_signals(p, c):\n    np.loadtxt('file.txt')\n",
        "import numpy as np\ndef generate_signals(p, c):\n    np.genfromtxt('file.txt')\n",
        "import pandas as pd\ndef generate_signals(p, c):\n    pd.io.common.get_handle('file.txt')\n",
        "import pandas as pd\ndef generate_signals(p, c):\n    pd.ExcelFile('file.xlsx')\n",
    ]
    for code in codes:
        res = _check_ast(code)
        assert not res.passed
        assert res.error_code == ValidationError.AST_BLOCKED_IMPORT


def test_import_from_alias_bypass() -> None:
    """Verifies import alias and from-imports are inspected for forbidden names."""
    codes = [
        "from pandas import read_csv as r\ndef generate_signals(p, c):\n    r('f.csv')\n",
        "import pandas as eval\ndef generate_signals(p, c):\n    eval('1+1')\n",
        "from pandas import read_table as rt\ndef generate_signals(p, c):\n    pass\n",
    ]
    for code in codes:
        res = _check_ast(code)
        assert not res.passed
        assert res.error_code == ValidationError.AST_BLOCKED_IMPORT


def test_lookahead_shape_mismatch_fails_sniff(
    mock_dirs: tuple[Path, Path],
) -> None:
    """Verifies lookahead sniffer fails when future data changes shape."""
    strat_dir, conf_dir = mock_dirs
    strat_file = strat_dir / "lookahead_shape.py"
    strat_file.write_text(
        """
import pandas as pd
def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    # Leakage: return different shape when future prices are appended
    if len(prices) > 756:
        return pd.DataFrame(1.0, index=prices.index, columns=["QQQ"])
    return pd.DataFrame(0.5, index=prices.index, columns=prices.columns)
""",
        encoding="utf-8",
    )
    conf_file = conf_dir / "lookahead_shape.yaml"
    conf_file.write_text("universe: [SPY]\n", encoding="utf-8")

    res = preflight("lookahead_shape", strat_dir, conf_dir)
    assert not res.passed
    assert res.error_code == ValidationError.LOOKAHEAD_DETECTED
    assert "Lookahead bias sniff test failed" in res.detail


def test_execution_timeout_sandbox(mock_dirs: tuple[Path, Path]) -> None:
    """Verifies that infinite loops inside strategy trigger timeouts."""
    strat_dir, conf_dir = mock_dirs
    strat_file = strat_dir / "timeout_strat.py"
    strat_file.write_text(
        """
import pandas as pd
def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    while True:
        pass  # Infinite loop to trigger timeout
""",
        encoding="utf-8",
    )
    conf_file = conf_dir / "timeout_strat.yaml"
    conf_file.write_text("universe: [SPY]\n", encoding="utf-8")

    start_time = time.time()
    res = preflight("timeout_strat", strat_dir, conf_dir)
    elapsed = time.time() - start_time

    assert not res.passed
    assert res.error_code == ValidationError.SMOKE_TEST_FAILED
    assert "timed out" in res.detail
    # Must have timed out well before the harness threshold
    assert elapsed < 16.0


def test_dynamic_config_collision_guard() -> None:
    """Verifies dynamic collision guard checks root-level extras."""
    # 1. Standard field collision
    with pytest.raises(PydanticValidationError) as exc_info:
        StrategyConfig.model_validate({"universe": ["SPY"], "params": {"universe": ["QQQ"]}})
    assert "collide with top-level schema fields" in str(exc_info.value)

    # 2. Extra field collision (dynamic guard - Finding 11)
    with pytest.raises(PydanticValidationError) as exc_info:
        StrategyConfig.model_validate(
            {
                "universe": ["SPY"],
                "custom_factor": 10.0,
                "params": {"custom_factor": 20.0},
            }
        )
    assert "collide with top-level schema fields" in str(exc_info.value)


class MockModule:
    pass


def test_signature_var_positional_rejection() -> None:
    """Verify signature checks block *args as second argument."""
    mod = MockModule()

    # 1. Positional *args as the second argument must be rejected
    def generate_signals_bad(prices, *args):
        pass

    mod.generate_signals = generate_signals_bad
    ok, err = validate_signature(mod)
    assert not ok
    assert "Second parameter must be positional" in err

    # 2. Positional *args as the third argument is allowed if defaults exist
    def generate_signals_ok(prices, config, *args):
        pass

    mod.generate_signals = generate_signals_ok
    ok, err = validate_signature(mod)
    assert ok


def test_output_index_validation() -> None:
    """Verifies that weights are rejected if they contain outside dates."""
    dates = pd.date_range("2023-01-01", periods=3)
    prices_index = dates

    # Valid index subset
    weights_ok = pd.DataFrame({"SPY": [0.5, 0.5]}, index=dates[:2])
    ok, err = validate_output(weights_ok, ["SPY"], expected_index=prices_index)
    assert ok

    # Invalid index (date not in price history)
    bad_date = pd.Timestamp("2024-01-01")
    weights_bad = pd.DataFrame({"SPY": [0.5]}, index=[bad_date])
    ok, err = validate_output(weights_bad, ["SPY"], expected_index=prices_index)
    assert not ok
    assert "contains dates not in the price history" in err


def test_gate_nan_error_message_formatting() -> None:
    """Verifies that clean float breaches do not output NaN messages."""
    window = WindowReport(
        start_date="2023-01-01",
        end_date="2025-12-31",
        annualized_return=0.15,
        annualized_volatility=0.10,
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        max_drawdown=0.20,
        turnover=0.5,
        information_ratio=1.0,
    )
    report = EvaluationReport(
        strategy_name="mock",
        dataset_hash="abc",
        gates_passed={},
        is_accepted=True,
        rejection_reason=None,
        holdout_metrics=window,
        walk_forward_metrics=[window],
        regime_drawdowns={},
        regime_passed=True,
        mc_sharpe_5th=0.5,
        mc_sharpe_50th=1.2,
        mc_sharpe_95th=2.0,
        observed_sharpe=1.5,
        effective_trials=1,
        deflated_sharpe=0.98,
    )

    # 1. Clean float breach
    res = gate_accept(report, baseline=None, dd_limit=0.15)
    assert not res.accepted
    assert "exceeds limit of 0.1500" in res.reason
    assert "NaN" not in res.reason

    # 2. NaN breach
    window.max_drawdown = float("nan")
    res_nan = gate_accept(report, baseline=None, dd_limit=0.15)
    assert not res_nan.accepted
    assert "drawdown is NaN" in res_nan.reason


def test_memory_limit_sandbox(mock_dirs: tuple[Path, Path]) -> None:
    """Verifies that strategy allocating too much memory gets blocked."""
    strat_dir, conf_dir = mock_dirs
    strat_file = strat_dir / "oom_strat.py"
    strat_file.write_text(
        """
import pandas as pd
def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    # Attempt to allocate ~2GB of memory (over 1GB limit)
    # 250,000,000 floats * 8 bytes = 2GB
    x = [0.0] * 250000000
    return pd.DataFrame()
""",
        encoding="utf-8",
    )
    conf_file = conf_dir / "oom_strat.yaml"
    conf_file.write_text("universe: [SPY]\n", encoding="utf-8")

    res = preflight("oom_strat", strat_dir, conf_dir)
    assert not res.passed
    assert res.error_code == ValidationError.SMOKE_TEST_FAILED
    allowed_terms = ("memory limit", "timed out", "exception", "failed")
    assert any(term in res.detail for term in allowed_terms)


def test_timeout_sandbox_non_main_thread() -> None:
    """Verifies timeout_sandbox runs cleanly on a background thread without crashing."""
    import threading

    from autobacktest.strategy.validator import timeout_sandbox

    errors = []

    def run_in_thread():
        try:
            with timeout_sandbox(seconds=2):
                pass
        except Exception as e:
            errors.append(e)

    t = threading.Thread(target=run_in_thread)
    t.start()
    t.join()
    assert len(errors) == 0


def test_validate_output_duplicate_columns() -> None:
    """Verifies that weights with duplicate columns are rejected."""
    dates = pd.date_range("2023-01-01", periods=3)
    weights = pd.DataFrame([[0.5, 0.5]], index=[dates[0]], columns=["SPY", "SPY"])
    ok, err = validate_output(weights, ["SPY"])
    assert not ok
    assert "duplicate columns" in err


def test_validate_output_non_numeric() -> None:
    """Verifies that weights with non-numeric types are rejected."""
    dates = pd.date_range("2023-01-01", periods=3)
    weights = pd.DataFrame([["0.5", 0.5]], index=[dates[0]], columns=["SPY", "QQQ"])
    ok, err = validate_output(weights, ["SPY", "QQQ"])
    assert not ok
    assert "must be numeric" in err


def test_evaluate_strategy_too_short_period() -> None:
    """Verifies that evaluate_strategy raises ValueError if the
    backtest period is too short.
    """
    from autobacktest.evaluator.evaluate import evaluate_strategy

    def dummy_generate_signals(prices, _config):
        return pd.DataFrame(0.5, index=prices.index, columns=prices.columns)

    config = {"universe": ["SPY"], "benchmark": "SPY"}
    # Passing a very short period will fail the 3-year holdout partition
    with pytest.raises(ValueError) as exc:
        evaluate_strategy(
            "dummy",
            dummy_generate_signals,
            config,
            start_date="2023-01-01",
            end_date="2023-01-15",
        )
    assert "In-sample or holdout period is empty" in str(exc.value)


def test_gate_dsr_none_handling() -> None:
    """Verifies that gate.accept handles None deflated_sharpe gracefully.

    Since DSR is no longer a hard gate, a None deflated_sharpe should be
    accepted as long as all other gates pass.
    """
    window = WindowReport(
        start_date="2023-01-01",
        end_date="2025-12-31",
        annualized_return=0.15,
        annualized_volatility=0.10,
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        max_drawdown=0.05,
        turnover=0.5,
        information_ratio=1.0,
    )
    report = EvaluationReport(
        strategy_name="mock",
        dataset_hash="abc",
        gates_passed={},
        is_accepted=True,
        rejection_reason=None,
        holdout_metrics=window,
        walk_forward_metrics=[window],
        regime_drawdowns={},
        regime_passed=True,
        mc_sharpe_5th=0.5,
        mc_sharpe_50th=1.2,
        mc_sharpe_95th=2.0,
        observed_sharpe=1.5,
        effective_trials=1,
        deflated_sharpe=None,  # type: ignore
    )

    res = gate_accept(report, baseline=None)
    assert res.accepted


def test_agent_edit_response_lessons_text_optional() -> None:
    """Verifies that AgentEditResponse validates successfully and leaves
    lessons_text unset when missing.
    """
    from autobacktest.llm.litellm_provider import AgentEditResponse

    json_data = """{
        "strategy_code": "def generate_signals(prices, config): return prices",
        "config_yaml": "universe: [SPY]",
        "reasoning": "Simple identity strategy."
    }"""
    parsed = AgentEditResponse.model_validate_json(json_data)
    assert parsed.lessons_text is None
    assert parsed.strategy_code == "def generate_signals(prices, config): return prices"


def test_system_prompt_leverage_constraint() -> None:
    """Verifies that the restored leverage constraint is in the SYSTEM_PROMPT."""
    from autobacktest.llm.prompts import SYSTEM_PROMPT

    assert "summing to at most 1.0 (sum <= 1.0) for every rebalance day" in SYSTEM_PROMPT


def test_db_schema_migration_and_custom_sorting(tmp_path: Path) -> None:
    """Verifies that old SQLite databases without target_metric/value
    are migrated and sorted properly.
    """
    import sqlite3

    import pandas as pd

    from autobacktest.ledger.store import LedgerStore

    db_file = tmp_path / "old_ledger.db"

    # 1. Create a mock old attempts table schema lacking the new columns
    conn = sqlite3.connect(str(db_file))
    conn.execute("""
        CREATE TABLE attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            iteration INTEGER NOT NULL,
            strategy_name TEXT NOT NULL,
            dataset_hash TEXT NOT NULL,
            config_yaml TEXT NOT NULL,
            observed_sharpe REAL NOT NULL,
            deflated_sharpe REAL NOT NULL,
            holdout_max_drawdown REAL NOT NULL,
            holdout_turnover REAL NOT NULL,
            regime_passed INTEGER NOT NULL,
            accepted INTEGER NOT NULL,
            committed INTEGER NOT NULL,
            commit_sha TEXT,
            rejection_reason TEXT,
            report_json TEXT NOT NULL,
            returns_blob BLOB NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    # Insert an old attempt
    conn.execute(
        """
        INSERT INTO attempts (
            run_id, iteration, strategy_name, dataset_hash, config_yaml,
            observed_sharpe, deflated_sharpe, holdout_max_drawdown, holdout_turnover,
            regime_passed, accepted, committed, report_json, returns_blob, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            "run-1",
            1,
            "haa",
            "abc",
            "universe: [SPY]",
            1.5,
            1.4,
            0.10,
            0.5,
            1,
            1,
            1,
            "{}",
            b"blob",
            "2026-05-28 10:00:00",
        ),
    )
    conn.commit()
    conn.close()

    # 2. Instantiate LedgerStore to trigger migration
    store = LedgerStore(db_file)
    try:
        # Verify columns are added and backfilled
        cursor = store._conn.cursor()
        cursor.execute("PRAGMA table_info(attempts)")
        columns = [row[1] for row in cursor.fetchall()]
        assert "target_metric" in columns
        assert "target_metric_value" in columns

        # Retrieve migrated leaderboard
        rows = store.leaderboard()
        assert len(rows) == 1
        assert rows[0]["target_metric"] == "sharpe"
        assert rows[0]["target_metric_value"] == 1.5

        # 3. Add a new attempt with a different target_metric and value
        store.record_attempt(
            run_id="run-1",
            iteration=2,
            strategy_name="haa",
            dataset_hash="abc",
            config_yaml="universe: [SPY]",
            observed_sharpe=0.8,
            deflated_sharpe=0.7,
            target_metric="sortino",
            target_metric_value=2.5,  # Higher value than the Sharpe attempt
            holdout_max_drawdown=0.08,
            holdout_turnover=0.4,
            regime_passed=True,
            accepted=True,
            committed=True,
            commit_sha="sha2",
            rejection_reason=None,
            report_json="{}",
            holdout_returns=pd.Series([0.01, 0.02]),
        )

        # Leaderboard should return the Sortino attempt as best
        rows2 = store.leaderboard()
        assert len(rows2) == 1
        assert rows2[0]["target_metric"] == "sortino"
        assert rows2[0]["target_metric_value"] == 2.5
    finally:
        store.close()


def test_git_ledger_upgrades(tmp_path: Path) -> None:
    """Verifies subdirectory safety, dynamic baseline branch checkout, and
    lessons.md decoupled rollback in GitLedger.
    """
    import git

    from autobacktest.ledger.git_ops import GitLedger

    # 1. Create a dummy git repo
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    repo = git.Repo.init(repo_dir)

    # Configure user
    repo.config_writer().set_value("user", "name", "Test").release()
    repo.config_writer().set_value("user", "email", "test@test.com").release()

    strat_dir = repo_dir / "strategies"
    cfg_dir = repo_dir / "configs"
    strat_dir.mkdir()
    cfg_dir.mkdir()

    strat_file = strat_dir / "test_strat.py"
    cfg_file = cfg_dir / "test_strat.yaml"
    lessons_file = repo_dir / "lessons.md"

    strat_file.write_text("code v1", encoding="utf-8")
    cfg_file.write_text("config v1", encoding="utf-8")
    lessons_file.write_text("lessons v1", encoding="utf-8")

    repo.index.add([str(strat_file), str(cfg_file), str(lessons_file)])
    repo.index.commit("Initial baseline commit")

    # Get the automatically created baseline branch name (master or main)
    baseline_branch_name = repo.active_branch.name

    # Verify CWD nested search
    nested_dir = strat_dir / "nested"
    nested_dir.mkdir()

    # Instantiate from nested directory
    git_ledger = GitLedger(nested_dir)
    assert git_ledger.repo_root.resolve() == repo_dir.resolve()

    # 2. Modify files and test rollback_strategy
    strat_file.write_text("code v2", encoding="utf-8")
    lessons_file.write_text("lessons v2", encoding="utf-8")

    git_ledger.rollback_strategy("test_strat")

    # Strategy files must be reverted to baseline, lessons.md must stay
    # modified (decoupled)
    assert strat_file.read_text(encoding="utf-8") == "code v1"
    assert lessons_file.read_text(encoding="utf-8") == "lessons v2"

    # 3. Create run branch and test reset_to_main (recovering baseline dynamically)
    run_branch = repo.create_head("autobacktest/run-abc")
    run_branch.checkout()

    strat_file.write_text("code v3", encoding="utf-8")
    # Commit changes on run branch
    repo.index.add([str(strat_file)])
    repo.index.commit("Iter 1 commit")

    # Reset strategy back to baseline
    git_ledger.reset_to_main("test_strat")

    # Baseline files must be restored, and branch must be baseline_branch_name
    assert repo.active_branch.name == baseline_branch_name
    assert strat_file.read_text(encoding="utf-8") == "code v1"


def test_orchestrator_lessons_persistence(tmp_path: Path) -> None:
    """Verifies that lessons.md is successfully updated and persisted on disk
    and in memory during rejected/exception orchestrator loops.
    """
    from unittest.mock import patch

    import git

    from autobacktest.gate import TargetMetric
    from autobacktest.llm.base import AgentContext, AgentEdit, LLMProvider
    from autobacktest.orchestrator import run_optimization

    # 1. Create a dummy git repo with project structure
    strat_dir = tmp_path / "strategies"
    cfg_dir = tmp_path / "configs"
    run_dir = tmp_path / "runs"
    strat_dir.mkdir()
    cfg_dir.mkdir()
    run_dir.mkdir()

    from tests.test_orchestrator_e2e import (
        BASELINE_STRATEGY,
        PROGRAM_MD,
        STRATEGY_CONFIG,
        _make_fake_provider,
        _make_synthetic_prices,
    )

    (strat_dir / "toy.py").write_text(BASELINE_STRATEGY, encoding="utf-8")
    (cfg_dir / "toy.yaml").write_text(STRATEGY_CONFIG, encoding="utf-8")
    (tmp_path / "program.md").write_text(PROGRAM_MD, encoding="utf-8")

    # Initial lessons.md file
    lessons_file = tmp_path / "lessons.md"
    lessons_file.write_text("# Initial Lessons\n", encoding="utf-8")

    repo = git.Repo.init(tmp_path)
    repo.config_writer().set_value("user", "name", "Test User").release()
    repo.config_writer().set_value("user", "email", "test@test.com").release()
    repo.index.add(["strategies/toy.py", "configs/toy.yaml", "lessons.md"])
    repo.index.commit("initial: baseline toy strategy")

    # 2. Mock LLM Provider
    # Iteration 1: returns a validation-failing edit.
    # Iteration 2: returns a valid strategy that gets evaluated.
    called_contexts = []

    class ScriptedLLMProvider(LLMProvider):
        @property
        def provider_name(self) -> str:
            return "scripted"

        def generate_edit(self, context: AgentContext) -> AgentEdit:
            called_contexts.append(context)
            if context.iteration == 1:
                return AgentEdit(
                    strategy_code="import os\n# bad code\n",  # fails validation
                    config_yaml=STRATEGY_CONFIG,
                    reasoning="Bad edit with os import.",
                    raw_response="{}",
                    lessons_text="# Lessons: validation failed because of os import.",
                )
            else:
                return AgentEdit(
                    strategy_code=BASELINE_STRATEGY,  # valid but identical -> rejected
                    config_yaml=STRATEGY_CONFIG,
                    reasoning="No changes reasoning.",
                    raw_response="{}",
                    lessons_text="# Lessons: rejected because no improvement.",
                )

    provider = ScriptedLLMProvider()
    synthetic_prices = _make_synthetic_prices()
    fake_instance = _make_fake_provider(synthetic_prices)

    with patch(
        "autobacktest.evaluator.evaluate.CachedDataProvider",
        return_value=fake_instance,
    ):
        run_optimization(
            program_path=tmp_path / "program.md",
            strategy_name="toy",
            iterations=2,
            provider=provider,
            run_dir=run_dir,
            strategies_dir=strat_dir,
            configs_dir=cfg_dir,
            target_metric=TargetMetric.SHARPE,
            repo_path=tmp_path,
            start_date="2013-01-01",
            end_date="2025-01-01",
        )

    # Verify that:
    # 1. At the start of Iteration 2, the context received updated lessons!
    # Iteration 1 produces 1 call (validation failure — no diversity check reached).
    # Iteration 2 produces 1 initial call + up to MAX_DIVERSITY_RETRIES retries (because
    # STRATEGY_CONFIG is identical to the baseline → diversity rejected each time).
    from autobacktest.orchestrator import MAX_DIVERSITY_RETRIES

    assert len(called_contexts) == 1 + (1 + MAX_DIVERSITY_RETRIES)
    assert called_contexts[0].lessons_text == "# Initial Lessons\n"
    assert called_contexts[1].lessons_text == "# Lessons: validation failed because of os import."

    # 2. After the run, the final lessons.md on disk is preserved and updated!
    assert lessons_file.read_text(encoding="utf-8") == "# Lessons: rejected because no improvement."


def test_cli_reset_safe_abort(tmp_path: Path) -> None:
    """Verifies that the reset CLI command immediately aborts with exit code 1
    if git checkout/reset fails, leaving files untouched.
    """
    from typer.testing import CliRunner

    from autobacktest.cli import app

    runner = CliRunner()

    strat_dir = tmp_path / "strategies"
    cfg_dir = tmp_path / "configs"
    run_dir = tmp_path / "runs"
    strat_dir.mkdir()
    cfg_dir.mkdir()
    run_dir.mkdir()

    # Create lessons.md with uncommitted content
    lessons_file = tmp_path / "lessons.md"
    lessons_file.write_text("# Saved lessons\n", encoding="utf-8")

    # Patch GitLedger to raise a GitCommandError or any Exception during reset_to_main
    from unittest.mock import patch

    with patch("autobacktest.ledger.git_ops.GitLedger") as mock_git_ledger:
        ledger_instance = mock_git_ledger.return_value
        ledger_instance.repo_root = tmp_path
        # Force an exception during git reset
        ledger_instance.reset_to_main.side_effect = RuntimeError("Dirty working tree conflict")

        def mock_path(*args):
            if not args:
                return tmp_path
            if args[0] == "lessons.md":
                return tmp_path / "lessons.md"
            return Path(*args)

        with patch("autobacktest.cli.Path", side_effect=mock_path):
            result = runner.invoke(app, ["reset", "--strategy", "toy", "--run-dir", str(run_dir)])

        # Assert safe abort occurred
        assert result.exit_code == 1
        assert "Abort: Reset could not be completed safely." in result.output

        # Verify run directory and lessons.md were NOT wiped or deleted!
        assert run_dir.exists()
        assert lessons_file.read_text(encoding="utf-8") == "# Saved lessons\n"
