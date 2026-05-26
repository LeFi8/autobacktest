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
        "import pandas as pd\n"
        "def generate_signals(p, c):\n"
        "    pd.read_table('file.txt')\n",
        "import numpy as np\ndef generate_signals(p, c):\n    np.loadtxt('file.txt')\n",
        "import numpy as np\n"
        "def generate_signals(p, c):\n"
        "    np.genfromtxt('file.txt')\n",
        "import pandas as pd\n"
        "def generate_signals(p, c):\n"
        "    pd.io.common.get_handle('file.txt')\n",
        "import pandas as pd\n"
        "def generate_signals(p, c):\n"
        "    pd.ExcelFile('file.xlsx')\n",
    ]
    for code in codes:
        res = _check_ast(code)
        assert not res.passed
        assert res.error_code == ValidationError.AST_BLOCKED_IMPORT


def test_import_from_alias_bypass() -> None:
    """Verifies import alias and from-imports are inspected for forbidden names."""
    codes = [
        "from pandas import read_csv as r\n"
        "def generate_signals(p, c):\n"
        "    r('f.csv')\n",
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
        StrategyConfig.model_validate(
            {"universe": ["SPY"], "params": {"universe": ["QQQ"]}}
        )
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
