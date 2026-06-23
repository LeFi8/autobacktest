"""Integration tests for Optuna winner re-validation (critical bug fix).

The optimized config must be RE-EVALUATED end-to-end and RE-GATED (select +
confirm) before it may replace the winner. If it fails the holdout ``confirm``
gate or does not improve the target metric, the original winner must be kept
unchanged — otherwise overfit, never-OOS-validated parameters get committed and
the ledger records metrics that do not match the committed config.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from autobacktest import orchestrator
from autobacktest.gate import TargetMetric
from autobacktest.orchestrator import _OptimizationState


@dataclass
class _FakeEdit:
    strategy_code: str = "def generate_signals(prices, config):\n    return prices\n"
    config_yaml: str = "momentum_lookback: 6\n"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    reasoning: str = "orig"


class _FakeCfgObj:
    def model_dump(self) -> dict[str, Any]:
        return {"universe": ["HIGH", "LOW"], "momentum_lookback": 6}


def _make_state(tmp_path: Any) -> _OptimizationState:
    state = object.__new__(_OptimizationState)
    sdir = tmp_path / "strategies"
    cdir = tmp_path / "configs"
    sdir.mkdir(exist_ok=True)
    cdir.mkdir(exist_ok=True)
    state.strategy_name = "toy"
    state.strategies_dir = sdir
    state.configs_dir = cdir
    state.start_date = "2013-01-01"
    state.end_date = "2025-01-01"
    state._eval_cache = None
    state.incumbent = None
    state.ledger = None
    state.target_metric = TargetMetric.SHARPE
    return state


def _patch_common(
    monkeypatch: pytest.MonkeyPatch,
    *,
    opt_config: dict[str, Any],
    improved: bool,
    select_ok: bool,
    confirm_ok: bool,
    opt_sharpe: float,
) -> tuple[Any, pd.Series]:
    """Patch the heavy/external collaborators of ``_optimize_winner_params``.

    Returns the (opt_report, opt_returns) the re-evaluation will yield so tests
    can assert identity (i.e. that the winner adopted the *re-evaluated* report).
    """
    opt_report = SimpleNamespace(observed_sharpe=opt_sharpe)
    opt_returns = pd.Series([0.01, 0.02])
    opt_flat = dict(opt_config)

    def _sel(*_a: Any, **_k: Any) -> SimpleNamespace:
        return SimpleNamespace(accepted=select_ok, reason="", failed_gate=None)

    def _cnf(*_a: Any, **_k: Any) -> SimpleNamespace:
        return SimpleNamespace(accepted=confirm_ok, reason="", failed_gate=None)

    def _reeval(*_a: Any, **_k: Any) -> tuple[Any, pd.Series, dict[str, Any], None]:
        return opt_report, opt_returns, opt_flat, None

    monkeypatch.setattr(orchestrator, "load_signals", lambda _p: None)
    monkeypatch.setattr(orchestrator.StrategyConfig, "from_yaml", classmethod(lambda _cls, _p: _FakeCfgObj()))
    monkeypatch.setattr(orchestrator, "_load_evaluation_data", lambda *_a, **_k: (pd.DataFrame(), None))
    monkeypatch.setattr(orchestrator, "optimize_numeric_params", lambda *_a, **_k: (opt_config, opt_sharpe, improved))
    monkeypatch.setattr(orchestrator, "_eval_single_candidate", _reeval)
    monkeypatch.setattr(orchestrator, "_deflate", lambda *_a, **_k: None)
    monkeypatch.setattr(orchestrator, "_deflate_holdout", lambda *_a, **_k: None)
    monkeypatch.setattr(orchestrator, "select", _sel)
    monkeypatch.setattr(orchestrator, "confirm", _cnf)
    monkeypatch.setattr(orchestrator, "_get_metric_value", lambda report, _tm: report.observed_sharpe)
    return opt_report, opt_returns


def _winner(orig_report: Any, orig_returns: pd.Series) -> dict[str, Any]:
    return {
        "edit": _FakeEdit(config_yaml="momentum_lookback: 6\n"),
        "config_yaml": "momentum_lookback: 6\n",
        "_config": {"momentum_lookback": 6},
        "_new_config": {"momentum_lookback": 6},
        "_report": orig_report,
        "_returns": orig_returns,
    }


def test_optimized_config_reverted_when_confirm_fails(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Optuna improved in-sample, but the tuned config FAILS the holdout confirm
    gate — the winner must stay on its original config and report."""
    state = _make_state(tmp_path)
    orig_report = SimpleNamespace(observed_sharpe=1.0)
    orig_returns = pd.Series([0.0])
    winner = _winner(orig_report, orig_returns)

    _patch_common(
        monkeypatch,
        opt_config={"momentum_lookback": 24},
        improved=True,
        select_ok=True,
        confirm_ok=False,  # holdout confirm rejects the tuned params
        opt_sharpe=5.0,
    )

    state._optimize_winner_params(1, winner)

    assert winner["optimization_applied"] is False
    assert winner["_report"] is orig_report
    assert winner["_returns"] is orig_returns
    assert winner["edit"].config_yaml == "momentum_lookback: 6\n"


def test_optimized_config_adopts_revalidated_report_when_gates_pass(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the tuned config passes select+confirm and improves the target
    metric, the winner adopts the RE-EVALUATED report/returns (not the stale
    pre-optimization ones), and the gain is measured on the same metric."""
    state = _make_state(tmp_path)
    orig_report = SimpleNamespace(observed_sharpe=1.0)
    orig_returns = pd.Series([0.0])
    winner = _winner(orig_report, orig_returns)

    opt_report, opt_returns = _patch_common(
        monkeypatch,
        opt_config={"momentum_lookback": 24},
        improved=True,
        select_ok=True,
        confirm_ok=True,
        opt_sharpe=2.0,
    )

    state._optimize_winner_params(1, winner)

    assert winner["optimization_applied"] is True
    assert winner["_report"] is opt_report
    assert winner["_returns"] is opt_returns
    assert "24" in winner["edit"].config_yaml
    assert winner["optimization_gain"] == pytest.approx(1.0)  # opt 2.0 - orig 1.0


def test_no_reeval_when_optuna_finds_no_improvement(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """If Optuna does not improve in-sample, no re-evaluation/holdout peek is
    spent and the winner is untouched."""
    state = _make_state(tmp_path)
    orig_report = SimpleNamespace(observed_sharpe=1.0)
    orig_returns = pd.Series([0.0])
    winner = _winner(orig_report, orig_returns)

    called = {"reeval": False}

    def _boom(*_a: Any, **_k: Any) -> Any:
        called["reeval"] = True
        raise AssertionError("re-evaluation must not run when not improved")

    _patch_common(
        monkeypatch,
        opt_config={"momentum_lookback": 24},
        improved=False,
        select_ok=True,
        confirm_ok=True,
        opt_sharpe=0.5,
    )
    monkeypatch.setattr(orchestrator, "_eval_single_candidate", _boom)

    state._optimize_winner_params(1, winner)

    assert called["reeval"] is False
    assert winner["optimization_applied"] is False
    assert winner["_report"] is orig_report
    assert winner["edit"].config_yaml == "momentum_lookback: 6\n"
