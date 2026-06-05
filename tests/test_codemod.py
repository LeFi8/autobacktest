"""Tests for the pandas codemod repair module."""

from autobacktest.strategy.codemod import repair_pandas_code, repair_strategy_code


# ---------------------------------------------------------------------------
# 1. groupby axis= removal
# ---------------------------------------------------------------------------
def test_groupby_axis_removed():
    code = "df.groupby(['x'], axis=1).sum()"
    result, fixes = repair_pandas_code(code)
    assert "axis" not in result
    assert fixes


# ---------------------------------------------------------------------------
# 2. fillna(method='ffill') → .ffill()
# ---------------------------------------------------------------------------
def test_fillna_ffill():
    code = "df.fillna(method='ffill')"
    result, fixes = repair_pandas_code(code)
    assert ".ffill()" in result
    assert "fillna" not in result
    assert fixes


# ---------------------------------------------------------------------------
# 3. fillna(method='bfill') → .bfill()
# ---------------------------------------------------------------------------
def test_fillna_bfill():
    code = "df.fillna(method='bfill')"
    result, fixes = repair_pandas_code(code)
    assert ".bfill()" in result
    assert "fillna" not in result
    assert fixes


# ---------------------------------------------------------------------------
# 4. fillna(method='pad') → .ffill()
# ---------------------------------------------------------------------------
def test_fillna_pad():
    code = "df.fillna(method='pad')"
    result, fixes = repair_pandas_code(code)
    assert ".ffill()" in result
    assert "fillna" not in result
    assert fixes


# ---------------------------------------------------------------------------
# 5. .mean(level=0) → .groupby(level=0).mean()
# ---------------------------------------------------------------------------
def test_mean_level():
    code = "df.mean(level=0)"
    result, fixes = repair_pandas_code(code)
    assert "groupby(level=0)" in result
    assert ".mean()" in result
    assert fixes


# ---------------------------------------------------------------------------
# 6. .append() is NOT transformed (risk of false-positive on list/str .append)
# ---------------------------------------------------------------------------
def test_append_not_transformed():
    code = "my_list.append(item)"
    result, fixes = repair_pandas_code(code)
    assert result == code, ".append() on non-DataFrame objects must not be rewritten"
    assert fixes == []


# ---------------------------------------------------------------------------
# 7. .resample('M') → .resample('ME')
# ---------------------------------------------------------------------------
def test_freq_alias_resample():
    code = "df.resample('M').mean()"
    result, fixes = repair_pandas_code(code)
    assert "resample('ME')" in result
    assert "'M'" not in result
    assert fixes


# ---------------------------------------------------------------------------
# 8. pd.date_range(..., freq='Q') → freq='QE'
# ---------------------------------------------------------------------------
def test_freq_alias_date_range():
    code = "pd.date_range('2020-01-01', '2021-01-01', freq='Q')"
    result, fixes = repair_pandas_code(code)
    assert "freq='QE'" in result
    assert "freq='Q'" not in result
    assert fixes


# ---------------------------------------------------------------------------
# 9. pd.Grouper(freq='H') → freq='h' (Grouper is a DatetimeIndex context)
# ---------------------------------------------------------------------------
def test_freq_alias_kwarg():
    code = "pd.Grouper(freq='H')"
    result, fixes = repair_pandas_code(code)
    assert "freq='h'" in result
    assert "freq='H'" not in result
    assert fixes


# ---------------------------------------------------------------------------
# 9b. Generic freq= kwarg on an UNKNOWN function must NOT be remapped
#     (only known datetime/period freq functions are touched)
# ---------------------------------------------------------------------------
def test_freq_alias_unknown_func_not_remapped():
    code = "some_func(freq='M')"
    result, fixes = repair_pandas_code(code)
    assert result == code
    assert fixes == []


# ---------------------------------------------------------------------------
# 10. No-op: clean code must return exact same string (critical)
# ---------------------------------------------------------------------------
def test_no_op_clean_code():
    code = "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})\nresult = df.ffill()\n"
    result, fixes = repair_pandas_code(code)
    assert result == code, "Clean code must be returned unchanged (no reformatting)"
    assert fixes == []


# ---------------------------------------------------------------------------
# 11. String 'M' as ticker/column — NOT remapped
# ---------------------------------------------------------------------------
def test_freq_alias_not_remapped_outside_context():
    code = "df['M'] = 1"
    result, fixes = repair_pandas_code(code)
    # The string 'M' in a subscript context should not be remapped
    assert result == code
    assert fixes == []


# ---------------------------------------------------------------------------
# 12. Syntax error passthrough
# ---------------------------------------------------------------------------
def test_syntax_error_passthrough():
    code = "def broken(:\n    pass"
    result, fixes = repair_pandas_code(code)
    assert result == code
    assert fixes == []


# ---------------------------------------------------------------------------
# Period contexts: 'M'/'Q'/'Y' are CORRECT and must NOT be remapped to 'ME'.
# This is the regression that broke real runs (to_period('ME') is invalid).
# ---------------------------------------------------------------------------
def test_to_period_month_not_remapped():
    code = "idx.to_period('M')"
    result, fixes = repair_pandas_code(code)
    assert result == code, "to_period('M') is valid for Period and must be left alone"
    assert fixes == []


def test_period_range_month_not_remapped():
    code = "pd.period_range('2020-01', periods=3, freq='M')"
    result, fixes = repair_pandas_code(code)
    assert result == code, "period_range freq='M' is valid for Period and must be left alone"
    assert fixes == []


# ---------------------------------------------------------------------------
# Period contexts: a model over-applying 'ME' gets reverse-repaired to 'M'.
# ---------------------------------------------------------------------------
def test_to_period_me_reverse_remapped():
    code = "idx.to_period('ME')"
    result, fixes = repair_pandas_code(code)
    assert "to_period('M')" in result
    assert "'ME'" not in result
    assert fixes


def test_period_range_me_reverse_remapped():
    code = "pd.period_range('2020-01', periods=3, freq='ME')"
    result, fixes = repair_pandas_code(code)
    assert "freq='M'" in result
    assert "freq='ME'" not in result
    assert fixes


# ---------------------------------------------------------------------------
# asfreq is ambiguous (datetime wants 'ME', period wants 'M') — never touched.
# ---------------------------------------------------------------------------
def test_asfreq_not_touched():
    code = "s.asfreq('M')"
    result, fixes = repair_pandas_code(code)
    assert result == code
    assert fixes == []


# ---------------------------------------------------------------------------
# MissingImportInjector tests
# ---------------------------------------------------------------------------


def test_inject_missing_any_import():
    """dict[str, Any] in signature without typing import → import injected."""
    code = """
def generate_signals(prices: 'pd.DataFrame', config: 'dict[str, Any]') -> 'pd.DataFrame':
    return pd.DataFrame()
"""
    result, fixes = repair_strategy_code(code)
    assert "from typing import Any" in result
    assert fixes


def test_inject_missing_any_import_annotation():
    """Any used in a type annotation without typing import → injected."""
    code = """
import pandas as pd

def generate_signals(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame()
"""
    result, fixes = repair_strategy_code(code)
    assert "from typing import Any" in result
    assert fixes


def test_skip_if_any_imported():
    """from typing import Any already present → no injection."""
    code = """
from typing import Any
import pandas as pd

def generate_signals(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame()
"""
    _, fixes = repair_strategy_code(code)
    assert fixes == []  # import already present, no pandas issues either


def test_skip_if_typing_module_imported():
    """import typing (module-level) → no injection needed."""
    code = """
import typing

def generate_signals(prices: pd.DataFrame, config: dict[str, typing.Any]) -> pd.DataFrame:
    return pd.DataFrame()
"""
    _, fixes = repair_strategy_code(code)
    assert fixes == []


def test_skip_if_no_type_hints():
    """Bare annotations without Any → no injection."""
    code = """
def generate_signals(prices, config):
    return pd.DataFrame()
"""
    _, fixes = repair_strategy_code(code)
    assert fixes == []


# ---------------------------------------------------------------------------
# WeightRenormalizer tests
# ---------------------------------------------------------------------------


def test_inject_weight_renormalization():
    """return w without renormalization → clip + div injected."""
    code = """
import pandas as pd

def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    w = prices * 0.5
    return w
"""
    result, fixes = repair_strategy_code(code)
    assert "clip(lower=0.0)" in result
    assert ".div(" in result
    assert ".fillna(0.0)" in result
    assert any("weight renormalization" in f for f in fixes)


def test_skip_if_already_renormalized():
    """Code that already clips(lower=0.0) → no injection."""
    code = """
import pandas as pd

def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    w = prices * 0.5
    w = w.clip(lower=0.0)
    return w
"""
    _, fixes = repair_strategy_code(code)
    assert fixes == []  # renormalization already in place, no pandas issues


def test_skip_complex_return():
    """Complex return expression → no injection."""
    code = """
import pandas as pd

def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
"""
    _, fixes = repair_strategy_code(code)
    assert fixes == []


def test_skip_not_generate_signals():
    """Renormalizer only applies to generate_signals, not other functions."""
    code = """
def helper():
    x = 42
    return x

def generate_signals(prices, config):
    return prices
"""
    result, _ = repair_strategy_code(code)
    assert "clip(lower=0.0)" in result


def test_repair_strategy_code_runs_all_passes():
    """repair_strategy_code with multiple issues fixes all of them."""
    code = """
def generate_signals(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    w = prices * 0.5
    return w
"""
    result, fixes = repair_strategy_code(code)
    assert "from typing import Any" in result  # missing import fixed
    assert "clip(lower=0.0)" in result  # renormalization added
    assert len(fixes) >= 2
