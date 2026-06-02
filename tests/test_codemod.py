"""Tests for the pandas codemod repair module."""

import pytest
from autobacktest.strategy.codemod import repair_pandas_code


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
# 6. .append(other) → pd.concat([df, other])
# ---------------------------------------------------------------------------
def test_append_to_concat():
    code = "df.append(other)"
    result, fixes = repair_pandas_code(code)
    assert "pd.concat" in result
    assert "append" not in result
    assert fixes


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
# 9. some_func(freq='H') → freq='h'
# ---------------------------------------------------------------------------
def test_freq_alias_kwarg():
    code = "some_func(freq='H')"
    result, fixes = repair_pandas_code(code)
    assert "freq='h'" in result
    assert "freq='H'" not in result
    assert fixes


# ---------------------------------------------------------------------------
# 10. No-op: clean code must return exact same string (critical)
# ---------------------------------------------------------------------------
def test_no_op_clean_code():
    code = (
        "import pandas as pd\n"
        "df = pd.DataFrame({'a': [1, 2, 3]})\n"
        "result = df.ffill()\n"
    )
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
