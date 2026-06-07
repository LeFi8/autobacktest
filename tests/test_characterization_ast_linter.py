import ast
import pytest

from autobacktest.strategy.ast_linter import _check_ast, _check_undefined_names
from autobacktest.strategy.validator import ValidationError


def test_characterization_check_ast_imports() -> None:
    # Whitelisted imports pass
    valid_code = "import pandas as pd\nimport numpy as np\n"
    res = _check_ast(valid_code)
    assert res.passed

    # Forbidden/non-whitelisted imports fail
    invalid_code = "import os\n"
    res = _check_ast(invalid_code)
    assert not res.passed
    assert res.error_code == ValidationError.AST_BLOCKED_IMPORT


def test_characterization_check_ast_dunders() -> None:
    # Forbidden dunder attributes fail
    invalid_code = "x = 'test'.__class__\n"
    res = _check_ast(invalid_code)
    assert not res.passed
    assert res.error_code == ValidationError.AST_BLOCKED_IMPORT


def test_characterization_check_undefined_names() -> None:
    # Undefined name references fail
    invalid_code = """
def generate_signals(prices, config):
    x = undefined_variable
    return prices
"""
    tree = ast.parse(invalid_code)
    res = _check_undefined_names(tree)
    assert res is not None
    assert not res.passed
    assert res.error_code == ValidationError.UNDEFINED_NAME

    # Defined local and module names pass
    valid_code = """
import pandas as pd

MY_CONST = 10

def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    x = MY_CONST
    y = [val * x for val in [1, 2, 3]]
    return prices
"""
    tree = ast.parse(valid_code)
    res = _check_undefined_names(tree)
    assert res is None
