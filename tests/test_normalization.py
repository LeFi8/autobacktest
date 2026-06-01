from autobacktest.strategy.normalization import normalize_python_code


def test_normalize_python_code_removes_comments():
    code_with_comments = """
# This is a module-level comment
def foo(x):
    # This is an inline comment
    return x * 2  # Double it
"""
    expected = "def foo(x):\n    return x * 2"
    assert normalize_python_code(code_with_comments) == expected


def test_normalize_python_code_removes_docstrings():
    code_with_docstring = '''
"""Module docstring."""
class Calculator:
    """Class docstring."""
    def add(self, a, b):
        """Method docstring."""
        return a + b
'''
    expected = "class Calculator:\n\n    def add(self, a, b):\n        return a + b"
    assert normalize_python_code(code_with_docstring) == expected


def test_normalize_python_code_standardizes_whitespace():
    unstandardized = """
def  bar(  y  ) :
    
    
    return   y   +   1
"""
    expected = "def bar(y):\n    return y + 1"
    assert normalize_python_code(unstandardized) == expected


def test_normalize_python_code_handles_empty_bodies():
    code_with_only_docstrings = """
def empty_func():
    \"\"\"This function does nothing but has a docstring.\"\"\"
"""
    expected = "def empty_func():\n    pass"
    assert normalize_python_code(code_with_only_docstrings) == expected
