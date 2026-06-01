"""Strategy code normalization utilities.

Provides ``normalize_python_code`` which strips comments, docstrings, and
standardises whitespace to produce a stable, content-addressable string
suitable for eval-cache key computation.  The normaliser parses the source
into an AST, removes docstring nodes, and unparses the tree.
"""

import ast


def normalize_python_code(code: str) -> str:
    """Normalize Python source code by removing comments, docstrings, and standardizing whitespace/indentation."""
    if not code:
        return ""
    try:
        tree = ast.parse(code)
    except Exception:
        # Fallback to stripped original code if syntax is invalid
        return code.strip()

    # Walk the tree and remove docstrings from modules, classes, and functions
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body.pop(0)
            # If the body is now empty, replace with pass statement
            if not node.body:
                node.body.append(ast.Pass())

    try:
        return ast.unparse(tree).strip()
    except Exception:
        return code.strip()
