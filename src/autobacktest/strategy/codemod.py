"""
AST-based repair module for deprecated pandas API calls.

Transforms pandas 1.x/2.x patterns to pandas 3.x-compatible equivalents.
Note: ast.unparse drops comments from repaired files (accepted tradeoff).
"""

import ast
from typing import Optional


# Deprecated freq aliases → pandas 3.x replacements
_FREQ_ALIAS_MAP: dict[str, str] = {
    "M": "ME",
    "BM": "BME",
    "Q": "QE",
    "A": "YE",
    "Y": "YE",
    "H": "h",
    "T": "min",
    "S": "s",
}

# Function names that take freq as their first positional argument
_FREQ_FIRST_ARG_FUNCS = {"resample", "asfreq", "date_range", "period_range", "to_period", "Grouper"}

# Aggregation functions that accept level= keyword
_LEVEL_AGG_FUNCS = {"mean", "sum", "std", "min", "max"}


def _get_call_name(node: ast.Call) -> Optional[str]:
    """Return the bare function/method name from a Call node, or None."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _remap_freq_string(value: str) -> Optional[str]:
    """Return remapped freq alias, or None if no remapping needed."""
    return _FREQ_ALIAS_MAP.get(value)


class _PandasDeprecationTransformer(ast.NodeTransformer):
    """Rewrites deprecated pandas API patterns in-place on the AST."""

    def __init__(self) -> None:
        self.fixes_applied: list[str] = []

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # Visit children first (bottom-up)
        self.generic_visit(node)

        name = _get_call_name(node)
        if name is None:
            return node

        # ------------------------------------------------------------------ #
        # 1. groupby(axis=...) — drop the axis= keyword                       #
        # ------------------------------------------------------------------ #
        if name == "groupby":
            new_keywords = [kw for kw in node.keywords if kw.arg != "axis"]
            if len(new_keywords) < len(node.keywords):
                node.keywords = new_keywords
                self.fixes_applied.append("Removed deprecated axis= argument from .groupby()")

        # ------------------------------------------------------------------ #
        # 2. .fillna(method='ffill'/'pad') → .ffill()                         #
        #    .fillna(method='bfill'/'backfill') → .bfill()                    #
        # ------------------------------------------------------------------ #
        elif name == "fillna":
            method_value = None
            for kw in node.keywords:
                if kw.arg == "method" and isinstance(kw.value, ast.Constant):
                    method_value = kw.value.value
                    break

            if method_value in ("ffill", "pad"):
                # Replace entire call with expr.ffill()
                new_call = ast.Call(
                    func=ast.Attribute(
                        value=node.func.value,  # type: ignore[attr-defined]
                        attr="ffill",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[],
                )
                ast.copy_location(new_call, node)
                ast.fix_missing_locations(new_call)
                self.fixes_applied.append("Replaced .fillna(method='ffill') with .ffill()")
                return new_call

            elif method_value in ("bfill", "backfill"):
                new_call = ast.Call(
                    func=ast.Attribute(
                        value=node.func.value,  # type: ignore[attr-defined]
                        attr="bfill",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[],
                )
                ast.copy_location(new_call, node)
                ast.fix_missing_locations(new_call)
                self.fixes_applied.append("Replaced .fillna(method='bfill') with .bfill()")
                return new_call

        # ------------------------------------------------------------------ #
        # 3. .mean/sum/std/min/max(level=L) → .groupby(level=L).FUNC()        #
        # ------------------------------------------------------------------ #
        elif name in _LEVEL_AGG_FUNCS:
            level_kw = None
            other_kwargs = []
            for kw in node.keywords:
                if kw.arg == "level":
                    level_kw = kw
                else:
                    other_kwargs.append(kw)

            if level_kw is not None:
                # Build: expr.groupby(level=L).FUNC(**other_kwargs)
                groupby_call = ast.Call(
                    func=ast.Attribute(
                        value=node.func.value,  # type: ignore[attr-defined]
                        attr="groupby",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[level_kw],
                )
                new_call = ast.Call(
                    func=ast.Attribute(
                        value=groupby_call,
                        attr=name,
                        ctx=ast.Load(),
                    ),
                    args=list(node.args),
                    keywords=other_kwargs,
                )
                ast.copy_location(groupby_call, node)
                ast.copy_location(new_call, node)
                ast.fix_missing_locations(new_call)
                self.fixes_applied.append(
                    f"Replaced .{name}(level=) with .groupby(level=).{name}()"
                )
                return new_call

        # ------------------------------------------------------------------ #
        # 4. .append(other) → pd.concat([expr, other])                        #
        # ------------------------------------------------------------------ #
        elif name == "append":
            if len(node.args) == 1 and not node.keywords:
                other = node.args[0]
                expr = node.func.value  # type: ignore[attr-defined]
                concat_call = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="pd", ctx=ast.Load()),
                        attr="concat",
                        ctx=ast.Load(),
                    ),
                    args=[ast.List(elts=[expr, other], ctx=ast.Load())],
                    keywords=[],
                )
                ast.copy_location(concat_call, node)
                ast.fix_missing_locations(concat_call)
                self.fixes_applied.append("Replaced .append() with pd.concat()")
                return concat_call

        # ------------------------------------------------------------------ #
        # 5. Deprecated freq aliases — ONLY inside freq contexts               #
        # ------------------------------------------------------------------ #
        # Check freq= keyword argument
        for kw in node.keywords:
            if kw.arg == "freq" and isinstance(kw.value, ast.Constant):
                old_val = kw.value.value
                new_val = _remap_freq_string(old_val)
                if new_val is not None:
                    kw.value = ast.Constant(value=new_val)
                    ast.fix_missing_locations(kw.value)
                    self.fixes_applied.append(
                        f"Remapped deprecated freq alias '{old_val}' to '{new_val}' in freq context"
                    )

        # Check first positional arg for freq-context functions
        if name in _FREQ_FIRST_ARG_FUNCS and node.args:
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Constant):
                old_val = first_arg.value
                new_val = _remap_freq_string(old_val)
                if new_val is not None:
                    node.args[0] = ast.Constant(value=new_val)
                    ast.fix_missing_locations(node.args[0])
                    self.fixes_applied.append(
                        f"Remapped deprecated freq alias '{old_val}' to '{new_val}' in freq context"
                    )

        return node


def repair_pandas_code(code: str) -> tuple[str, list[str]]:
    """
    Parse *code*, apply pandas deprecation fixes, and return the repaired source.

    Returns:
        (repaired_code, list_of_fix_descriptions)

    If no fixes apply, returns the **exact original string** (no reformatting).
    If the input has a SyntaxError, returns (code, []).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, []

    transformer = _PandasDeprecationTransformer()
    transformer.visit(tree)

    if not transformer.fixes_applied:
        return code, []

    ast.fix_missing_locations(tree)
    repaired = ast.unparse(tree)
    return repaired, transformer.fixes_applied
