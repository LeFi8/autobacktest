"""
AST-based repair module for deprecated pandas API calls.

Transforms pandas 1.x/2.x patterns to pandas 3.x-compatible equivalents.
Note: ast.unparse drops comments from repaired files (accepted tradeoff).

Frequency aliases are context-sensitive in pandas 3.x:
- DatetimeIndex operations (resample, date_range, bdate_range, Grouper) require
  the NEW aliases: 'ME', 'BME', 'QE', 'YE', 'h', 'min', 's'.
- Period operations (to_period, period_range) require the ORIGINAL codes:
  'M', 'Q', 'Y'. Passing 'ME'/'QE'/'YE' to a Period context raises ValueError.
- ``asfreq`` is intentionally left untouched: its correct alias depends on
  whether the receiver is a DatetimeIndex (wants 'ME') or PeriodIndex (wants
  'M'), which cannot be determined statically.
"""

import ast

# Deprecated DatetimeIndex freq aliases → pandas 3.x replacements (forward)
_FREQ_ALIAS_MAP: dict[str, str] = {
    "M": "ME",
    "BM": "BME",
    "Q": "QE",
    "BQ": "BQE",
    "A": "YE",
    "Y": "YE",
    "BA": "BYE",
    "BY": "BYE",
    "H": "h",
    "T": "min",
    "S": "s",
}

# DatetimeIndex aliases mistakenly applied to Period contexts → original codes (reverse)
_FREQ_REVERSE_MAP: dict[str, str] = {
    "ME": "M",
    "BME": "BM",
    "QE": "Q",
    "BQE": "BQ",
    "YE": "Y",
    "BYE": "BY",
}

# Functions whose freq argument targets a DatetimeIndex (want NEW aliases).
# resample takes freq as its first positional arg ("rule"); the others via freq=.
_DT_FREQ_FIRST_ARG_FUNCS = {"resample"}
_DT_FREQ_KW_FUNCS = {"resample", "date_range", "bdate_range", "Grouper"}

# Functions whose freq argument targets a Period (want ORIGINAL codes).
# to_period takes freq as its first positional arg; period_range via freq=.
_PERIOD_FREQ_FIRST_ARG_FUNCS = {"to_period"}
_PERIOD_FREQ_KW_FUNCS = {"to_period", "period_range"}

# Aggregation functions that accept level= keyword
_LEVEL_AGG_FUNCS = {"mean", "sum", "std", "min", "max"}


def _get_call_name(node: ast.Call) -> str | None:
    """Return the bare function/method name from a Call node, or None."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


class _PandasDeprecationTransformer(ast.NodeTransformer):
    """Rewrites deprecated pandas API patterns in-place on the AST."""

    def __init__(self) -> None:
        self.fixes_applied: list[str] = []

    def _remap_first_arg(self, node: ast.Call, alias_map: dict[str, str], context: str) -> None:
        """Remap node.args[0] if it is a deprecated freq string for *context*."""
        if not node.args:
            return
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            new_val = alias_map.get(first_arg.value)
            if new_val is not None:
                old_val = first_arg.value
                node.args[0] = ast.Constant(value=new_val)
                ast.fix_missing_locations(node.args[0])
                self.fixes_applied.append(
                    f"Remapped freq alias '{old_val}' to '{new_val}' for {context} context"
                )

    def _remap_freq_kwarg(self, node: ast.Call, alias_map: dict[str, str], context: str) -> None:
        """Remap a freq= keyword if it is a deprecated freq string for *context*."""
        for kw in node.keywords:
            if kw.arg == "freq" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                new_val = alias_map.get(kw.value.value)
                if new_val is not None:
                    old_val = kw.value.value
                    kw.value = ast.Constant(value=new_val)
                    ast.fix_missing_locations(kw.value)
                    self.fixes_applied.append(
                        f"Remapped freq alias '{old_val}' to '{new_val}' for {context} context"
                    )

    def _remap_freq(self, node: ast.Call, name: str) -> None:
        """Apply context-sensitive freq alias remapping for known freq functions."""
        # DatetimeIndex contexts: old codes ('M') → new aliases ('ME').
        if name in _DT_FREQ_FIRST_ARG_FUNCS:
            self._remap_first_arg(node, _FREQ_ALIAS_MAP, "DatetimeIndex")
        if name in _DT_FREQ_KW_FUNCS:
            self._remap_freq_kwarg(node, _FREQ_ALIAS_MAP, "DatetimeIndex")
        # Period contexts: new aliases ('ME') → original codes ('M').
        if name in _PERIOD_FREQ_FIRST_ARG_FUNCS:
            self._remap_first_arg(node, _FREQ_REVERSE_MAP, "Period")
        if name in _PERIOD_FREQ_KW_FUNCS:
            self._remap_freq_kwarg(node, _FREQ_REVERSE_MAP, "Period")

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # Visit children first (bottom-up)
        self.generic_visit(node)

        name = _get_call_name(node)
        if name is None:
            return node

        # Structural transforms below require a method call (node.func.value).
        if isinstance(node.func, ast.Attribute):
            # -------------------------------------------------------------- #
            # 1. groupby(axis=...) — drop the axis= keyword                   #
            # -------------------------------------------------------------- #
            if name == "groupby":
                new_keywords = [kw for kw in node.keywords if kw.arg != "axis"]
                if len(new_keywords) < len(node.keywords):
                    node.keywords = new_keywords
                    self.fixes_applied.append("Removed deprecated axis= argument from .groupby()")

            # -------------------------------------------------------------- #
            # 2. .fillna(method='ffill'/'pad') → .ffill()                     #
            #    .fillna(method='bfill'/'backfill') → .bfill()                #
            # -------------------------------------------------------------- #
            elif name == "fillna":
                method_value = None
                for kw in node.keywords:
                    if kw.arg == "method" and isinstance(kw.value, ast.Constant):
                        method_value = kw.value.value
                        break

                if method_value in ("ffill", "pad"):
                    new_call = ast.Call(
                        func=ast.Attribute(value=node.func.value, attr="ffill", ctx=ast.Load()),
                        args=[],
                        keywords=[],
                    )
                    ast.copy_location(new_call, node)
                    ast.fix_missing_locations(new_call)
                    self.fixes_applied.append("Replaced .fillna(method='ffill') with .ffill()")
                    return new_call

                if method_value in ("bfill", "backfill"):
                    new_call = ast.Call(
                        func=ast.Attribute(value=node.func.value, attr="bfill", ctx=ast.Load()),
                        args=[],
                        keywords=[],
                    )
                    ast.copy_location(new_call, node)
                    ast.fix_missing_locations(new_call)
                    self.fixes_applied.append("Replaced .fillna(method='bfill') with .bfill()")
                    return new_call

            # -------------------------------------------------------------- #
            # 3. .mean/sum/std/min/max(level=L) → .groupby(level=L).FUNC()    #
            # -------------------------------------------------------------- #
            elif name in _LEVEL_AGG_FUNCS:
                level_kw = None
                other_kwargs = []
                for kw in node.keywords:
                    if kw.arg == "level":
                        level_kw = kw
                    else:
                        other_kwargs.append(kw)

                if level_kw is not None:
                    groupby_call = ast.Call(
                        func=ast.Attribute(value=node.func.value, attr="groupby", ctx=ast.Load()),
                        args=[],
                        keywords=[level_kw],
                    )
                    new_call = ast.Call(
                        func=ast.Attribute(value=groupby_call, attr=name, ctx=ast.Load()),
                        args=list(node.args),
                        keywords=other_kwargs,
                    )
                    ast.copy_location(groupby_call, node)
                    ast.copy_location(new_call, node)
                    ast.fix_missing_locations(new_call)
                    self.fixes_applied.append(f"Replaced .{name}(level=) with .groupby(level=).{name}()")
                    return new_call

        # ------------------------------------------------------------------ #
        # 4. Context-sensitive freq alias remapping (any call style)          #
        # ------------------------------------------------------------------ #
        self._remap_freq(node, name)
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
