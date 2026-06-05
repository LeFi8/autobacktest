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
import re

_ANY_WORD_RE = re.compile(r"\bAny\b")

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
                self.fixes_applied.append(f"Remapped freq alias '{old_val}' to '{new_val}' for {context} context")

    def _remap_freq_kwarg(self, node: ast.Call, alias_map: dict[str, str], context: str) -> None:
        """Remap a freq= keyword if it is a deprecated freq string for *context*."""
        for kw in node.keywords:
            if kw.arg == "freq" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                new_val = alias_map.get(kw.value.value)
                if new_val is not None:
                    old_val = kw.value.value
                    kw.value = ast.Constant(value=new_val)
                    ast.fix_missing_locations(kw.value)
                    self.fixes_applied.append(f"Remapped freq alias '{old_val}' to '{new_val}' for {context} context")

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
                    new_keywords = [kw for kw in node.keywords if kw.arg != "method"]
                    new_call = ast.Call(
                        func=ast.Attribute(value=node.func.value, attr="ffill", ctx=ast.Load()),
                        args=node.args,
                        keywords=new_keywords,
                    )
                    ast.copy_location(new_call, node)
                    ast.fix_missing_locations(new_call)
                    self.fixes_applied.append("Replaced .fillna(method='ffill') with .ffill()")
                    return new_call

                if method_value in ("bfill", "backfill"):
                    new_keywords = [kw for kw in node.keywords if kw.arg != "method"]
                    new_call = ast.Call(
                        func=ast.Attribute(value=node.func.value, attr="bfill", ctx=ast.Load()),
                        args=node.args,
                        keywords=new_keywords,
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


class _MissingImportInjector(ast.NodeTransformer):
    """Injects ``from typing import Any`` when ``Any`` is used in type annotations
    but not imported.  This is a common LLM mistake that causes a hard runtime
    :class:`NameError` in the sandboxed subprocess.

    The import is placed after the module docstring (if one exists) to avoid
    breaking Python's docstring recognition — any statement before a string
    literal at module level prevents it from being registered as ``__doc__``.
    """

    def __init__(self) -> None:
        self.fixes_applied: list[str] = []

    def visit_Module(self, node: ast.Module) -> ast.AST:
        # Scan the entire tree for ``Any`` usage and existing typing imports.
        any_used = False
        typing_imported = False
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id == "Any":
                any_used = True
            elif isinstance(child, ast.Constant) and isinstance(child.value, str):
                # String annotations like ``'dict[str, Any]'``
                if _ANY_WORD_RE.search(child.value):
                    any_used = True
            elif isinstance(child, ast.ImportFrom):
                if child.module == "typing":
                    for alias in child.names:
                        if alias.name == "Any":
                            typing_imported = True
            elif isinstance(child, ast.Import):
                for alias in child.names:
                    if alias.name == "typing":
                        typing_imported = True

        if any_used and not typing_imported:
            import_node = ast.ImportFrom(
                module="typing",
                names=[ast.alias(name="Any")],
                level=0,
            )
            ast.fix_missing_locations(import_node)
            insert_idx = 0
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                insert_idx = 1
            node.body.insert(insert_idx, import_node)
            self.fixes_applied.append("Injected missing `from typing import Any`")

        return self.generic_visit(node)


class _WeightRenormalizer(ast.NodeTransformer):
    """Injects a final weight-renormalization step before ``return`` inside
    ``generate_signals`` when the function does not already perform one.

    Float-accumulation errors from multiple sequential normalisation / capping
    passes routinely cause row sums to drift above the ``1.0 + 1e-5`` tolerance
    enforced by :func:`~autobacktest.strategy.contract.validate_output`.
    """

    def __init__(self) -> None:
        self.fixes_applied: list[str] = []

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _has_renormalization(stmts: list[ast.stmt]) -> bool:
        """Return ``True`` if *stmts* already contain ``.clip(lower=0.0)``."""
        for stmt in stmts:
            for child in ast.walk(stmt):
                if isinstance(child, ast.Call):
                    name = _get_call_name(child)
                    if name == "clip":
                        for kw in child.keywords:
                            if kw.arg == "lower" and isinstance(kw.value, ast.Constant) and kw.value.value == 0.0:
                                return True
        return False

    @staticmethod
    def _extract_return_name(stmt: ast.Return) -> str | None:
        """If *stmt* is ``return <name>`` (a simple :class:`ast.Name`), return
        the identifier.  Otherwise return ``None``."""
        if isinstance(stmt.value, ast.Name):
            return stmt.value.id
        return None

    def _build_renorm_stmts(self, var: str) -> list[ast.stmt]:
        r"""Build the AST for::

        {var} = {var}.clip(lower=0.0)
        {var} = {var}.div({var}.sum(axis=1), axis=0).fillna(0.0)
        """
        v_load = ast.Name(id=var, ctx=ast.Load())
        v_store = ast.Name(id=var, ctx=ast.Store())

        #   .clip(lower=0.0)
        clip = ast.Assign(
            targets=[v_store],
            value=ast.Call(
                func=ast.Attribute(value=v_load, attr="clip", ctx=ast.Load()),
                args=[],
                keywords=[ast.keyword(arg="lower", value=ast.Constant(value=0.0))],
            ),
        )

        #   .sum(axis=1)
        sum_call = ast.Call(
            func=ast.Attribute(value=v_load, attr="sum", ctx=ast.Load()),
            args=[],
            keywords=[ast.keyword(arg="axis", value=ast.Constant(value=1))],
        )
        #   .div(…, axis=0)
        div_call = ast.Call(
            func=ast.Attribute(value=v_load, attr="div", ctx=ast.Load()),
            args=[sum_call],
            keywords=[ast.keyword(arg="axis", value=ast.Constant(value=0))],
        )
        #   .fillna(0.0)
        fillna_call = ast.Call(
            func=ast.Attribute(value=div_call, attr="fillna", ctx=ast.Load()),
            args=[ast.Constant(value=0.0)],
            keywords=[],
        )
        div_assign = ast.Assign(targets=[v_store], value=fillna_call)

        ast.fix_missing_locations(clip)
        ast.fix_missing_locations(div_assign)
        return [clip, div_assign]

    # ------------------------------------------------------------------
    # transformer hook
    # ------------------------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)

        if node.name != "generate_signals":
            return node

        # Locate the final simple ``return <var>``.
        return_idx: int | None = None
        return_var: str | None = None
        for i, stmt in enumerate(node.body):
            if isinstance(stmt, ast.Return):
                name = self._extract_return_name(stmt)
                if name is not None:
                    return_idx = i
                    return_var = name

        if return_idx is None or return_var is None:
            return node

        # Skip if renormalization already present.
        if self._has_renormalization(node.body):
            return node

        renorm = self._build_renorm_stmts(return_var)
        node.body[return_idx:return_idx] = renorm
        self.fixes_applied.append(f"Injected weight renormalization for ``{return_var}`` before return")
        return node


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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


def repair_strategy_code(code: str) -> tuple[str, list[str]]:
    """Run all AST-based repair passes on *code* and return the fixed source.

    Current passes in order:

    1. :class:`_PandasDeprecationTransformer` — deprecated pandas API fixes
    2. :class:`_MissingImportInjector` — inject ``from typing import Any``
    3. :class:`_WeightRenormalizer` — inject ``.clip(lower=0.0)`` /
       ``.div(…).fillna(0.0)`` before the final return of ``generate_signals``

    Returns:
        (repaired_code, list_of_fix_descriptions)

    If no fixes apply across any pass, returns the **exact original string**
    (no reformatting).  If the input has a SyntaxError, returns (code, []).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, []

    passes: list[ast.NodeTransformer] = [
        _PandasDeprecationTransformer(),
        _MissingImportInjector(),
        _WeightRenormalizer(),
    ]

    all_fixes: list[str] = []
    for p in passes:
        p.visit(tree)
        all_fixes.extend(getattr(p, "fixes_applied", []))

    if not all_fixes:
        return code, []

    ast.fix_missing_locations(tree)
    repaired = ast.unparse(tree)
    return repaired, all_fixes
