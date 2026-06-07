"""AST-based static validation and linting for strategy files."""

import ast
import re
from typing import TYPE_CHECKING

from autobacktest.config import settings
from autobacktest.strategy.constants import FORBIDDEN_NAMES

if TYPE_CHECKING:
    from autobacktest.strategy.validator import ValidationResult

# Names available as Python builtins (never cause NameError).
_BUILTIN_NAMES: frozenset[str] = frozenset(
    {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "classmethod",
        "compile",
        "complex",
        "delattr",
        "dict",
        "dir",
        "divmod",
        "enumerate",
        "eval",
        "exec",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "globals",
        "hasattr",
        "hash",
        "hex",
        "id",
        "input",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "locals",
        "map",
        "max",
        "memoryview",
        "min",
        "next",
        "object",
        "oct",
        "open",
        "ord",
        "pow",
        "print",
        "property",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "setattr",
        "slice",
        "sorted",
        "staticmethod",
        "str",
        "sum",
        "super",
        "tuple",
        "type",
        "vars",
        "zip",
        # Standard exceptions and warnings
        "BaseException",
        "Exception",
        "ArithmeticError",
        "BufferError",
        "LookupError",
        "AssertionError",
        "AttributeError",
        "EOFError",
        "FloatingPointError",
        "GeneratorExit",
        "ImportError",
        "ModuleNotFoundError",
        "IndexError",
        "KeyError",
        "KeyboardInterrupt",
        "MemoryError",
        "NameError",
        "NotImplementedError",
        "OSError",
        "OverflowError",
        "RecursionError",
        "ReferenceError",
        "RuntimeError",
        "StopIteration",
        "StopAsyncIteration",
        "SyntaxError",
        "IndentationError",
        "TabError",
        "SystemError",
        "SystemExit",
        "TypeError",
        "UnboundLocalError",
        "UnicodeError",
        "UnicodeEncodeError",
        "UnicodeDecodeError",
        "UnicodeTranslateError",
        "ValueError",
        "ZeroDivisionError",
        "Warning",
        "UserWarning",
        "DeprecationWarning",
        "PendingDeprecationWarning",
        "SyntaxWarning",
        "RuntimeWarning",
        "FutureWarning",
        "ImportWarning",
        "UnicodeWarning",
        "BytesWarning",
        "ResourceWarning",
    }
)


def _get_attribute_chain(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        val = _get_attribute_chain(node.value)
        if val is not None:
            return f"{val}.{node.attr}"
    return None


def _count_node_lines(node: ast.AST) -> int:
    """Measure the line count of a parsed AST node."""
    end: int | None = getattr(node, "end_lineno", None)
    start: int | None = getattr(node, "lineno", None)
    if end is not None and start is not None:
        return end - start + 1
    return 0


def _calculate_complexity(node: ast.AST) -> int:
    """Calculate McCabe-style cyclomatic complexity of a function AST.

    Counts decision points (if/for/while/and/or/ternary/except-handler/comprehensions)
    and adds 1 for the base path.
    """
    complexity = 1
    for child in ast.walk(node):
        if isinstance(
            child,
            (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.IfExp),
        ):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            if isinstance(child.op, (ast.And, ast.Or)):
                complexity += len(child.values) - 1
        elif isinstance(
            child,
            (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp),
        ):
            complexity += 1

    return complexity


def _check_ast(content: str) -> "ValidationResult":
    """Parse strategy code via AST and block non-whitelisted imports or unsafe calls."""
    from autobacktest.strategy.validator import ValidationError, ValidationResult

    try:
        tree = ast.parse(content)
    except Exception as e:
        return ValidationResult(
            passed=False,
            error_code=ValidationError.IMPORT_FAILED,
            detail=f"AST parsing syntax error: {e}",
        )

    for node in ast.walk(tree):
        # Inspect imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_module = alias.name.split(".")[0]
                if root_module not in settings.parsed_safe_imports:
                    msg = f"Import of non-whitelisted module '{alias.name}' is strictly blocked."
                    return ValidationResult(
                        passed=False,
                        error_code=ValidationError.AST_BLOCKED_IMPORT,
                        detail=msg,
                    )
                # Inspect imported alias names
                if alias.asname and alias.asname in FORBIDDEN_NAMES:
                    return ValidationResult(
                        passed=False,
                        error_code=ValidationError.AST_BLOCKED_IMPORT,
                        detail=f"Import alias '{alias.asname}' is strictly blocked.",
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_module = node.module.split(".")[0]
                if root_module not in settings.parsed_safe_imports:
                    msg = f"Import from non-whitelisted module '{node.module}' is strictly blocked."
                    return ValidationResult(
                        passed=False,
                        error_code=ValidationError.AST_BLOCKED_IMPORT,
                        detail=msg,
                    )
                # Inspect imported names and aliases
                for alias in node.names:
                    if alias.name == "*":
                        return ValidationResult(
                            passed=False,
                            error_code=ValidationError.AST_BLOCKED_IMPORT,
                            detail="Wildcard imports (*) are strictly blocked.",
                        )
                    imported_name = alias.name.split(".")[0]
                    if imported_name in FORBIDDEN_NAMES:
                        return ValidationResult(
                            passed=False,
                            error_code=ValidationError.AST_BLOCKED_IMPORT,
                            detail=f"Forbidden import '{alias.name}' is blocked.",
                        )
                    if alias.asname and alias.asname in FORBIDDEN_NAMES:
                        return ValidationResult(
                            passed=False,
                            error_code=ValidationError.AST_BLOCKED_IMPORT,
                            detail=f"Import alias '{alias.asname}' is blocked.",
                        )

        # Inspect string constants for dunder format-string exploits
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            val: str = node.value
            if "{" in val and "__" in val:
                # Detect patterns like "{0.__class__}" or "{__import__}"
                brace_contents = re.findall(r"\{[^}]*\}", val)
                for brace in brace_contents:
                    if "__" in brace:
                        msg = f"String constant contains dunder reference in format pattern: '{brace}'"
                        return ValidationResult(
                            passed=False,
                            error_code=ValidationError.AST_BLOCKED_IMPORT,
                            detail=msg,
                        )

        # Inspect forbidden variables, functions, and builtin names
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            msg = f"Use of forbidden name or builtin '{node.id}' is strictly blocked."
            return ValidationResult(
                passed=False,
                error_code=ValidationError.AST_BLOCKED_IMPORT,
                detail=msg,
            )

        # Inspect forbidden attributes (prevents dunder escapes & chained)
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_NAMES or node.attr.startswith("__"):
                msg = f"Use of forbidden attribute or dunder property '{node.attr}' is strictly blocked."
                return ValidationResult(
                    passed=False,
                    error_code=ValidationError.AST_BLOCKED_IMPORT,
                    detail=msg,
                )
            chain = _get_attribute_chain(node)
            if chain:
                parts = chain.split(".")
                for part in parts:
                    if part in FORBIDDEN_NAMES or part.startswith("__"):
                        msg = (
                            f"Use of forbidden attribute or dunder property '{part}' in '{chain}' is strictly blocked."
                        )
                        return ValidationResult(
                            passed=False,
                            error_code=ValidationError.AST_BLOCKED_IMPORT,
                            detail=msg,
                        )

    # Cyclomatic complexity and line-count check (second pass over function defs)
    for func_node in ast.walk(tree):
        if isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            line_count = _count_node_lines(func_node)
            if line_count > settings.max_function_lines:
                return ValidationResult(
                    passed=False,
                    error_code=ValidationError.AST_LINE_LIMIT_EXCEEDED,
                    detail=(
                        f"Function '{func_node.name}' has {line_count} lines, "
                        f"exceeding the limit of {settings.max_function_lines}."
                    ),
                )
            complexity = _calculate_complexity(func_node)
            if complexity > settings.max_cyclomatic_complexity:
                return ValidationResult(
                    passed=False,
                    error_code=ValidationError.AST_CYCLOMATIC_COMPLEXITY_EXCEEDED,
                    detail=(
                        f"Function '{func_node.name}' has cyclomatic complexity "
                        f"{complexity}, exceeding the limit of "
                        f"{settings.max_cyclomatic_complexity}."
                    ),
                )

    # Undefined-name pre-check
    undefined_res = _check_undefined_names(tree)
    if undefined_res is not None:
        return undefined_res

    return ValidationResult(passed=True)


def _extract_names(node: ast.AST | None, scope: set[str]) -> None:
    """Recursively extract variable names from *node* into *scope*.

    Handles simple ``ast.Name`` targets, tuple/list unpacking, and ``None``.
    """
    if node is None:
        return
    if isinstance(node, ast.Name):
        scope.add(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            _extract_names(elt, scope)


def _walk_body(body: list[ast.stmt], scope: set[str]) -> None:
    """Recursively collect locally-defined names from *body*.

    Traverses compound-statement bodies (``if``, ``for``, ``while``,
    ``try``, ``with``) but stops at ``FunctionDef`` / ``AsyncFunctionDef``
    boundaries — nested function bodies are not walked.
    """
    for stmt in body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                _extract_names(target, scope)
        elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)) and isinstance(stmt.target, ast.Name):
            scope.add(stmt.target.id)
        elif isinstance(stmt, ast.For):
            _extract_names(stmt.target, scope)
            _walk_body(stmt.body, scope)
            _walk_body(stmt.orelse, scope)
        elif isinstance(stmt, (ast.While, ast.If)):
            _walk_body(stmt.body, scope)
            _walk_body(stmt.orelse, scope)
        elif isinstance(stmt, ast.Try):
            _walk_body(stmt.body, scope)
            for handler in stmt.handlers:
                if isinstance(handler.name, str):
                    scope.add(handler.name)
                _walk_body(handler.body, scope)
            _walk_body(stmt.orelse, scope)
            _walk_body(stmt.finalbody, scope)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                _extract_names(item.optional_vars, scope)
            _walk_body(stmt.body, scope)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope.add(stmt.name)
        elif (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.NamedExpr)
            and isinstance(stmt.value.target, ast.Name)
        ):
            scope.add(stmt.value.target.id)


def _build_function_scope(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Build a ``set`` of locally defined names within *func_node*'s body."""
    scope: set[str] = set()

    for arg in func_node.args.args:
        scope.add(arg.arg)
    if func_node.args.vararg:
        scope.add(func_node.args.vararg.arg)
    if func_node.args.kwarg:
        scope.add(func_node.args.kwarg.arg)
    for arg in func_node.args.kwonlyargs:
        scope.add(arg.arg)
    for arg in func_node.args.posonlyargs:
        scope.add(arg.arg)

    _walk_body(func_node.body, scope)
    return scope


def _is_descendant(child: ast.AST, ancestor: ast.AST, parent_map: dict[int, ast.AST]) -> bool:
    """Return True if child is a descendant of ancestor node in the AST."""
    curr: ast.AST | None = child
    while curr is not None:
        if curr == ancestor:
            return True
        curr = parent_map.get(id(curr))
    return False


def _check_undefined_names(tree: ast.Module) -> "ValidationResult | None":
    """Return a failed :class:`ValidationResult` if any function references a

    name that is not defined in the function body, the module scope, or among
    builtins.
    """
    from autobacktest.strategy.validator import ValidationError, ValidationResult

    # Build module-level scope (imports + top-level defs + top-level assignments)
    module_scope: set[str] = set()
    for child in tree.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_scope.add(child.name)
        elif isinstance(child, ast.Import):
            for alias in child.names:
                name = alias.asname or alias.name.split(".")[0]
                module_scope.add(name)
        elif isinstance(child, ast.ImportFrom):
            for alias in child.names:
                name = alias.asname or alias.name
                module_scope.add(name)
        elif isinstance(child, ast.Assign):
            for target in child.targets:
                _extract_names(target, module_scope)
        elif isinstance(child, (ast.AnnAssign, ast.AugAssign)) and isinstance(child.target, ast.Name):
            module_scope.add(child.target.id)

    # Build parent map for closure-scope resolution
    parent_map: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):  # type: ignore[assignment]
            parent_map[id(child)] = node

    for func_node in ast.walk(tree):
        if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        func_scope = _build_function_scope(func_node)

        # Build closure scope: names from all enclosing function defs
        closure_scope: set[str] = set()
        curr = parent_map.get(id(func_node))
        while curr is not None:
            if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef)):
                closure_scope |= _build_function_scope(curr)
            curr = parent_map.get(id(curr))

        for ref_node in ast.walk(func_node):
            if not isinstance(ref_node, ast.Name) or not isinstance(ref_node.ctx, ast.Load):
                continue
            # Only validate names whose immediate enclosing function is func_node
            enclosing = None
            cur = parent_map.get(id(ref_node))
            while cur is not None:
                if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    enclosing = cur
                    break
                cur = parent_map.get(id(cur))
            if enclosing != func_node:
                continue
            name = ref_node.id
            if name in closure_scope:
                continue

            # Check enclosing comprehension or lambda scopes
            defined_in_comp_or_lambda = False
            curr_parent = parent_map.get(id(ref_node))
            while curr_parent is not func_node and curr_parent is not None:
                if isinstance(curr_parent, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                    for gen_idx, gen in enumerate(curr_parent.generators):
                        is_in_iter = False
                        for p_idx in range(gen_idx + 1):
                            if _is_descendant(ref_node, curr_parent.generators[p_idx].iter, parent_map):
                                is_in_iter = True
                                break
                        if not is_in_iter:
                            comp_targets: set[str] = set()
                            _extract_names(gen.target, comp_targets)
                            if name in comp_targets:
                                defined_in_comp_or_lambda = True
                                break
                    if defined_in_comp_or_lambda:
                        break
                elif isinstance(curr_parent, ast.Lambda):
                    if _is_descendant(ref_node, curr_parent.body, parent_map):
                        lambda_args: set[str] = set()
                        for arg in curr_parent.args.args:
                            lambda_args.add(arg.arg)
                        if curr_parent.args.vararg:
                            lambda_args.add(curr_parent.args.vararg.arg)
                        if curr_parent.args.kwarg:
                            lambda_args.add(curr_parent.args.kwarg.arg)
                        for arg in curr_parent.args.kwonlyargs:
                            lambda_args.add(arg.arg)
                        for arg in curr_parent.args.posonlyargs:
                            lambda_args.add(arg.arg)
                        if name in lambda_args:
                            defined_in_comp_or_lambda = True
                            break
                curr_parent = parent_map.get(id(curr_parent))

            if defined_in_comp_or_lambda:
                continue

            if name not in func_scope and name not in module_scope and name not in _BUILTIN_NAMES:
                return ValidationResult(
                    passed=False,
                    error_code=ValidationError.UNDEFINED_NAME,
                    detail=(f"Name '{name}' is not defined in function '{func_node.name}' or any enclosing scope."),
                )

    return None
