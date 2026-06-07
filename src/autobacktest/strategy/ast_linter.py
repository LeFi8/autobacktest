"""AST-based static validation and linting for strategy files."""

import ast
import re
from typing import TYPE_CHECKING, Any

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
    """Resolve a chained attribute access (e.g. ``pd.DataFrame.read_csv``) to a dotted string.

    Recursively walks ``ast.Attribute`` and ``ast.Name`` nodes to reconstruct
    the full dotted path. Returns ``None`` for non-traversable node types.
    """
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        val = _get_attribute_chain(node.value)
        if val is not None:
            return f"{val}.{node.attr}"
    return None


def _count_node_lines(node: ast.AST) -> int:
    """Measure the line count of a parsed AST node.

    Uses ``end_lineno`` and ``lineno`` attributes available on all nodes
    in Python 3.8+.

    Args:
        node: AST node to measure.

    Returns:
        int: Number of source lines spanned by the node, or 0 if
        position attributes are unavailable.
    """
    end: int | None = getattr(node, "end_lineno", None)
    start: int | None = getattr(node, "lineno", None)
    if end is not None and start is not None:
        return end - start + 1
    return 0


def _calculate_complexity(node: ast.AST) -> int:
    """Calculate McCabe-style cyclomatic complexity of a function AST.

    Counts decision points (if/for/while/and/or/ternary/except-handler/comprehensions)
    and adds 1 for the base path.

    Args:
        node: A function or method AST node.

    Returns:
        int: Cyclomatic complexity score (minimum 1).
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


def _check_import_node(node: ast.Import) -> "ValidationResult | None":
    """Return a failed result if an ``ast.Import`` is non-whitelisted or uses a forbidden alias."""
    from autobacktest.strategy.validator import ValidationError, ValidationResult

    for alias in node.names:
        root_module = alias.name.split(".")[0]
        if root_module not in settings.parsed_safe_imports:
            return ValidationResult(
                passed=False,
                error_code=ValidationError.AST_BLOCKED_IMPORT,
                detail=f"Import of non-whitelisted module '{alias.name}' is strictly blocked.",
            )
        if alias.asname and alias.asname in FORBIDDEN_NAMES:
            return ValidationResult(
                passed=False,
                error_code=ValidationError.AST_BLOCKED_IMPORT,
                detail=f"Import alias '{alias.asname}' is strictly blocked.",
            )
    return None


def _check_import_from_node(node: ast.ImportFrom) -> "ValidationResult | None":
    """Return a failed result if an ``ast.ImportFrom`` violates whitelist or uses forbidden names."""
    from autobacktest.strategy.validator import ValidationError, ValidationResult

    if not node.module:
        return None
    root_module = node.module.split(".")[0]
    if root_module not in settings.parsed_safe_imports:
        return ValidationResult(
            passed=False,
            error_code=ValidationError.AST_BLOCKED_IMPORT,
            detail=f"Import from non-whitelisted module '{node.module}' is strictly blocked.",
        )
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
    return None


def _check_string_constant_node(node: ast.Constant) -> "ValidationResult | None":
    """Return a failed result if a string constant embeds a dunder in a format-string pattern."""
    from autobacktest.strategy.validator import ValidationError, ValidationResult

    val: Any = node.value
    if "{" not in val or "__" not in val:
        return None
    for brace in re.findall(r"\{[^}]*\}", val):
        if "__" in brace:
            return ValidationResult(
                passed=False,
                error_code=ValidationError.AST_BLOCKED_IMPORT,
                detail=f"String constant contains dunder reference in format pattern: '{brace}'",
            )
    return None


def _check_name_node(node: ast.Name) -> "ValidationResult | None":
    """Return a failed result if a ``Name`` node uses a forbidden identifier."""
    from autobacktest.strategy.validator import ValidationError, ValidationResult

    if node.id in FORBIDDEN_NAMES:
        return ValidationResult(
            passed=False,
            error_code=ValidationError.AST_BLOCKED_IMPORT,
            detail=f"Use of forbidden name or builtin '{node.id}' is strictly blocked.",
        )
    return None


def _check_attribute_node(node: ast.Attribute) -> "ValidationResult | None":
    """Return a failed result if an ``Attribute`` node accesses a forbidden or dunder attribute."""
    from autobacktest.strategy.validator import ValidationError, ValidationResult

    if node.attr in FORBIDDEN_NAMES or node.attr.startswith("__"):
        return ValidationResult(
            passed=False,
            error_code=ValidationError.AST_BLOCKED_IMPORT,
            detail=f"Use of forbidden attribute or dunder property '{node.attr}' is strictly blocked.",
        )
    chain = _get_attribute_chain(node)
    if chain:
        for part in chain.split("."):
            if part in FORBIDDEN_NAMES or part.startswith("__"):
                return ValidationResult(
                    passed=False,
                    error_code=ValidationError.AST_BLOCKED_IMPORT,
                    detail=(
                        f"Use of forbidden attribute or dunder property '{part}' in '{chain}' is strictly blocked."
                    ),
                )
    return None


def _check_ast_security_walk(tree: ast.Module) -> "ValidationResult | None":
    """Walk *tree* once and return the first security violation found, or ``None``."""
    for node in ast.walk(tree):
        result: ValidationResult | None = None
        if isinstance(node, ast.Import):
            result = _check_import_node(node)
        elif isinstance(node, ast.ImportFrom):
            result = _check_import_from_node(node)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            result = _check_string_constant_node(node)
        elif isinstance(node, ast.Name):
            result = _check_name_node(node)
        elif isinstance(node, ast.Attribute):
            result = _check_attribute_node(node)
        if result is not None:
            return result
    return None


def _check_ast_complexity(tree: ast.Module) -> "ValidationResult | None":
    """Return a failed result if any function exceeds the line or cyclomatic-complexity limits."""
    from autobacktest.strategy.validator import ValidationError, ValidationResult

    for func_node in ast.walk(tree):
        if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
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
    return None


def _check_ast(content: str) -> "ValidationResult":
    """Parse strategy code via AST and block non-whitelisted imports or unsafe calls.

    Checks import whitelist, forbidden names/attributes, dunder escapes
    in format strings, cyclomatic complexity, function line limits, and
    undefined name references.

    Args:
        content: Raw Python source code of the strategy.

    Returns:
        ValidationResult: ``passed=True`` when all AST checks succeed.
    """
    from autobacktest.strategy.validator import ValidationError, ValidationResult

    try:
        tree = ast.parse(content)
    except Exception as e:
        return ValidationResult(
            passed=False,
            error_code=ValidationError.IMPORT_FAILED,
            detail=f"AST parsing syntax error: {e}",
        )

    security_result = _check_ast_security_walk(tree)
    if security_result is not None:
        return security_result

    complexity_result = _check_ast_complexity(tree)
    if complexity_result is not None:
        return complexity_result

    undefined_res = _check_undefined_names(tree)
    if undefined_res is not None:
        return undefined_res

    return ValidationResult(passed=True)


def _extract_names(node: ast.AST | None, scope: set[str]) -> None:
    """Recursively extract variable names from *node* into *scope*.

    Handles simple ``ast.Name`` targets, tuple/list unpacking, and ``None``.

    Args:
        node: AST node to extract names from (may be ``None``).
        scope: Mutable set to populate with extracted names.
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

    Args:
        body: List of AST statement nodes to traverse.
        scope: Mutable set to populate with defined names.
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
    """Build a ``set`` of locally defined names within *func_node*'s body.

    Includes parameter names and any names defined via assignments, ``for``
    targets, ``with`` variables, and named expressions within the body.

    Args:
        func_node: The function definition AST node.

    Returns:
        set[str]: Locally defined names.
    """
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
    """Return True if child is a descendant of ancestor node in the AST.

    Walks the parent_map upward from *child* to root looking for *ancestor*.

    Args:
        child: Node to check for descent.
        ancestor: Potential ancestor node.
        parent_map: Mapping of ``id(node) -> parent`` built by AST traversal.

    Returns:
        bool: True when *ancestor* appears between *child* and the root.
    """
    curr: ast.AST | None = child
    while curr is not None:
        if curr == ancestor:
            return True
        curr = parent_map.get(id(curr))
    return False


def _build_module_scope(tree: ast.Module) -> set[str]:
    """Collect all names defined at module level (imports, defs, assignments).

    Args:
        tree: Parsed module AST.

    Returns:
        set[str]: Names visible at module scope.
    """
    module_scope: set[str] = set()
    for child in tree.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_scope.add(child.name)
        elif isinstance(child, ast.Import):
            for alias in child.names:
                module_scope.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(child, ast.ImportFrom):
            for alias in child.names:
                module_scope.add(alias.asname or alias.name)
        elif isinstance(child, ast.Assign):
            for target in child.targets:
                _extract_names(target, module_scope)
        elif isinstance(child, (ast.AnnAssign, ast.AugAssign)) and isinstance(child.target, ast.Name):
            module_scope.add(child.target.id)
    return module_scope


def _build_closure_scope(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    parent_map: dict[int, ast.AST],
) -> set[str]:
    """Return names visible to *func_node* from all enclosing function scopes.

    Args:
        func_node: The function whose closure scope to build.
        parent_map: Mapping of ``id(node) -> parent`` for the full AST.

    Returns:
        set[str]: Combined names from all enclosing function definitions.
    """
    closure_scope: set[str] = set()
    curr = parent_map.get(id(func_node))
    while curr is not None:
        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef)):
            closure_scope |= _build_function_scope(curr)
        curr = parent_map.get(id(curr))
    return closure_scope


def _get_enclosing_func(
    ref_node: ast.AST,
    parent_map: dict[int, ast.AST],
) -> ast.AST | None:
    """Walk *parent_map* from *ref_node* to find its immediately enclosing function.

    Args:
        ref_node: The name reference node.
        parent_map: Mapping of ``id(node) -> parent`` for the full AST.

    Returns:
        ast.AST | None: The nearest enclosing ``FunctionDef``/``AsyncFunctionDef``, or ``None``.
    """
    cur = parent_map.get(id(ref_node))
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cur
        cur = parent_map.get(id(cur))
    return None


def _defined_in_lambda(
    name: str,
    ref_node: ast.AST,
    lambda_node: ast.Lambda,
    parent_map: dict[int, ast.AST],
) -> bool:
    """Return True when *name* is a parameter of *lambda_node* and *ref_node* is in its body.

    Args:
        name: Identifier to look up.
        ref_node: The ``ast.Name`` node being validated.
        lambda_node: The enclosing ``ast.Lambda`` node.
        parent_map: Mapping of ``id(node) -> parent`` for the full AST.

    Returns:
        bool: True if *name* resolves to a lambda parameter.
    """
    if not _is_descendant(ref_node, lambda_node.body, parent_map):
        return False
    lambda_args: set[str] = set()
    for arg in lambda_node.args.args:
        lambda_args.add(arg.arg)
    if lambda_node.args.vararg:
        lambda_args.add(lambda_node.args.vararg.arg)
    if lambda_node.args.kwarg:
        lambda_args.add(lambda_node.args.kwarg.arg)
    for arg in lambda_node.args.kwonlyargs:
        lambda_args.add(arg.arg)
    for arg in lambda_node.args.posonlyargs:
        lambda_args.add(arg.arg)
    return name in lambda_args


def _defined_in_comprehension(
    name: str,
    ref_node: ast.AST,
    comp_node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
    parent_map: dict[int, ast.AST],
) -> bool:
    """Return True when *name* is a generator target visible at *ref_node* in *comp_node*.

    Args:
        name: Identifier to look up.
        ref_node: The ``ast.Name`` node being validated.
        comp_node: The enclosing comprehension or generator expression.
        parent_map: Mapping of ``id(node) -> parent`` for the full AST.

    Returns:
        bool: True if *name* is bound by a comprehension target in scope.
    """
    for gen_idx, gen in enumerate(comp_node.generators):
        is_in_iter = any(
            _is_descendant(ref_node, comp_node.generators[p_idx].iter, parent_map) for p_idx in range(gen_idx + 1)
        )
        if not is_in_iter:
            comp_targets: set[str] = set()
            _extract_names(gen.target, comp_targets)
            if name in comp_targets:
                return True
    return False


def _defined_in_comp_or_lambda(
    name: str,
    ref_node: ast.AST,
    func_node: ast.AST,
    parent_map: dict[int, ast.AST],
) -> bool:
    """Return True if *name* is defined in an enclosing comprehension or lambda scope.

    Walks up the parent chain from *ref_node* until reaching *func_node*,
    checking each comprehension and lambda boundary it crosses.

    Args:
        name: Identifier to look up.
        ref_node: The ``ast.Name`` node being validated.
        func_node: The function-def node that bounds the walk.
        parent_map: Mapping of ``id(node) -> parent`` for the full AST.

    Returns:
        bool: True when *name* is bound by a comprehension target or lambda parameter.
    """
    curr_parent = parent_map.get(id(ref_node))
    while curr_parent is not func_node and curr_parent is not None:
        if isinstance(curr_parent, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            if _defined_in_comprehension(name, ref_node, curr_parent, parent_map):
                return True
        elif isinstance(curr_parent, ast.Lambda) and _defined_in_lambda(name, ref_node, curr_parent, parent_map):
            return True
        curr_parent = parent_map.get(id(curr_parent))
    return False


def _check_function_names(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    func_scope: set[str],
    closure_scope: set[str],
    module_scope: set[str],
    parent_map: dict[int, ast.AST],
) -> "ValidationResult | None":
    """Return a failed result if any ``Name`` load in *func_node* is undefined.

    Only validates references whose immediately enclosing function is *func_node*
    (inner functions are skipped — they are validated by their own iteration).

    Args:
        func_node: Function definition to validate.
        func_scope: Names locally defined inside *func_node*.
        closure_scope: Names visible from all enclosing function defs.
        module_scope: Names defined at module level.
        parent_map: Mapping of ``id(node) -> parent`` for the full AST.

    Returns:
        ValidationResult | None: ``None`` if all names resolve, else a failed result.
    """
    from autobacktest.strategy.validator import ValidationError, ValidationResult

    for ref_node in ast.walk(func_node):
        if not isinstance(ref_node, ast.Name) or not isinstance(ref_node.ctx, ast.Load):
            continue
        if _get_enclosing_func(ref_node, parent_map) is not func_node:
            continue
        name = ref_node.id
        if name in closure_scope:
            continue
        if _defined_in_comp_or_lambda(name, ref_node, func_node, parent_map):
            continue
        if name not in func_scope and name not in module_scope and name not in _BUILTIN_NAMES:
            return ValidationResult(
                passed=False,
                error_code=ValidationError.UNDEFINED_NAME,
                detail=f"Name '{name}' is not defined in function '{func_node.name}' or any enclosing scope.",
            )
    return None


def _check_undefined_names(tree: ast.Module) -> "ValidationResult | None":
    """Return a failed :class:`ValidationResult` if any function references a

    name that is not defined in the function body, the module scope, or among
    builtins.

    Handles closure scopes (walking ``parent_map`` for enclosing function
    defs), comprehension generators, and lambda parameter names.

    Args:
        tree: Parsed AST module to validate.

    Returns:
        ValidationResult | None: ``None`` when no undefined names are found,
        or a failed ``ValidationResult`` with ``UNDEFINED_NAME`` code.
    """
    module_scope = _build_module_scope(tree)

    parent_map: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent_map[id(child)] = node

    for func_node in ast.walk(tree):
        if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        func_scope = _build_function_scope(func_node)
        closure_scope = _build_closure_scope(func_node, parent_map)
        result = _check_function_names(func_node, func_scope, closure_scope, module_scope, parent_map)
        if result is not None:
            return result

    return None
