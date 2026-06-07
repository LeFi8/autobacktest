"""Security-critical constants for strategy validation.

``FORBIDDEN_NAMES`` is a set of ~73 blocked Python identifiers that the
AST scanner uses to reject unsafe code patterns.  It covers:

- Dangerous builtins (``exec``, ``eval``, ``compile``, ``open``).
- ``__dunder__`` attribute access (``__builtins__``, ``__import__``).
- NumPy / pandas file I/O escapes (``load``, ``save``, ``to_csv``,
  ``read_sql``, ``to_pickle``, etc.).

Any strategy code referencing these names — at import level, inside
string constants, or as attribute chains — is rejected during preflight.
"""

# Forbidden variables, functions, and names that compromise sandboxing
FORBIDDEN_NAMES = {
    "exec",
    "eval",
    "compile",
    "format",
    "format_map",
    "vformat",
    "open",
    "__import__",
    "globals",
    "locals",
    "getattr",
    "setattr",
    "delattr",
    "__builtins__",
    "vars",
    "breakpoint",
    # numpy / pandas sandboxing escapes
    "load",
    "save",
    "savez",
    "savez_compressed",
    "memmap",
    "fromfile",
    "tofile",
    "loadtxt",
    "genfromtxt",
    "fromregex",
    "DataSource",
    "read_csv",
    "read_table",
    "read_fwf",
    "to_csv",
    "read_json",
    "to_json",
    "read_excel",
    "to_excel",
    "read_pickle",
    "to_pickle",
    "read_parquet",
    "to_parquet",
    "read_hdf",
    "to_hdf",
    "read_feather",
    "to_feather",
    "read_xml",
    "to_xml",
    "read_html",
    "to_html",
    "read_sql",
    "read_sql_table",
    "read_sql_query",
    "to_sql",
    "read_clipboard",
    "to_clipboard",
    "io",
    "get_handle",
    "lib",
    "npyio",
    "HDFStore",
    "ExcelWriter",
    "ExcelFile",
    "read_sas",
    "read_spss",
    "read_gbq",
    "read_stata",
    "read_orc",
    "to_stata",
    "to_orc",
}
