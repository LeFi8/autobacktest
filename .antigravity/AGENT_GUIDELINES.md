# Agent Guidelines: AutoBacktest

## Code Conventions
- **Strict Typing**: All code under `src/` must have full type annotations. No untyped parameters, no `Any` declarations unless explicitly necessary (e.g. dynamic callbacks/imports). Verify strictly using:
  ```bash
  uv run mypy --strict src/
  ```
- **Ruff Compliance**: Line length limits are set to `88`. All code changes must be formatted and checked:
  ```bash
  uv run ruff check . --fix
  ```
- **Docstrings & Comments**: Keep existing comments and docstrings. Write Google-style docstrings for new functions, including parameter lists and return values.

## Routing and Folder Structure
- **Core Package**: Keep core logic isolated in `src/autobacktest/`.
- **Dynamic Strategies**: Put customizable strategies inside the `strategies/` root folder. Every strategy module must export:
  ```python
  def generate_signals(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
  ```
- **Strategy Configs**: Put YAML configuration definitions under `configs/` matching strategy filename (e.g., `configs/haa.yaml`).
- **Tests**: Put test cases matching module layout under `tests/` named `test_<module>.py`.

## Forbidden Practices
- Do not import private sub-modules. Keep system APIs flat.
- Do not run untrusted dynamic scripts. Only run imported strategies with bounded parameter inputs.
- Do not bypass type rules or formatting. Diffs containing formatting inconsistencies will fail pre-commit hooks.
- Do not edit human-facing workspace rules (such as `README.md`, `CONTRIBUTING.md`, or `LICENSE`) unless specifically instructed.

## Error Handling
- Never use bare `except:` constructs. Catch specific, granular exceptions (e.g. `FileNotFoundError`, `ValueError`).
- Prefer returning descriptive dataclasses (like `WindowReport` or `EvaluationReport`) instead of nested dict structures.
- Use `structlog` for application log emissions. Do not pollute standard streams with raw `print()` statements unless writing CLI-directed outputs.

## Testing Guidelines
- Any code change must preserve test integrity. Run existing suite to confirm:
  ```bash
  uv run pytest
  ```
- Add unit tests for new modules. Aim for module line coverage >= 85%.
- Maintain determinism: ensure all statistical simulations (DSR, bootstrap) accept and execute with explicit, locked random seeds (default: `42`).


