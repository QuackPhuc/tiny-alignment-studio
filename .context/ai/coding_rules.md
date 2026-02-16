# Coding Rules: Tiny Alignment Studio

## Naming

- Functions and variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private members: `_leading_underscore`
- File names: `snake_case.py`, descriptive of contents

## Type Hints

- Required on all public function signatures and class attributes.
- Use `from __future__ import annotations` for forward references.
- Prefer concrete types over `Any`. Use `Any` only at true boundaries.

## Docstrings

- Google style. Required on all public classes and functions.
- One-liner for trivial/obvious methods.
- Include `Args`, `Returns`, `Raises` sections when non-trivial.

## Comments

Comments must earn their place. Allowed:

- **Why** something is done a non-obvious way
- Tensor shape annotations: `# Shape: (batch, seq_len, hidden)`
- `TODO(owner): actionable description`
- Hardcoded values that require explanation
- Complex algorithmic decisions

Banned:

- Restating what the code does (`x = x + 1  # increment x`)
- Edit-history tags: `FIX:`, `UPDATE:`, `OLD:`, `NEW:`
- Decorative separators or ASCII art
- Commented-out code blocks

## Code Quality

- Each function does one thing. Target < 30 lines, hard limit 50.
- Max 3 levels of indentation. Use early returns and guard clauses.
- No magic numbers. Name constants or pull from config.
- No dead code: no unused imports, no unreachable branches.
- Fail fast: validate at boundaries, raise with context.

## Emoji Policy

- Python source, YAML, JSON, configs: **forbidden**
- Commit messages, log messages: **forbidden**
- Streamlit page filenames: **forbidden** (use `1_Training.py`)
- Docs (.md): minimal, only if it genuinely aids scanning

## Git

- Conventional Commits: `<type>(<scope>): <summary>`
- Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`
- Branches: `<type>/<short-description>`

## Tooling

- Formatter: `ruff format`
- Linter: `ruff check`
- Tests: `pytest`
- Config validation: Pydantic v2
