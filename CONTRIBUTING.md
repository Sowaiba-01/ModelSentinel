# Contributing to ModelSentinel

Thanks for your interest in improving ModelSentinel! 🎉

## Development setup

```bash
git clone https://github.com/sowaiba/modelsentinel.git
cd modelsentinel
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Before you open a pull request

1. **Tests pass:** `pytest`
2. **Lint is clean:** `ruff check .`
3. **New behaviour is covered by tests** in `tests/`.
4. **Public functions have docstrings** and type hints.

## Guidelines

- Keep dependencies minimal (numpy / pandas / scikit-learn / scipy only, unless discussed in an issue first).
- One logical change per pull request.
- Every scoring function returns a bounded value in `[0, 100]` for consistency.
- Open an issue to discuss large features before building them.

## Reporting bugs

Open an issue with a minimal reproducible example, your Python version, and the full traceback.
