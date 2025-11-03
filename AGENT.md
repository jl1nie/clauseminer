# ClauseMiner Agent Notes

- `clause_miner.py`: Main entry point for clause extraction utilities.
- `pyproject.toml`: Poetry project definition and dependencies.
- `uv.lock`: Lockfile managed by `uv`.
- `elements.json`: Reference data consumed by the miner.
- Sample patent documents now live under `tests/fixtures/docs/` (e.g., `tests/fixtures/docs/JP4743919B2.txt`).
- When making changes, prefer shell tools over running Python; run Python only when strictly necessary.
