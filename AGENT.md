# ClauseMiner Agent Notes

- `clause_miner.py`: Main entry point for clause extraction utilities.
- `pyproject.toml`: Poetry project definition and dependencies.
- `uv.lock`: Lockfile managed by `uv`.
- `elements.json`: Reference data consumed by the miner.
- `JP4743919B2.txt`, `JP6469758B2.pdf`, `JP6469758B2_formatted.txt`: Sample patent documents for testing.
- When making changes, prefer shell tools over running Python; run Python only when strictly necessary.
