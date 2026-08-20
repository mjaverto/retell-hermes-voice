# Contributing

## Development setup

```sh
uv venv
uv pip install -e '.[dev]'
cp .env.example .env   # only needed to run the server; tests construct Settings directly
```

Python >= 3.11. The project uses hatchling for builds and a `src/` layout.

## Quality gates

All four must pass before a PR is mergeable (CI runs them on 3.11 and 3.12):

```sh
ruff format --check src tests
ruff check src tests
mypy
pytest
```

`mypy` runs in strict mode against the `retell_hermes_voice` package (configured in
`pyproject.toml`). Ruff line length is 100.

## Test conventions

- **Everything runs offline.** No real network, no live Retell or Hermes calls, no
  credentials in CI — ever. Tests that need a Hermes backend use the programmable
  fake in `tests/fake_hermes.py`, mounted with `httpx.ASGITransport` into
  `HermesClient` or `server.create_app(settings, hermes_transport=...)`.
- Construct `Settings` directly in tests (with `_env_file=None` so an ambient `.env`
  can't leak in) and always set `warmup_on_start=False`.
- Deterministic and fast: tune timeout settings (e.g. `filler_after_seconds=0.05`)
  instead of sleeping; no sleep longer than 0.3 s.
- Use `+1555…` fixture phone numbers; never real numbers or secrets.
- `pytest-asyncio` runs in auto mode — plain `async def test_*` functions work.

## Pull requests

- Focused, atomic commits; one logical change per PR.
- Behavior changes come with tests that fail without the change.
- No secrets, tokens, real phone numbers, or personal data in code, tests, or
  fixtures.
- Update docs when you change configuration, endpoints, or wire behavior.

## Wire contracts

The adapter's external behavior is pinned by two documents — read them before
touching protocol code:

- [`docs/integration-contracts.md`](docs/integration-contracts.md) — the verified
  Retell Custom LLM and Hermes API wire contract (frame shapes, latency numbers,
  failure modes, with verification class per claim).
- [`docs/INTERFACES.md`](docs/INTERFACES.md) — the internal cross-module contract
  (signatures and behavior requirements per module).

If observed reality diverges from `integration-contracts.md`, fix the doc in the
same PR and mark the claim's verification class.
