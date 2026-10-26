# Contributing

Thanks for helping. Bug reports, small fixes and new examples are all welcome.

## Set up

```bash
git clone https://github.com/prashant7799/placebokit
cd placebokit
uv venv .venv --python 3.11
uv pip install -p .venv -e ".[dev]"
```

(`python -m venv .venv` and `pip install -e ".[dev]"` work too.)

## Before you open a pull request

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
```

Both must pass. The full suite takes under a minute, mostly the examples and the
calibration simulations.

## Ground rules

- Honest p-values are the point of this package. A change to a control or to the p-value
  needs a test that shows it is still calibrated, ideally a seeded null simulation.
- Tests and examples use synthetic data. Do not commit real or private data. The `data/`
  folder is ignored on purpose.
- Tests must not use the network.
- If you change an example, update its output in `README.md`. A test checks that they match.
- Prefer plain, readable code over clever code. Short functions, clear names, a comment where
  the reason is not obvious.
- Add a line to `CHANGELOG.md` for anything a user would notice.
