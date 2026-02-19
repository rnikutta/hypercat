# Contributing to Hypercat

Thank you for your interest in contributing! This document covers everything
needed to set up a development environment, run the test suite, work with CI,
and get a pull request merged.

---

## Development environment

Clone the repository and install in editable mode with the full developer
extras:

```bash
git clone https://github.com/rnikutta/hypercat.git
cd hypercat
pip install -e ".[dev]"
pre-commit install
```

The `[dev]` extra adds `pytest`, `pytest-cov`, `pytest-mock`, `pre-commit`,
and `jupyter`. The `pre-commit install` step installs git hooks that run
linters and formatters automatically before every commit.

### tkinter on headless systems

The interactive GUI (`hypercatgui`) and the `pickfile()` utility in
`hypercat.utils` require `tkinter`. On Ubuntu/Debian servers without a
display this must be installed separately:

```bash
sudo apt-get install python3-tk
```

On systems where `tkinter` is genuinely unavailable the core `hypercat`
library still imports and works correctly; only the GUI entry point
(`hypercatgui`) is affected.

---

## Running the tests

```bash
pytest                        # all tests
pytest -x                     # stop on first failure
pytest --cov=hypercat         # with coverage report
pytest tests/test_units.py    # single module
```

Tests that require `tkinter` or `urwid` are automatically skipped when those
libraries are not present.

---

## Building the documentation locally

```bash
pip install sphinx sphinx-rtd-theme sphinx-copybutton
cd docs
make html
# output → docs/_build/html/index.html
```

---

## CI/CD workflows

| Workflow file | Trigger | Purpose |
|---|---|---|
| `testing-and-coverage.yml` | push / PR to `main` | Unit tests on Python 3.10–3.13, Codecov upload |
| `build-documentation.yml`  | push / PR to `main` | Sphinx HTML build check |
| `pre-commit-ci.yml`        | push / PR to `main` | Pre-commit hook lint pass |
| `smoke-test.yml`           | daily cron + manual | Full test matrix on latest deps |
| `publish-to-pypi.yml`      | GitHub Release      | Build sdist/wheel, upload to PyPI via OIDC |
| `publish-to-conda.yml`     | GitHub Release      | Build noarch conda pkg, upload to Anaconda.org |
| `publish-to-dockerhub.yml` | GitHub Release + push to `main` | Multi-arch Docker image, push to Docker Hub |

The first four run automatically on every pull request. The publish workflows
only fire on a GitHub Release and require repository secrets configured by the
maintainer — see [MAINTAINERS.md](./MAINTAINERS.md).

---

## How to contribute

1. **Fork** the repository and create a feature branch off `main`.
2. **Install** in development mode (see above) and ensure all tests pass.
3. **Write tests** for any new or changed functionality.
4. **Run pre-commit** manually if needed: `pre-commit run --all-files`
5. **Open a pull request** against `main` with a clear title and description
   of what changed and why.

Please keep pull requests focused — one logical change per PR makes review
faster and keeps the git history clean.

---

## Bug reports and feature requests

Please open an issue on GitHub:
<https://github.com/rnikutta/hypercat/issues>

For **bug reports**, include:
- Hypercat version: `python -c "import hypercat; print(hypercat.__version__)"`
- Python version and OS
- Minimal reproducible example

For **feature requests**, describe the use case and, if possible, sketch the
desired API.
