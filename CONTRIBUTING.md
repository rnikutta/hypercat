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

---

## CI/CD secrets

The publish workflows require the following secrets to be stored in the
repository. All GitHub secrets are added via:

> **GitHub repo → Settings → Secrets and variables → Actions → New repository secret**
> Enter the name exactly as shown and paste the value.

### `ANACONDA_TOKEN`

**Used by:** `publish-to-conda.yml`

Authorises the CI runner to upload packages to your
[anaconda.org](https://anaconda.org) channel.

**How to obtain:**

1. Log in to <https://anaconda.org>.
2. Click your avatar (top-right) → **Settings** → **Access**.
3. Scroll to **Access Tokens** and click **+ Add**.
4. Fill in a description (e.g. `hypercat-ci`).
5. Tick **Allow write access to API site** and
   **Allow uploads to PyPI servers**.
6. Click **Create** and **copy the token immediately** — it is only shown once.

**Add to GitHub:** name `ANACONDA_TOKEN`, value = the token string.

---

### `DOCKERHUB_USERNAME`

**Used by:** `publish-to-dockerhub.yml`

Your Docker Hub account username (e.g. `rnikutta`). Stored as a secret so
the workflow YAML stays generic and free of hardcoded names.

**How to obtain:**

1. Log in to <https://hub.docker.com>.
2. Your username is shown in the top-right corner.

**Add to GitHub:** name `DOCKERHUB_USERNAME`, value = your username.

---

### `DOCKERHUB_TOKEN`

**Used by:** `publish-to-dockerhub.yml`

A Docker Hub access token that grants the CI runner permission to push images.
Using a token (rather than your password) means you can revoke CI access
independently of your account password.

**How to obtain:**

1. Log in to <https://hub.docker.com>.
2. Click your avatar → **Account Settings → Security**.
3. Click **New Access Token**.
4. Description: `hypercat-ci`. Access permissions: **Read, Write, Delete**.
5. Click **Generate** and **copy the token immediately** — it is only shown once.

**Add to GitHub:** name `DOCKERHUB_TOKEN`, value = the token string.

---

### `CODECOV_TOKEN`

**Used by:** `testing-and-coverage.yml`

Authenticates the coverage upload to [codecov.io](https://codecov.io) so
reports are associated with the correct repository.

**How to obtain:**

1. Go to <https://app.codecov.io> and sign in with your GitHub account.
2. Click **+ Add new repository** and select `rnikutta/hypercat`.
3. Codecov displays a **Repository Upload Token** on the setup page. You can
   also find it later under **Settings → General** for the repository.
4. Copy the token.

**Add to GitHub:** name `CODECOV_TOKEN`, value = the token string.

---

### PyPI — trusted publisher (no GitHub secret needed)

**Used by:** `publish-to-pypi.yml`

The PyPI workflow uses [OIDC trusted publishing](https://docs.pypi.org/trusted-publishers/)
instead of a long-lived API token. GitHub and PyPI negotiate a short-lived
credential automatically at publish time, so nothing is stored as a GitHub
secret.

**One-time setup on PyPI:**

1. Log in to <https://pypi.org>.
2. Go to **Account Settings → Publishing** (left sidebar).
3. Under **Add a new pending publisher**, fill in:
   - **PyPI project name:** `hypercat`
   - **Owner:** `rnikutta`
   - **Repository name:** `hypercat`
   - **Workflow filename:** `publish-to-pypi.yml`
   - **Environment name:** *(leave blank)*
4. Click **Add**.

Every GitHub Release will then trigger a trusted upload automatically — no
token rotation required.

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
