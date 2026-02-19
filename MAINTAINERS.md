# Maintainer guide

This document is for the repository owner. It covers the one-time
infrastructure setup required before the publish workflows will succeed.

---

## CI/CD secrets

All GitHub secrets are added via:

> **GitHub repo → Settings → Secrets and variables → Actions → New repository secret**

Enter each name exactly as shown below and paste the corresponding value.

---

### `ANACONDA_TOKEN`

**Used by:** `publish-to-conda.yml`

Authorises the CI runner to upload packages to your
[anaconda.org](https://anaconda.org) channel.

**How to obtain:**

1. Log in to <https://anaconda.org>.
2. Click your avatar (top-right) → **Settings** → **Access**.
3. Scroll to **Access Tokens** and click **+ Add**.
4. Fill in a description (e.g. `hypercat-ci`).
5. Tick **Allow write access to API site** and **Allow uploads to PyPI servers**.
6. Click **Create** and **copy the token immediately** — it is only shown once.

**Add to GitHub:** name `ANACONDA_TOKEN`, value = the token string.

---

### `DOCKERHUB_USERNAME`

**Used by:** `publish-to-dockerhub.yml`

Your Docker Hub account username (e.g. `rnikutta`). Stored as a secret so the
workflow YAML stays generic and free of hardcoded names.

**How to obtain:**

1. Log in to <https://hub.docker.com>.
2. Your username is shown in the top-right corner of the page.

**Add to GitHub:** name `DOCKERHUB_USERNAME`, value = your username.

---

### `DOCKERHUB_TOKEN`

**Used by:** `publish-to-dockerhub.yml`

A Docker Hub access token that grants the CI runner permission to push images.
Using a token (rather than your password) lets you revoke CI access
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
credential automatically at publish time, so nothing needs to be stored as a
GitHub secret.

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

## Cutting a release

1. Ensure all tests pass on `main` and the changelog is up to date.
2. Create and push a version tag (the version is derived from git tags via
   `setuptools_scm`):
   ```bash
   git tag v1.2.3
   git push origin v1.2.3
   ```
3. On GitHub, go to **Releases → Draft a new release**, select the tag, write
   release notes, and click **Publish release**.
4. The three publish workflows (`publish-to-pypi.yml`, `publish-to-conda.yml`,
   `publish-to-dockerhub.yml`) will fire automatically.
