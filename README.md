# HYPERCAT — Hypercubes of AGN Tori

[![PyPI](https://img.shields.io/pypi/v/hypercat?label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/hypercat/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/hypercat?label=conda-forge&logo=conda-forge&logoColor=white)](https://anaconda.org/conda-forge/hypercat)
[![Docker Hub](https://img.shields.io/docker/v/rnikutta/hypercat?label=Docker%20Hub&logo=docker&logoColor=white&sort=semver)](https://hub.docker.com/r/rnikutta/hypercat)
[![Documentation](https://readthedocs.org/projects/hypercat/badge/?version=latest)](https://hypercat.readthedocs.io/en/latest/)
[![CI](https://github.com/rnikutta/hypercat/actions/workflows/testing-and-coverage.yml/badge.svg)](https://github.com/rnikutta/hypercat/actions/workflows/testing-and-coverage.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?logo=python&logoColor=white)](https://github.com/rnikutta/hypercat/actions/workflows/testing-and-coverage.yml)
[![Template](https://img.shields.io/badge/template-LINCC%20Frameworks-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

![Hypercat images at 2.2 and 30 micron, and their composite](./rgb.png)

*Hypercat images at 2.2 μm (blue) and 30 μm (gold), and their false-colour composite.*

## What is Hypercat?

Hypercat is a Python toolkit for working with the **CLUMPY** hypercubes of AGN
torus model images (Nenkova et al. 2008). The CLUMPY models describe infrared
emission from a dusty, clumpy torus surrounding an active galactic nucleus. The
image hypercube spans a seven-dimensional parameter space (inclination, torus
width, radial extent, cloud number, cloud distribution, optical depth,
wavelength) and contains hundreds of thousands of pre-computed images.

Hypercat provides:

- **N-dimensional interpolation** — retrieve a model image at any parameter
  combination via multi-linear or cubic-spline interpolation, with
  memory-mapped HDF5 access (only the needed slab is loaded into RAM)
- **Physical calibration** — attach luminosity, distance, and position angle to
  obtain images in physical brightness units (Jy/arcsec², etc.) with a WCS
- **Single-dish observations** — convolve with a Gaussian + Airy PSF, add
  noise, resample to detector pixel scale
- **Interferometric observations** — compute correlated fluxes and
  visibilities at arbitrary UV baselines via 2-D FFT
- **Morphological analysis** — image moments, covariance, Gini coefficient,
  eigenanalysis, half-light radius
- **Interactive GUI** — point-and-click exploration of the parameter space
  (requires `tkinter`)

## Installation

### pip

```bash
pip install hypercat
```

The core library works on any system regardless of whether `tkinter` is
installed. The interactive GUI (`hypercatgui`) additionally requires `tkinter`,
which is usually bundled with CPython but may need a separate install on
headless Linux systems:

```bash
sudo apt-get install python3-tk   # Ubuntu / Debian
```

### conda-forge

```bash
conda install -c conda-forge hypercat
```

Or with [mamba](https://mamba.readthedocs.io) for a faster solve:

```bash
mamba install -c conda-forge hypercat
```

The conda-forge package ships binary builds of all dependencies and installs
`tkinter` automatically on most platforms, so the GUI works out of the box.

### Docker

A self-contained image with Hypercat and JupyterLab is available from Docker
Hub — no local Python environment required:

```bash
docker run --rm -p 8888:8888 \
    -v /path/to/clumpy/data:/data \
    -v $(pwd):/work \
    rnikutta/hypercat
```

Open <http://localhost:8888> in your browser. Data files mounted at `/data/`
are accessible from all notebooks. To build the image locally:

```bash
git clone https://github.com/rnikutta/hypercat.git
cd hypercat
docker build -t hypercat .
```

### Development install

```bash
git clone https://github.com/rnikutta/hypercat.git
cd hypercat
pip install -e ".[dev]"
pre-commit install
```

## Quick start

```python
import hypercat as hc

# Memory-map the CLUMPY image hypercube (loads slabs on demand)
cube = hc.ModelCube('hypercat_20200830_all.hdf5')
cube.print_sampling()   # show parameter names and sampled values

# Interpolate an image: (i, sigma, Y, N0, q, tau_v, lambda_micron)
image_array = cube((30., 30., 10., 5., 1., 40., 10.))

# Attach physical scales to get a calibrated Image
src = hc.Source(cube, luminosity='1e45 erg/s', distance='14.4 Mpc', pa='90 deg')
img = src((30., 30., 10., 5., 1., 40., 10.), total_flux_density='0.5 Jy')

# Simulate a single-dish observation
telescope = hc.Imaging(psfdict={'psf': 'model', 'diameter': '8.2 m',
                                 'wavelength': '10 micron', 'strehl': 0.9})
observed, psf, _ = telescope(img)

# Multi-panel plot
fig, axes = hc.multiplot([img, observed], titles=['Sky', 'Convolved'],
                          units='Jy/arcsec^2', colorbars=True)
```

See the [Quick start guide](https://hypercat.readthedocs.io/en/latest/quickstart.html)
for more annotated examples.

## Model data files

The CLUMPY image hypercubes are distributed separately (tens to hundreds of
GB). Several files are available, covering different wavelength ranges:

| File | Size gz / raw (GB) | N<sub>wave</sub> | Wavelengths (μm) |
|---|---|---|---|
| `hypercat_20200830_all.hdf5`   | 271 / 913 | 25 | all below |
| `hypercat_20200830_nir.hdf5`   |  44 / 146 |  4 | 1.2, 2.2, 3.5, 4.8 |
| `hypercat_20200830_mir.hdf5`   | 120 / 402 | 11 | 8.7 – 18.5 |
| `hypercat_20200830_fir.hdf5`   |  65 / 219 |  6 | 31.5 – 214 |
| `hypercat_20200830_submm.hdf5` |  42 / 146 |  4 | 350 – 945 |

Download instructions and MD5 checksums are at
<https://www.clumpy.org/images/> (currently mirrored at
`ftp://ftp.tuc.noirlab.edu/pub/nikutta/hypercat/`).

## Documentation

Full documentation is on ReadTheDocs: <https://hypercat.readthedocs.io>

| | |
|---|---|
| [Overview](https://hypercat.readthedocs.io/en/latest/overview.html) | Architecture, parameter table |
| [Installation](https://hypercat.readthedocs.io/en/latest/installation.html) | All installation methods in detail |
| [Quick start](https://hypercat.readthedocs.io/en/latest/quickstart.html) | Annotated code examples |
| [API reference](https://hypercat.readthedocs.io/en/latest/api/core.html) | Full API docs from docstrings |

## Contributing

Contributions of all kinds are welcome — bug fixes, new features, documentation
improvements, and additional tests.

1. Fork the repository and create a feature branch.
2. Install in development mode: `pip install -e ".[dev]" && pre-commit install`
3. Add tests for any new functionality; run the suite with `pytest`.
4. Open a pull request against `main` with a clear description of the change.

## Bug reports and feature requests

Please open an issue on GitHub:
<https://github.com/rnikutta/hypercat/issues>

Include the Hypercat version (`python -c "import hypercat; print(hypercat.__version__)"`),
your Python version, and a minimal reproducible example where applicable.

## Citation

If you use Hypercat in published research, please cite both papers:

- Nikutta, Lopez-Rodriguez, Ichikawa, Levenson, Packham, Hönig, Alonso-Herrero,
  *Hypercubes of AGN Tori (Hypercat) — I. Models and Image Morphology*,
  ApJ 2021, 919, 136.
  [ADS](https://ui.adsabs.harvard.edu/abs/2021arXiv210912123N/abstract)

- Nikutta, Lopez-Rodriguez, Ichikawa, Levenson, Packham, Hönig, Alonso-Herrero,
  *Hypercubes of AGN Tori (Hypercat) — II. Resolving the torus with Extremely
  Large Telescopes*, ApJ 2021.
  [ADS](https://ui.adsabs.harvard.edu/abs/2021arXiv210912130N/abstract)

## License

BSD 3-Clause — see [LICENSE](./LICENSE).

## Authors

Robert Nikutta, Enrique Lopez-Rodriguez, Kohei Ichikawa

---

## CI/CD secrets

The publish workflows require four secrets to be stored in the repository.
All are added the same way:

> **GitHub repo → Settings → Secrets and variables → Actions → New repository secret**
> Enter the name exactly as shown below and paste the value.

The PyPI workflow uses OIDC trusted publishing and needs one extra
one-time setup step on pypi.org instead of a GitHub secret (see below).

---

### `ANACONDA_TOKEN`

**Used by:** `.github/workflows/publish-to-conda.yml`

This token authorises the CI runner to upload packages to your
[anaconda.org](https://anaconda.org) channel.

**How to obtain:**

1. Log in to <https://anaconda.org>.
2. Click your avatar (top-right) → **Settings** → **Access**.
3. Scroll to **Access Tokens** and click **+ Add**.
4. Fill in a description (e.g. `hypercat-ci`).
5. Tick **Allow write access to API site** and
   **Allow uploads to PyPI servers**.
6. Click **Create** and **copy the token immediately** — it is only
   shown once.

**Add to GitHub:** name `ANACONDA_TOKEN`, value = the token string.

---

### `DOCKERHUB_USERNAME`

**Used by:** `.github/workflows/publish-to-dockerhub.yml`

Your Docker Hub account username (e.g. `rnikutta`). It is stored as a
secret only so the workflow can reference it without hardcoding a name
in the YAML.

**How to obtain:**

1. Log in to <https://hub.docker.com>.
2. Your username is shown in the top-right corner of the page.

**Add to GitHub:** name `DOCKERHUB_USERNAME`, value = your username.

---

### `DOCKERHUB_TOKEN`

**Used by:** `.github/workflows/publish-to-dockerhub.yml`

A Docker Hub access token that grants the CI runner permission to push
images. Using a token (rather than your password) means you can revoke
CI access independently without changing your account password.

**How to obtain:**

1. Log in to <https://hub.docker.com>.
2. Click your avatar → **Account Settings → Security**.
3. Click **New Access Token**.
4. Description: `hypercat-ci`. Access permissions: **Read, Write, Delete**.
5. Click **Generate** and **copy the token immediately** — it is only
   shown once.

**Add to GitHub:** name `DOCKERHUB_TOKEN`, value = the token string.

---

### `CODECOV_TOKEN`

**Used by:** `.github/workflows/testing-and-coverage.yml`

Authenticates the coverage upload to [codecov.io](https://codecov.io),
so coverage reports are associated with the correct repository.

**How to obtain:**

1. Go to <https://app.codecov.io> and sign in with your GitHub account.
2. Click **+ Add new repository** and select `rnikutta/hypercat`.
3. Codecov will display a **Repository Upload Token** on the setup page.
   You can also find it later under **Settings → General** for the repo.
4. Copy the token.

**Add to GitHub:** name `CODECOV_TOKEN`, value = the token string.

---

### PyPI — trusted publisher (no GitHub secret needed)

**Used by:** `.github/workflows/publish-to-pypi.yml`

The PyPI workflow uses [OIDC trusted publishing](https://docs.pypi.org/trusted-publishers/)
instead of a long-lived API token. GitHub and PyPI negotiate a
short-lived credential automatically at publish time, so nothing is
stored as a GitHub secret.

**One-time setup on PyPI:**

1. Log in to <https://pypi.org>.
2. Go to **Account Settings → Publishing** (in the left sidebar).
3. Under **Add a new pending publisher**, fill in:
   - **PyPI project name:** `hypercat`
   - **Owner:** `rnikutta`
   - **Repository name:** `hypercat`
   - **Workflow filename:** `publish-to-pypi.yml`
   - **Environment name:** *(leave blank)*
4. Click **Add**.

From then on, every GitHub Release automatically triggers a trusted
upload — no token rotation required.
