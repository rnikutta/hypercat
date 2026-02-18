Overview
========

What is Hypercat?
-----------------

Hypercat is a Python toolkit for operating on the **CLUMPY** hypercubes of AGN
(Active Galactic Nuclei) torus model images produced by the CLUMPY radiative
transfer code (Nenkova et al. 2008). The physical model describes the infrared
emission from a dusty, clumpy torus surrounding a central AGN.

The CLUMPY image data span a six-dimensional parameter space:

========  ======================================  ==================
Symbol    Parameter                               Typical range
========  ======================================  ==================
*i*       Inclination angle (degrees)             0 – 90
*σ*       Torus width (degrees)                   15 – 70
*Y*       Radial extent (dust cloud cloud ratio)  5 – 30
*N₀*      Number of clouds along the equator      1 – 15
*q*       Radial cloud distribution index         0 – 3
*τᵥ*      Individual cloud optical depth          5 – 150
*λ*       Wavelength (μm)                         1.2 – 1000
========  ======================================  ==================

The full hypercube contains hundreds of thousands of images. Hypercat
memory-maps the HDF5 data file so that only the small slab required for a
given interpolation is loaded into RAM, keeping memory use manageable even
for the complete dataset (~400 GB).

Architecture
------------

The package is structured in three conceptual layers:

**High-level (physical)**

:class:`~hypercat.hypercat.ModelCube`
    Wraps the HDF5 hypercube, manages sub-cube selection, and exposes an
    N-dimensional interpolator. Calling an instance with a parameter vector
    returns an interpolated image array.

:class:`~hypercat.hypercat.Source`
    Attaches physical scales (dust sublimation radius, distance, position
    angle, WCS) to a ``ModelCube``. Returns calibrated
    :class:`~hypercat.imageops.Image` objects with flux-density units.

:class:`~hypercat.obsmodes.Imaging`
    Simulates single-dish telescope observations: PSF convolution, optional
    noise, and resampling to detector pixel scale.

:class:`~hypercat.obsmodes.Interferometry`
    Computes correlated fluxes and visibilities at user-supplied UV
    baselines via 2-D FFT.

**Mid-level (images)**

:class:`~hypercat.imageops.ImageFrame`
    Square image with pixel scale, field-of-view, rotation, and resampling.

:class:`~hypercat.imageops.Image`
    Extends ``ImageFrame`` with brightness calibration (Jy/arcsec², etc.)
    and noise injection.

:class:`~hypercat.psf.PSF`
    Extends ``ImageFrame`` with convolution and Richardson-Lucy deconvolution.

**Low-level (utilities)**

* :mod:`~hypercat.ndiminterpolation` – N-dimensional interpolation on
  rectilinear grids
* :mod:`~hypercat.morphology` – image moments, Gini coefficient, eigenanalysis
* :mod:`~hypercat.interferometry` – FFT, correlated flux, UV utilities
* :mod:`~hypercat.ioops` – FITS, HDF5, JSON I/O helpers
* :mod:`~hypercat.units` – unit parsing and conversion
* :mod:`~hypercat.plotting` – multi-panel Matplotlib helpers
* :mod:`~hypercat.utils` – WCS construction, array utilities

Citation
--------

If you use Hypercat in your research, please cite:

* Nikutta, Lopez-Rodriguez, Ichikawa et al. (2021),
  *Hypercubes of AGN Tori (Hypercat) — I. Models and Image Morphology*,
  ApJ.

* Nikutta, Lopez-Rodriguez, Ichikawa et al. (2021),
  *Hypercubes of AGN Tori (Hypercat) — II. Resolving the torus with
  Extremely Large Telescopes*, ApJ.
