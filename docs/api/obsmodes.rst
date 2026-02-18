Observing modes
===============

.. automodule:: hypercat.obsmodes
   :no-members:

The observing-mode classes simulate the effect of a specific instrument on an
idealised sky brightness distribution (:class:`~hypercat.imageops.Image`).
They are intentionally designed to compose: first create a
:class:`~hypercat.hypercat.Source` image, then pass it to whichever instrument
class you need.

.. contents:: On this page
   :local:
   :depth: 2


Base class
----------

.. autoclass:: hypercat.obsmodes.ObsMode
   :members:
   :special-members: __init__
   :member-order: bysource


Single-dish imaging
-------------------

.. autoclass:: hypercat.obsmodes.Imaging
   :members:
   :special-members: __call__, __init__
   :member-order: bysource

.. rubric:: PSF dictionary keys

The ``psfdict`` argument controls how the PSF is constructed:

``'psf'``
    One of:

    * ``None`` — no PSF is applied (pass-through mode).
    * ``'model'`` — an analytic Gaussian + Airy disk PSF is computed from
      the telescope parameters below.
    * ``'pupil'`` — a pupil-plane PSF is computed from a JSON pupil
      definition file.
    * A file path string ending in ``.fits`` — the PSF is loaded from a FITS
      image.

``'diameter'``
    Telescope aperture diameter, e.g. ``'8.2 m'``. Required for
    ``psf='model'``.

``'wavelength'``
    Observing wavelength, e.g. ``'10 micron'``. Required for ``psf='model'``.

``'strehl'``
    Strehl ratio (0–1). Used together with ``psf='model'`` to scale the PSF
    peak relative to a perfect Airy disk.

``'pixelscale_detector'``
    Angular pixel scale of the detector, e.g. ``'53 mas'``. If provided, the
    convolved image is resampled to this scale after PSF application.

.. rubric:: Return value

:meth:`~hypercat.obsmodes.Imaging.__call__` returns a three-tuple
``(image, psf, psf_resampled)`` where:

* ``image`` — the PSF-convolved (and optionally noise-added) image
* ``psf`` — the PSF at the model pixel scale
* ``psf_resampled`` — the PSF resampled to the detector pixel scale


Interferometry
--------------

.. autoclass:: hypercat.obsmodes.Interferometry
   :members:
   :special-members: __call__, __init__
   :member-order: bysource

.. rubric:: UV point formats

The ``uv`` argument to :meth:`~hypercat.obsmodes.Interferometry.set_uv_points`
accepts:

* A list/tuple of ``(u, v)`` pairs (metres).
* A path to an OIFITS file; the UV coordinates are read from HDU 4.

.. rubric:: Return value

:meth:`~hypercat.obsmodes.Interferometry.__call__` returns
``(corrflux, BL, fftscale)`` where:

* ``corrflux`` — complex correlated flux at each UV point
* ``BL`` — baseline length array (metres)
* ``fftscale`` — conversion factor from FFT pixel to sky angle (rad/pixel)
