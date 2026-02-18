PSF modelling
=============

.. automodule:: hypercat.psf
   :no-members:

This module provides the :class:`PSF` class and the routines used to construct
or load point-spread functions for single-dish telescope simulations.

The default PSF model is a **Gaussian core + Airy disk** mixture:

.. math::

   \text{PSF} = S \cdot A + (1 - S) \cdot G

where :math:`A` is the Airy disk from a circular aperture of diameter *D* at
wavelength *λ*, :math:`G` is a Gaussian approximation to the seeing disk,
and :math:`S` is the Strehl ratio.

.. contents:: On this page
   :local:
   :depth: 2


PSF class
---------

.. autoclass:: hypercat.psf.PSF
   :members:
   :special-members: __init__
   :member-order: bysource

.. rubric:: Convolution

:meth:`~hypercat.psf.PSF.convolve` uses ``scipy.signal.fftconvolve`` for
efficient FFT-based convolution and normalises the result to conserve total
flux.

:meth:`~hypercat.psf.PSF.deconvolve` applies the Richardson–Lucy algorithm
(a maximum-likelihood iterative method) for the inverse problem.


Factory functions
-----------------

.. autofunction:: hypercat.psf.getPSF
.. autofunction:: hypercat.psf.modelPSF
.. autofunction:: hypercat.psf.loadPSFfromFITS
.. autofunction:: hypercat.psf.getPupil
.. autofunction:: hypercat.psf.fft_pxscale
.. autofunction:: hypercat.psf.get_normalization
