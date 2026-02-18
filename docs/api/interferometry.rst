Interferometry utilities
========================

.. automodule:: hypercat.interferometry
   :no-members:

Low-level utilities for optical/infrared interferometry: reading OIFITS files,
computing baseline geometry, performing 2-D FFTs of model images, and
extracting complex correlated fluxes at arbitrary UV points.

These functions are used internally by :class:`~hypercat.obsmodes.Interferometry`
but can also be called directly for custom workflows.

.. contents:: On this page
   :local:
   :depth: 2


FFT and correlated flux
-----------------------

.. autofunction:: hypercat.interferometry.ima2fft
.. autofunction:: hypercat.interferometry.correlatedflux
.. autofunction:: hypercat.interferometry.ima_ifft
.. autofunction:: hypercat.interferometry.fft_pxscale
.. autofunction:: hypercat.interferometry.fft_pixelscale


UV-plane geometry
-----------------

.. autofunction:: hypercat.interferometry.get_BLPhi
.. autofunction:: hypercat.interferometry.get_uv


OIFITS I/O
----------

.. autofunction:: hypercat.interferometry.load_uv
.. autofunction:: hypercat.interferometry.uvload
.. autofunction:: hypercat.interferometry.getObsPerWave


Visualisation
-------------

.. autofunction:: hypercat.interferometry.plot_inter
