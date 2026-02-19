Image operations
================

.. automodule:: hypercat.imageops
   :no-members:

This module defines the image class hierarchy and a collection of standalone
image-processing helpers.

.. contents:: On this page
   :local:
   :depth: 2


Class hierarchy
---------------

.. code-block:: text

   ImageFrame          – square pixel array + pixel scale / FOV / rotation
       └── Image       – adds brightness calibration in physical flux units
   ImageFrame
       └── PSF         – (defined in psf.py) adds convolution / deconvolution


ImageFrame
----------

.. autoclass:: hypercat.imageops.ImageFrame
   :members:
   :special-members: __init__
   :member-order: bysource

.. rubric:: Key attributes

``npix``
    Number of pixels along one side (image must be square and odd).

``pixelscale``
    Angular size of one pixel (``astropy.units.Quantity``).

``pixelarea``
    Solid angle of one pixel (``pixelscale²``).

``FOV``
    Total angular field of view (``npix × pixelscale``).

``data``
    Image array with relative brightness (dimensionless, summing to 1 for
    a normalised model image).

``I``
    Property that returns the *transposed* image array (x → columns,
    y → rows, consistent with the RA/Dec convention).


Image
-----

.. autoclass:: hypercat.imageops.Image
   :members:
   :special-members: __call__, __init__
   :member-order: bysource

:class:`Image` extends :class:`ImageFrame` with physical brightness
calibration. After construction, the image brightness is set so that the
total flux density within the field of view equals ``total_flux_density``.

.. rubric:: Brightness units

Any *per-solid-angle* unit recognised by astropy can be used, for example:

* ``'Jy/arcsec^2'``
* ``'Jy/mas^2'``
* ``'mJy/arcsec^2'``

The total flux density is retrieved with :attr:`~hypercat.imageops.Image.F`
(an alias for :meth:`~hypercat.imageops.Image.getTotalFluxDensity`).


Standalone helpers
------------------

Noise
^^^^^

.. autofunction:: hypercat.imageops.add_noise
.. autofunction:: hypercat.imageops.measure_snr

Image manipulation
^^^^^^^^^^^^^^^^^^

.. autofunction:: hypercat.imageops.rotateImage
.. autofunction:: hypercat.imageops.resampleImage
.. autofunction:: hypercat.imageops.makepositive
.. autofunction:: hypercat.imageops.thresholding
.. autofunction:: hypercat.imageops.trim_square
.. autofunction:: hypercat.imageops.trim_square_odd
.. autofunction:: hypercat.imageops.radial_profile

Validation helpers
^^^^^^^^^^^^^^^^^^

.. autofunction:: hypercat.imageops.checkImage
.. autofunction:: hypercat.imageops.checkInt
.. autofunction:: hypercat.imageops.checkOdd
.. autofunction:: hypercat.imageops.checkEven
.. autofunction:: hypercat.imageops.check2d
.. autofunction:: hypercat.imageops.checkSquare
