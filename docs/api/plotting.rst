Plotting
========

.. automodule:: hypercat.plotting
   :no-members:

Matplotlib helpers for visualising Hypercat :class:`~hypercat.imageops.Image`
objects. The main entry point is :func:`multiplot`, which generates a
publication-ready multi-panel figure from a list of images.

.. contents:: On this page
   :local:
   :depth: 2


Multi-panel figures
-------------------

.. autofunction:: hypercat.plotting.multiplot

.. rubric:: Layout

By default ``multiplot`` arranges images in a single row (``direction='x'``).
To control the layout explicitly, pass ``geometry=(nrows, ncols)``.

.. rubric:: Contours

The ``contours`` argument accepts:

* ``None`` — no contours (default).
* ``'lin'`` — 5 linearly spaced contour levels.
* ``'log'`` — 5 logarithmically spaced contour levels.
* A sequence of explicit contour values.

.. rubric:: Example

.. code-block:: python

   import hypercat as hc
   import matplotlib.pyplot as plt

   # images is a list of hc.Image instances
   fig, axes = hc.multiplot(
       images,
       geometry=(2, 3),
       units='Jy/arcsec^2',
       titles=['i=0°', 'i=30°', 'i=60°', 'i=0°', 'i=30°', 'i=60°'],
       colorbars=True,
       contours='log',
       figtitle='CLUMPY torus images',
   )
   plt.savefig('hypercat_grid.pdf', bbox_inches='tight')


Single-panel helper
-------------------

.. autofunction:: hypercat.plotting.plotPanel


WCS-aware plotting
------------------

.. autofunction:: hypercat.plotting.plot_with_wcs
