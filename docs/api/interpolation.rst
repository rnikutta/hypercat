N-dimensional interpolation
===========================

.. automodule:: hypercat.ndiminterpolation
   :no-members:

This module provides :class:`NdimInterpolation`, a general-purpose
rectilinear-grid interpolator that operates in *image-coordinate space* rather
than in the physical parameter space. The mapping from real parameter values to
fractional pixel positions along each axis is performed internally, allowing the
standard ``scipy.ndimage.map_coordinates`` routine to do the actual
interpolation.

The interpolator supports:

* **Multi-linear interpolation** (``order=1``, default) — fast and usually
  sufficient.
* **Cubic-spline interpolation** (``order=3``) — slower; use only when
  accuracy matters more than speed.
* **Log-space interpolation** (``mode='log'``) — takes ``log10`` of the data
  before interpolating and exponentiates afterwards. Strongly recommended when
  the data span many orders of magnitude (the typical case for model image
  brightness).

.. contents:: On this page
   :local:
   :depth: 2


NdimInterpolation
-----------------

.. autoclass:: hypercat.ndiminterpolation.NdimInterpolation
   :members:
   :special-members: __call__, __init__
   :member-order: bysource

.. rubric:: How the coordinate mapping works

Each axis of the hypercube is sampled at the values stored in ``theta[k]``.
These are not necessarily uniformly spaced. A 1-D interpolator is built
for each axis mapping real parameter values → fractional pixel indices.
Given a query vector :math:`v`, the method :meth:`get_coords` produces the
full coordinate matrix needed by ``map_coordinates``.

When the query vector contains *tuples* of values for one or more axes, the
result is an array of interpolated images whose shape reflects the product of
all tuple lengths.

.. rubric:: Example

.. code-block:: python

   import numpy as np
   from hypercat.ndiminterpolation import NdimInterpolation

   # 3-D toy example: data sampled on a 4×5×6 rectilinear grid
   theta = [np.linspace(0, 1, 4),
            np.linspace(0, 2, 5),
            np.linspace(0, 3, 6)]
   shape = tuple(len(t) for t in theta)
   data  = np.random.rand(*shape)

   ip = NdimInterpolation(data, theta, order=1, mode='linear')

   # Interpolate at a single point
   val = ip((0.5, 1.0, 1.5))

   # Interpolate over a grid: axis-0 varies, axes 1 and 2 fixed
   vals = ip(((0.0, 0.5, 1.0), 1.0, 1.5))
   print(vals.shape)  # (3,)
