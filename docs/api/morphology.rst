Morphological analysis
======================

.. automodule:: hypercat.morphology
   :no-members:

The :mod:`~hypercat.morphology` module provides a comprehensive set of tools
for characterising the spatial structure of 2-D images: image moments
(raw, central, scale-invariant), covariance-based shape descriptors, the Gini
coefficient, and helper functions for generating synthetic test images.

.. contents:: On this page
   :local:
   :depth: 2


Moment classes
--------------

Moment
^^^^^^

.. autoclass:: hypercat.morphology.Moment
   :members:
   :special-members: __call__, __init__
   :member-order: bysource

Calling an instance with orders ``(p, q)`` computes the raw, central, and
scale-invariant moment of order ``(p, q)``, as well as the covariance matrix,
its eigenvalues, the source elongation, and the position angle.

MomentAnalytics
^^^^^^^^^^^^^^^

.. autoclass:: hypercat.morphology.MomentAnalytics
   :members:
   :special-members: __call__, __init__
   :member-order: bysource


Moment computation
------------------

The functions below are the fast, vectorised moment routines used internally
(implemented with ``np.tensordot`` / matrix multiplication; ~14× faster than
a naïve nested loop).

.. autofunction:: hypercat.morphology.get_moment_raw_matmul
.. autofunction:: hypercat.morphology.get_all_moments_raw_matmul
.. autofunction:: hypercat.morphology.get_moment_central_matmul
.. autofunction:: hypercat.morphology.get_cov_from_moments
.. autofunction:: hypercat.morphology.get_centroid
.. autofunction:: hypercat.morphology.get_rgyr
.. autofunction:: hypercat.morphology.get_eigenvalues
.. autofunction:: hypercat.morphology.get_elongation
.. autofunction:: hypercat.morphology.get_angle
.. autofunction:: hypercat.morphology.halflight_radius

The scalar convenience wrappers below call the routines above:

.. autofunction:: hypercat.morphology.get_moment
.. autofunction:: hypercat.morphology.get_moment_raw
.. autofunction:: hypercat.morphology.get_moment_central
.. autofunction:: hypercat.morphology.get_moment_scaleinvariant


Concentration / Gini statistics
---------------------------------

.. autofunction:: hypercat.morphology.gini
.. autofunction:: hypercat.morphology.gini_pure


Geometric shape generators
---------------------------

These functions create synthetic 2-D images useful for testing morphological
metrics:

.. autofunction:: hypercat.morphology.gaussian
.. autofunction:: hypercat.morphology.gaussian_asymmetric
.. autofunction:: hypercat.morphology.circle
.. autofunction:: hypercat.morphology.square
.. autofunction:: hypercat.morphology.rectangle


Vector and orientation utilities
---------------------------------

.. autofunction:: hypercat.morphology.rotateVector
.. autofunction:: hypercat.morphology.rot90ccw
.. autofunction:: hypercat.morphology.whichside
.. autofunction:: hypercat.morphology.getImageEigenvectors
.. autofunction:: hypercat.morphology.imageToEigenvectors
.. autofunction:: hypercat.morphology.getUnitVector
.. autofunction:: hypercat.morphology.getAngle
.. autofunction:: hypercat.morphology.get_cutout
.. autofunction:: hypercat.morphology.mask_outside_square
.. autofunction:: hypercat.morphology.findEmissionCentroid
.. autofunction:: hypercat.morphology.findEmissionPA
