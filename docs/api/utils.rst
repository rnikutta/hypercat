Utilities
=========

.. automodule:: hypercat.utils
   :no-members:

General-purpose helper functions used across the package.

.. contents:: On this page
   :local:
   :depth: 2


WCS construction
----------------

.. autofunction:: hypercat.utils.get_wcs

If :attr:`~hypercat.imageops.Image.objectname` is a resolvable source name
(e.g. ``'NGC 1068'``), the real equatorial coordinates are looked up via the
`Vizier <https://vizier.u-strasbg.fr>`_ service and embedded in the WCS.
Otherwise a generic WCS centred on ``(RA, Dec) = (0, 0)`` is used.


Array utilities
---------------

.. autofunction:: hypercat.utils.arrayify
.. autofunction:: hypercat.utils.mirror_axis
.. autofunction:: hypercat.utils.seq2str


File utilities
--------------

.. autofunction:: hypercat.utils.get_rootdir
.. autofunction:: hypercat.utils.pickfile
