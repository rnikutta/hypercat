Core — model cube and source
============================

The two classes described here form the primary interface for most Hypercat
workflows. :class:`~hypercat.hypercat.ModelCube` wraps the HDF5 data file and
the N-dimensional interpolator; :class:`~hypercat.hypercat.Source` adds
physical scales and returns calibrated :class:`~hypercat.imageops.Image`
instances.

.. contents:: On this page
   :local:
   :depth: 2


ModelCube
---------

.. autoclass:: hypercat.hypercat.ModelCube
   :members:
   :special-members: __call__, __init__
   :member-order: bysource

The constructor accepts either *onthefly* mode (default — minimal RAM usage,
loads only the needed slab per call) or an *interactive* terminal dialog for
pre-loading a sub-cube into RAM for faster repeated access.

.. rubric:: Sub-cube selection modes

``'onthefly'``
    The full hypercube is memory-mapped but never fully loaded. Each call to
    :meth:`~hypercat.hypercat.ModelCube.get_image` selects the minimal spanning
    slab on the fly. This is the recommended mode for exploratory work.

``'interactive'``
    A terminal UI (powered by *urwid*) lets you select a subset of parameter
    values for each axis. The resulting sub-cube is loaded into RAM. Store the
    index lists to a JSON file with ``subcube_selection_save`` for re-use.

``path/to/selection.json``
    Directly supply a JSON file of index lists produced by a previous
    interactive session.


Source
------

.. autoclass:: hypercat.hypercat.Source
   :members:
   :special-members: __call__, __init__
   :member-order: bysource

.. rubric:: Physical quantities

The dust sublimation radius is computed with Eq. (1) from Nenkova et al.
(2008b):

.. math::

   R_d = 0.4 \left(\frac{L}{10^{45}\,\text{erg s}^{-1}}\right)^{1/2}
             \left(\frac{1500\,\text{K}}{T_\text{sub}}\right)^{2.6} \text{pc}

The pixel scale then follows from :math:`R_d`, the torus extent *Y*, and the
angular diameter distance.


Module-level helpers
---------------------

.. autofunction:: hypercat.hypercat.get_Rd
.. autofunction:: hypercat.hypercat.get_pixelscale
.. autofunction:: hypercat.hypercat.get_sed_from_fitsfile
.. autofunction:: hypercat.hypercat.get_clean_file_list
.. autofunction:: hypercat.hypercat.mirror_fitsfile
.. autofunction:: hypercat.hypercat.mirror_all_fitsfiles
