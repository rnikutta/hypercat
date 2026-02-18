I/O operations
==============

.. automodule:: hypercat.ioops
   :no-members:

File I/O helpers for FITS, HDF5, and JSON data. Also contains
:class:`CheckListSelector`, a terminal-based interactive widget used by
:class:`~hypercat.hypercat.ModelCube` in ``'interactive'`` sub-cube selection
mode.

.. contents:: On this page
   :local:
   :depth: 2


FitsFile
--------

.. autoclass:: hypercat.ioops.FitsFile
   :members:
   :special-members: __init__
   :member-order: bysource

A thin wrapper around ``astropy.io.fits`` providing named access to HDU headers
and data, including support for binary table record arrays.


Interactive sub-cube selector
------------------------------

.. autoclass:: hypercat.ioops.CheckListSelector
   :members:
   :special-members: __init__
   :member-order: bysource

This class launches a ``urwid``-based terminal UI. It requires ``urwid`` to be
installed. In practice it is invoked via the higher-level
:func:`getIndexLists` function.


HDF5 helpers
------------

.. autofunction:: hypercat.ioops.make_subcube_hdf
.. autofunction:: hypercat.ioops.storeCubeToHdf5
.. autofunction:: hypercat.ioops.memmap_hdf5_dataset
.. autofunction:: hypercat.ioops.get_hyperslab_via_mesh


FITS helpers
------------

.. autofunction:: hypercat.ioops.save2fits


JSON helpers
------------

.. autofunction:: hypercat.ioops.storejson
.. autofunction:: hypercat.ioops.loadjson


Sub-cube selection
------------------

.. autofunction:: hypercat.ioops.getIndexLists
.. autofunction:: hypercat.ioops.get_bytesize
.. autofunction:: hypercat.ioops.get_bytes_human
.. autofunction:: hypercat.ioops.isragged
