Units
=====

.. automodule:: hypercat.units
   :no-members:

Unit parsing, validation, and conversion utilities used throughout Hypercat.
All physical quantities are passed as strings (e.g. ``'1e45 erg/s'``,
``'10 micron'``) and parsed into :class:`astropy.units.Quantity` objects by
the functions in this module.

.. contents:: On this page
   :local:
   :depth: 2


Recognised units
----------------

.. autodata:: hypercat.units.UNITS

The ``UNITS`` dictionary maps category names to tuples of recognised unit
strings. The categories are:

``'ANGULAR'``
    ``arcsec``, ``mas``, ``uarcsec``, ``deg``, ``rad``, ``arcmin``

``'LINEAR'``
    ``cm``, ``m``, ``km``, ``AU``, ``pc``, ``kpc``, ``Mpc``, ``Gpc``

``'TEMPERATURE'``
    ``K``

``'WAVE'``
    ``Angstrom``, ``nm``, ``micron``, ``mm``, ``cm``, ``m``

``'LUMINOSITY'``
    ``erg/s``, ``W``, ``Lsun``

``'BRIGHTNESS'``
    ``Jy/pix``, ``mJy/pix``

``'FLUXDENSITY'``
    ``Jy``, ``mJy``, ``MJy``, ``uJy``, ``nJy``

``'CUNITS'``
    Union of ``ANGULAR`` and ``LINEAR`` (used for pixel-scale arguments
    that can be either angular or linear).

.. autofunction:: hypercat.units.list_recognized_units


Parsing functions
-----------------

.. autofunction:: hypercat.units.getQuantity
.. autofunction:: hypercat.units.getValueUnit

.. rubric:: Usage examples

.. code-block:: python

   from hypercat.units import getQuantity, getValueUnit, UNITS

   # Parse a string into an astropy Quantity
   q = getQuantity('1e45 erg/s', UNITS['LUMINOSITY'])
   print(q)          # 1e+45 erg / s

   # Split into value and unit object
   val, unit = getValueUnit('14.4 Mpc', UNITS['LINEAR'])
   print(val)        # 14.4
   print(unit)       # Mpc
