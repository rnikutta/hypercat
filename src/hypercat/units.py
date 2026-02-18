__version__ = '20200913'   #yyymmdd
__author__ = 'Robert Nikutta <robert.nikutta@gmail.com>'

"""Utilities for handling units and units strings.

.. automodule:: units
"""

import numpy as np
from astropy import units as u
import re

## Allowed units, per category; add more as required
UNITS = {\
         'ANGULAR' : ('arcsec','mas','milliarcsecond','deg','degree','rad','radian'), #: Recognized angular units, e.g. for pixel scale.
         'LINEAR' : ('cm','m','AU','lyr','pc','kpc','Mpc','Gpc'), #: Recognized linear units (either for pixel scale, or for source distance, etc.)
         'TEMPERATURE' : ('K',), #: Recognized units of temperature
         'WAVE' : ('Angstrom','nm','micron','mm'), #: Recognized units of wavelength
         'LUMINOSITY' : ('erg/s','W','Lsun','solLum'), #: Recognized units of luminosity
         'BRIGHTNESS' : ('Jy/pix','mJy/pix'), #: Recognized units of brightness-per-pixel.
         'FLUXDENSITY' : ('Jy','mJy','MJy','erg/s/cm^2/Hz','W/m^2/Hz') #: Recognized units of flux denisty
} #: Dictionary of permitted units.

UNITS['CUNITS'] = UNITS['ANGULAR'] + UNITS['LINEAR'] # Union of permitted angular and linear units.
UNITS['INVERSELINEAR'] = tuple(['1/%s'%e for e in UNITS['LINEAR']])  # inverse lenghts  # TODO: check if we can remove these

## TODO: implement also per-beam, brightness specifications (and maybe also per-pc^2 etc.)
##        self.UNITS_BRIGHTNESS_SOLIDANGLE = ('Jy/arcsec^2','Jy/mas^2','Jy/milliarcsec^2','mJy/arcsec^2','mJy/mas^2','mJy/milliarcsec^2')

         
# compile regex pattern to identify numbers in a string
numeric_pattern = r"""^\s*
         [-+]? # optional sign
         \s*
         (?:
             (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
             |
             (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
         )
         # followed by optional exponent part if desired
         (?: [Ee] [+-]? \d+ ) ?
         """ #: Regex pattern to identify numbers in a string.
rx = re.compile(numeric_pattern, re.VERBOSE)


def list_recognized_units():

    """Print simple table of permitted units per category.

    Example
    -------
    .. code-block:: python

        units.list_recognized_units()

    Prints

    .. code-block:: text

        Category       Permitted units
         FLUXDENSITY   ('Jy', 'mJy', 'MJy', 'erg/s/cm^2/Hz', 'W/m^2/Hz')
         TEMPERATURE   ('K',)
          BRIGHTNESS   ('Jy/pix', 'mJy/pix')
          LUMINOSITY   ('erg/s', 'W', 'Lsun', 'solLum')
                WAVE   ('Angstrom', 'nm', 'micron', 'mm')
              CUNITS   ('arcsec', 'mas', 'milliarcsecond', 'deg', 'degree', 'rad', 'radian', 'cm', 'm', 'AU', 'lyr', 'pc', 'kpc', 'Mpc', 'Gpc')
             ANGULAR   ('arcsec', 'mas', 'milliarcsecond', 'deg', 'degree', 'rad', 'radian')
              LINEAR   ('cm', 'm', 'AU', 'lyr', 'pc', 'kpc', 'Mpc', 'Gpc')
    """
    
    print("Category       Permitted units")
    for k,v in UNITS.items():
        print("{:12s}".format(k_),v)
    

def getQuantity(quantity,recognized_units):

    """Split a string 'Value Units' into numerical value and string-units,
    and create an instance of astropy.units.Quantity.

    Parameters
    ----------
    
    quantity : str
       E.g. ``quantity='1 Jy'`` would return ``<Quantity 1.0 Jy>``

    recognized_units : tuple
        Tuple of valid units (as strings) for ``quantity``. E.g., for
        a length quantity, recognized units could be
        ('micron','m','pc','AU'), etc.

    Returns
    -------
    quantity : instance
        Instance of class :class:`astropy.units.Quantity`, i.e. a value with units.

    Examples
    --------

    .. code-block:: python

        getQuantity('1 m',('micron','m','pc','AU'))
          <Quantity 1.0 m>

        getQuantity('1 W',('micron','m','pc','AU'))
          ValueError: Specified unit 'W' is not recognized. Recognized are: m,AU,pc

    """
    
    cdelt, cunit = getValueUnit(quantity,recognized_units)
    return cdelt * cunit


def getValueUnit(quantity,recognized_units):

    """Split `string` into value and units string.

    Evaluate both and return the numerical value, and the units object.

    Parameters
    ----------
    quantity : str or `Quantity` instance
        If string, its format is `num units`, where `num` is the
        string representation of a numerical value, and `units` is one
        of the recognized units in the sequence of strings provided in
        ``recognized_units``. `num` and `units` can but need not be
        separated by whitespace.

        If instance of `Quantity`, ``quantity.value`` and
        ``quantity.unit`` are the equivalents of `num` and `unit`.

    recognized_units :  seq of strings
        Sequence of strings representing units that the `units` part
        of `string` will be checked against.

    Examples
    --------
    .. code:: python

        recognized_units = ('Jy/arcsec^2','mJy/mas^2','uJy/sr')
        getValueUnit('5.2 Jy/arcsec^2',recognized_units)
          (5.2, Unit("Jy / arcsec2"))

    """

    if isinstance(quantity,u.Quantity):
        value, unit = quantity.value, quantity.unit
        unit = unit.to_string().replace(" ","")
        
    elif isinstance(quantity,str):
        value = rx.findall(quantity)

        try:
            value = value[0]
        except IndexError:
            value = ''
            
        unit = quantity[quantity.find(value)+len(value):].replace(" ","")
        
    else:
        raise AttributeError("Argument 'quantity' is neither string nor instance of 'Quantity'.")

    try:
        value = np.float(value)
    except ValueError:
        value = ''

    if unit not in recognized_units:
        raise ValueError("Specified unit '{:s}' is not recognized. Recognized are: {:s}".format(unit,','.join(recognized_units)))

    unit = u.Unit(unit)

    return value, unit
