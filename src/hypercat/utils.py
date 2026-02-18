__version__ = '20200913'   #yyymmdd
__author__ = 'Robert Nikutta <robert.nikutta@gmail.com>'

"""General helper func for hypercat.

.. automodule:: utils
"""

import os
from tkinter import Tk, filedialog
import numpy as np
from astropy.coordinates import name_resolve
from astropy import wcs
import logging


def pickfile():
    """Ask user to select a local file in a pop-up window.

    Returns the selected file path as a string. Works in a local
    Jupyter.

    """
    
    root = Tk()
    filename = filedialog.askopenfilename(title="Select a file")
    root.destroy()
    print("File %s selected." % filename)
    
    return filename



def get_rootdir():

    """Construct Hypercat rootdir.

    Only assumes that the current working directory is somewhere below
    FOO/hypercat/.

    Returns
    -------
    rootdir : str
        Root directory of Hypercat, terminating in 'hypercat/'.

    """

    pwd = os.path.realpath(__file__)
    rootdir = pwd[:pwd.find('hypercat')]+'hypercat/'
    return rootdir
    
#    cwd = os.path.realpath('.')
#    rootdir = cwd[:cwd.find('hypercat')]+'hypercat/'
#    return rootdir


def get_wcs(image,projection=("RA---TAN","DEC--TAN"),cunit='deg'):

    """Create WCS object.

    If image.objectname is name-resolvable, the real coordinates will
    be resolved (via Vizier) and used for the WCS.

    With the resolved coordinates and with ``image.npix`` an
    ``image.pixelscale`` construct a world coordinate system (WCS).

    Parameters
    ----------
    image : instance
        Instance of :class:`imageops.Image` class. Must have members
        ``.objectname``, ``.npix`` and ``.pixelscale``.

    projection : tuple
        Tuple of two strings, each being the projection to be used for
        the WCS principal axes (usually RA and DEC). Default is
        ("RA---TAN","DEC--TAN"), corresponding to a gnomonic
        projection. The formatting of the strings and the possible
        values are described in http://docs.astropy.org/en/stable/wcs/

    cunit : str
        Coordinate unit. Default: 'deg'

    Returns
    -------
    w : instance
        Instance of :class:`astropy.wcs.wcs.WCS`. Can be used to plot
        image with sky coordinates (see e.g. function
        :func:`plotting.plot_with_wcs()`), or to save image to FITS
        file with correct WCS in the header (see function
        :func:`ioops.save2fits()`).

    Example
    -------
    Creating a source and a sky image...

    .. code-block:: python

       import hypercat
       cube = ... # load (sub-)hypercube
       ngc1068 = hypercat.Source(cube,luminosity='2e45 erg/s',distance='14.4 Mpc',objectname='ngc1068',pa='45 deg')
       vec = (40.,5,10) # parameter vector suitable for ``cube``
       sky = ngc1068(vec,total_flux_density='2500 mJy') # ``sky`` inherits ``.objectname`` from source instance (here ``ngc1068``)

    This prints:

    .. code-block:: text

       Rotated image (see self.data) by 45.0 deg in direction 'NE'.
       Coordinates for source 'ngc1068' successfully resolved. WCS created.

    """

    try:
        coords = name_resolve.get_icrs_coordinates(image.objectname)
        logging.info("Coordinates for source '{:s}' resolved. WCS created.".format(image.objectname))
        crvals = (coords.ra.to(cunit).value, coords.dec.to(cunit).value)
        
    except name_resolve.NameResolveError as e:
        msg = """Coordinate resolution for source name '{:s}' failed. Either a source with such name could not be resolved, or your network connection is down. If you wish to have a WCS created, reconnect to the internet and try again. Otherwise proceed without WCS.""".format(image.objectname)
        logging.warn(msg)
        crvals = (0,0)
    
    crpix = image.npix//2 + 1
    cdelt = image.pixelscale.to(cunit).value
        
    w = wcs.WCS(naxis=2)
    w.wcs.cunit = (cunit,cunit)
    w.wcs.crpix = (crpix,crpix)
    w.wcs.cdelt = np.array((-cdelt,cdelt))
    w.wcs.ctype = projection
    w.wcs.crval = crvals

    return w
    

def arrayify(seq,shape=None,fill=False,direction='x'):

    """Arrange elements of sequence `seq` into an array of shape `shape`.

    Definitions:
      * `seq` is input sequence of elements
      * `arr` is output array
      * ``ne = len(seq)`` # number of elements in input sequence
      * ``na = np.prod(shape)`` # number of elements in output array

    Behavior:
      * If `seq` is just a single element, then ``arr.shape`` is (1,1).
      * If ``ne>=na``, ``arr[:ne] = seq[:]``, i.a. any extra elements from `seq` will be truncated
      * If ``ne<na``, then ``arr[ne:] = None``, i.e. the extra elements in `arr` will be set to ``None``
      * If ``ne==1`` and ``fill==True``, then ``arr[:] = seq``, i.e. all elements of `arr` will be set to the value of `seq`

    Parameters
    ----------
    seq : tuple or list or single element
        Sequence of elements to be arranged in an array. If single
        element, it will be turned into a len-1 sequence.

    shape : tuple or None
       2-tuple giving the shape of the output array. If ``None``
       (default), the array will have shape (1,ne).
    
    fill : bool
        If ``True`` and ``ne==1`` and ``na>ne``, the entire output
        array will be filled with the value of `seq`.

    direction : str
        If ``'x'`` (default), transpose the output array.

    """

    
    # turn single-element argument into a len-1 sequence
    if not isinstance(seq,(list,tuple)):
        seq = tuple([seq])

    # if no shape arg was given
    if not isinstance(shape,tuple):
        shape = (1,len(seq))

    ne = len(seq) # number of elements in inout sequence
    na = np.prod(shape) # number of elements in output array

    arr = np.array([None]*na) # flatted version of output array

    # deal with sequences too long or too short for array
    nmin = min(ne,na)
    arr[:nmin] = seq[:nmin]
    if ne == 1 and fill is True:
        arr[:] = seq[0]
    
    arr = arr.reshape(shape) # final shape of array

    # transpose if necessary
    if direction == 'x':
        arr = arr.reshape(shape[::-1]).T

    return arr
    

def mirror_axis(cube,axis=-2):

    """Mirror one axis of an n-cube.

    Parameters
    ----------
    cube : array
        Expected shape of cube: (...,nx,...).

    axis : int
        axis index to be mirrored. Default: -2. Mirroring assumes a
        central columns of elements. See 'Returns' section.

    Returns
    -------
    newcube : array
        Array with 'axis' dimension mirrored, such that
          newcube.shape = (...,2*nx-1,...)
        i.e. newcube.shape[axis] always odd.

    Examples
    --------
    .. code-block:: python

        c = np.arange(9).reshape((3,3))
        c
        --> array([[0, 1, 2],
                   [3, 4, 5],
                   [6, 7, 8]])

        mirror_axis(c,axis=0)
        --> array([[6, 7, 8],
                   [3, 4, 5],
                   [0, 1, 2],
                   [3, 4, 5],
                   [6, 7, 8]])

        mirror_axis(c,axis=1)
        --> array([[2, 1, 0, 1, 2],
                   [5, 4, 3, 4, 5],
                   [8, 7, 6, 7, 8]])

    """
        
    ndim = cube.ndim
    npix = cube.shape[axis]
    allpads = [(0,0)]*ndim
    allpads[axis] = (npix-1,0)
    
    newcube = np.pad(cube,tuple(allpads),'reflect')

    return newcube
    

def seq2str(seq,jstr=','):
    
    """Join a sequence ``seq`` of string-ifiable elements with a string
    ``jstr``.

    Example
    -------
    .. code:: python

        seq2str(('a','b','c'),' - ')
          'a - b - c'

        seq2str([1,2,3],':')
          '1:2:3'

    """
    
    return jstr.join([str(_) for _ in seq])
