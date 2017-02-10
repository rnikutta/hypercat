__version__ = '20170209'   #yyymmdd
__author__ = 'Robert Nikutta <robert.nikutta@gmail.com>'

"""Utilities for handling the CLUMPY image hypercube.

.. automodule:: imageops
"""

import logging
import numpy as N
from scipy import ndimage
from units import *
from astropy import nddata  # todo: re-implement the functionality provided by nddata.extract_array() & remove this dependency


class ImageFrame:

    def __init__(self,image,pixelscale=None,distance=None):

        # SANITY CHECKS
        self.npix = checkImage(image,returnsize=True)
        
        # PIXELSCALE
        if pixelscale is not None:
            self.setPixelscale(pixelscale=pixelscale,distance=distance)

        # IMAGE DATA
        self.data_raw = image  # keep original array
#        self.data = image   # will rescale this array in setBrightness()
        self.data = image * u.Quantity(1)   # will rescale this array in setBrightness()
        

    def setPixelscale(self,pixelscale='1 arcsec',distance=None):

        """Set size and area of a pixel from user input.

        This does not affect the brightness per pixel.

        Parameters
        ----------
        pixelscale : str or 'Quantity' instance
            Angular or linear scale of one pixel. See docstring of
            :func:`getValueUnit` for the requirements.

            Recognized angular units: (see ``UNITS_ANGULAR`` in :module:`units`)

            Recognized linear units: (see ``UNITS_LINEAR`` in :module:`units`)

            If linear unit, then it is the linear size of a pixel at
            the distance of the source. Then the `distance` argument
            must be also specified (see below).

        distance : str | 'Quantity' instance | None
            If `pixelscale` was specified in linear units (e.g. pc),
            then the distance to the source must be not ``None``, but
            rather an argument akin to 'pixelscale' above, i.e. str or
            'Quantity' instance.

            Recognized linear units: (see ``UNITS_LINEAR`` in :module:`units`)

        Examples
        --------
        .. code-block:: python

            I = Image(raw_image,pixelscale='0.2 arsec')  # definition of an 'Image' instance
            print I.pixelscale; print I.pixelarea
              0.2 arcsec
              0.04 arcsec2 / pix

            I.setPixelscale(pixelscale='1 AU',distance='1 pc')
            print I.pixelscale, I.pixelarea
              1.0 arcsec
              1.0 arcsec2 / pix

        """

        self.pixelscale = getQuantity(pixelscale,CUNITS)

        if self.pixelscale.unit in UNITS_LINEAR:
            try:
                self.distance = getQuantity(distance,UNITS_LINEAR)
            except AttributeError:
                logging.error("Must provide a value for 'distance' argument. Current value is: "+str(distance))
                raise

            self.pixelscale = N.arctan2(self.pixelscale,self.distance).to('arcsec')
            

        self.__computePixelarea()
        self.__computeFOV()

        
    def __computeFOV(self):

        self.FOV = self.npix * self.pixelscale
        
        
    def __computePixelarea(self):

        self.pixelarea = self.pixelscale**2


    def resample(self,newpixelscale):

        """Increase or decrease the image sampling in x and y by `resamplingfactor`.

        The resampling is performed by third-order spline
        interpolation. The FOV remains constant, but
        ``self.pixelscale`` and ``self.pixelarea`` are adjusted.  The
        brightness-per-pixel is adjusted according to the change in
        pixel area, but total flux density is preserved.  Note that
        consecutive application of :func:`resample` to the same image
        can yield inexact results due to accumulated interpolation
        errors.
        
        Because resampling is performed by spline interpolation, the
        total flux in the image may change slightly. This function
        thus renormalizes the resampled image to the previous total
        flux.

        Parameters
        ----------
        resamplingfactor : float
            The factor by which the the image sampling (number of
            pixels) is increased (if >1) or decreased (if <1). Note
            that `resamplingfactor` may be adjusted because of the
            requirement that ``npix_new`` be integer and odd-valued.

        Examples
        --------
        .. code:: python

            theta = (30,0,3,0,20,9.7) # (sig,i,N0,q,tauv,lambda)
            img = M.get_image(theta)  # raw model image

            # `Image` instance
            I = hypercat.Image(img,pixelscale='1 AU',distance='1 pc',peak_pixel_brightness='1.0 mJy/pix')

            print ", ".join([str(e) for e in (I.npix,I.pixelscale,I.pixelarea,I.FOV,I.data.max(),I.getBrightness('mJy/arcsec^2').max())])
              221, 1.0 arcsec, 1.0 arcsec2 / pix, 221.0 arcsec, 1.0 mJy / pix, 1.0 mJy / arcsec2
            print I.getTotalFluxDensity()
              5.35844 Jy  # total flux density

            # new instance, upsampled 2x
            I = hypercat.Image(img,pixelscale='1 AU',distance='1 pc',peak_pixel_brightness='1.0 mJy/pix')
            I.resample(2.0)
            print ", ".join([str(e) for e in (I.npix,I.pixelscale,I.pixelarea,I.FOV,I.data.max(),I.getBrightness('mJy/arcsec^2').max())])
              443, 0.49887 arcsec, 0.24887 arcsec2 / pix, 221.0 arcsec, 0.26328 mJy / pix, 1.05787 mJy / arcsec2
            print I.getTotaFluxDensity()
              5.38289 Jy  # total flux density preserved; difference due to interpolation errors

            # new instance, downsampled 0.5x
            I = hypercat.Image(img,pixelscale='1 AU',distance='1 pc',peak_pixel_brightness='1.0 mJy/pix')
            I.resample(0.5)
            print ", ".join([str(e) for e in (I.npix,I.pixelscale,I.pixelarea,I.FOV,I.data.max(),I.getBrightness('mJy/arcsec^2').max())])
              111, 1.99099099 arcsec, 3.96405 arcsec2 / pix, 221.0 arcsec, 3.92935 mJy / pix, 0.99125 mJy / arcsec2
            print I.getTotalFluxDensity()
              5.30354 Jy  # still preserved

        """

        newpixelscale = getQuantity(newpixelscale,recognized_units=UNITS_ANGULAR)
        
        resamplingfactor = (self.pixelscale / newpixelscale).decompose().value
        newimage, newfactor, self.npix = resampleImage(self.data.value,resamplingfactor,conserve=True)
        
        self.data = newimage * self.data.unit
        
        self.setPixelscale(pixelscale=self.pixelscale/newfactor)
        if newfactor != resamplingfactor:
            logging.warning("The requested resampling to pixel scale (%g %s) was slightly adjusted due to discretization (now %g %s). This is to preserve sizes on the sky." % (newpixelscale.value,newpixelscale.unit,self.pixelscale.value,self.pixelscale.unit))
            

    def rotate(self,angle,direction='NE',returnimage=False):

        """Rotate ``self.data`` by `angle` degrees from North towards
        `direction` (either East or West).

        See docstring of :func:`rotateImage()`

        """
        
        self.data = rotateImage(self.data.value,angle=angle,direction=direction) * self.data.unit
        logging.info("Rotated image (see self.data) by %s in direction '%s'." % (str(angle),direction))

        if returnimage:
            return self.data


    def setFOV(self,fov):

        """From field-of-view `fov`, compute new ``pixelscale`` & new ``pixelarea``.

        ``self.npix`` remains the same, i.e. recomputes
        ``self.pixelscale`` and ``self.pixelarea`` and sets
        ``self.FOV``.

        Parameters
        ----------
        fov : str or 'Quantity' instance
            Target angular size of the field-of-view. See docstring of
            :func:`getValueUnit` for the requirements.

            Recognized angular units: (see ``UNITS_ANGULAR`` in :class:`Image`)

        Examples
        --------
        .. code-block:: python

            I = hypercat.Image(img,pixelscale='1 AU',distance='1 pc',peak_pixel_brightness='4.0 mJy/pix')
            print ('%s\\n'*4) % (I.pixelscale,I.FOV,I.data.max(),I.getBrightness('mJy/arcsec^2').max())
              1.0 arcsec
              7.0 arcsec
              4.0 mJy / pix
              4.0 mJy / arcsec2

            I.setFOV(fov='14.0 arcsec')
            print ('%s\\n'*4) % (I.pixelscale,I.FOV,I.data.max(),I.getBrightness('mJy/arcsec^2').max())
              2.0 arcsec
              14.0 arcsec
              4.0 mJy / pix
              1.0 mJy / arcsec2

        """

        FOV = getQuantity(fov,UNITS_ANGULAR)
        self.pixelscale = FOV / N.float(self.npix)
        self.__computePixelarea()
        self.__computeFOV()

        
    def changeFOV(self,fov):

        """Embed ``self.data`` in a larger field of view, or crop to a smaller one.

        If ``fov > self.FOV``, embed self.data in a larger array. If
        ``fov < self.FOV``, crop the central pixels of ``self.data``
        to `fov`. Cropping removes the cropped parts of the image
        permanently.

        Parameters
        ----------
        fov : str or 'Quantity' instance
            Angular size of the target field of view (FOV). See
            docstring of :func:`getValueUnit` for the requirements.
            `fov` can be both larger or smaller than the current image
            FOV ``self.FOV``.

        Notes
        -----
        Since the image is pixelated (discretized), embed/crop to the
        nearest odd integer ``npix``, and use the corrected FOV, which is
        not necessarily exactly the specified `fov`.

        Note that ``self.pixelscale`` and ``self.pixelarea`` remain
        unchanged!

        """
        
        FOV = getQuantity(fov,UNITS_ANGULAR)
        factor = (FOV/self.FOV).decompose().value
        newsize_int, newfactor = computeIntCorrections(self.npix,factor)
        cpix = self.npix/2


        def get_newimage(image):
            return nddata.extract_array(image,(newsize_int,newsize_int),(cpix,cpix),fill_value=0.)
        
        if hasattr(self.data,'value'):
            newimage = get_newimage(self.data.value)
        else:
            newimage = get_newimage(self.data)
            
        if hasattr(self.data,'unit'):
            self.data = newimage * self.data.unit
        else:
            self.data = newimage
            
        # updates
        self.npix = self.data.shape[0]
        self.__computeFOV()

    # convenience functions
    def getImageT(self):
        try:
            return self.data.value.T
        except AttributeError:
            return self.data.T
        except:
            raise

    I = property(getImageT)


class Image(ImageFrame):

    def __init__(self,image,pixelscale='1 arcsec',distance=None,\
                 total_flux_density='1 Jy',pa='0 deg'):

        ImageFrame.__init__(self,image,pixelscale=pixelscale,distance=distance)

        """From a 2D array instantiate an Image object.

        The ``Image`` instance has members for pixelscale, brightness,
        unit conversions, etc.  The way to specify physical quantities
        with units follows roughly the CASA (Common Astronomy Software
        Applications package) philosophy, i.e. they are strings of the
        form 'value unit', e.g. '1.5 Jy'.

        Parameters
        ----------
        image : float array (2D)
            Monochromatic relative brightness per pixel. Will be
            scaled to physical values via arguments 'pixelscale' and
            'peak_pixel_brightness' (see below).

        pixelscale : str
            Pixel scale of a single pixel in the model image. See
            docstring of setPixelscale() function.

        distance : str | None
            Physical distance to the source. Only required if
            'pixelscale' was specified in linear (not angular)
            units. See docstring of setPixelscale() function.

        total_flux_density : str | 'Quantity' instance
            Target total flux (sum of all pixels) in
            ``self.data``. See docstring of :func:`setBrightness()`.

        pa : str | 'Quantity' instance
            Position angle with respect to North (=0 deg). If not 0,
            the image will be rotated by pa (positive values rotate
            North-to-East, i.e. anti-clockwise, negative values rotate
            North-to-West, i.e. clockwise). Example: ```pa='42
            deg'```.

        Examples
        --------
        .. code-block:: python

            # get raw image (see :class:`ModelCube`:func:`get_image`)
            image = M.get_image(theta)

            # init `Image` instance using angular pixel scale
            I = Image(image,pixelscale='30 mas',total_flux_density='500 mJy')
            print I.pixelscale; print I.data.max(); print I.getTotalFluxDensity()
              30.0 mas
              0.525156676769 mJy / pix
              0.49999995972 Jy

            # init using linear pixel scale and source distance
            I = Image(image,pixelscale='1 AU',distance='1 pc',peak_pixel_brightness='3.0 mJy/pix')
            print I.pixelscale; print I.distance; print I.data.max()
              1.0 arcsec
              1.0 pc
              3.0 mJy / pix

        """
        
        if total_flux_density is not None:
            self.setBrightness(total_flux_density)

        if pa != None:
            self.rotate(pa)
            

    def setBrightness(self,total_flux_density='1 Jy'):
        
        """Scale the brightness of the image such that total flux is ``total_flux_density``.

        This does not touch the ``self.pixelsize`` and
        ``self.pixelarea`` values of the image.

        Parameters
        ----------
        total_flux_density : str | 'Quantity' instance
            Target total flux (sum of all pixels) in
            ``self.data``. See docstring of :func:`getValueUnit` for
            the requirements.

            Recognized brightness units: (see ``UNITS_FLUXDENSITY`` in
            :module:`units`)

        Examples
        --------
        .. code-block:: python

            print I.data.max(); print I.getTotalFluxDensity()
            0.000383368023904 Jy / pix
            1.00000080509 Jy

            I.setBrightness('2 Jy')
            print I.data.max(); print I.getTotalFluxDensity()
            0.000766735291108 Jy / pix
            1.99999992838 Jy
        """

        total_flux_density = getQuantity(total_flux_density,recognized_units=UNITS_FLUXDENSITY)
        self.data = (self.data/self.data.sum()) * total_flux_density / u.pix
        
        
    def getBrightness(self,units='Jy/arcsec^2'):

        """Compute the image brightness in the requested 'units'
        (brightness-per-solid-angle).

        Parameters
        ----------
        units : str
            Units of brightness-per-solid-angle, to which the
            self.data array will be converted using self.pixelscale

            The units are quite flexible. An incomplete list of
            possibilities: ``Jy/mas^2, mJy/arcsec^2, uJy/sr,
            Jy/deg^2``, etc.

        Example
        -------
        .. code-block:: python

             print I.pixelscale; print I.data.max(); print I.getBrightness('Jy/arcsec^2')
               1.0 arcsec
               3.0 mJy / pix
               0.003 Jy / arcsec2

        """
        
        try:
#            brightness =  (self.data / self.pixelarea).to(units)
            brightness =  (u.pix * self.data / self.pixelarea).to(units)
        except ValueError:
            logging.error("Use valid brightness-per-solid-angle units, e.g. 'Jy/arcsec^2' or 'mJy/sr', etc.")
            raise

        return brightness
    

    def getTotalFluxDensity(self,units='Jy'):

        """Compute the total flux density within the FOV.

        The total flux density is the brightness integral over the solid
        angle of the FOV.

        Parameters
        ----------
        units : str

            The brightness units in which the total flux density
            should be output. Default: 'Jy'


        Returns
        -------
        totalFluxDensity : float
            Total flux density in Jy.

        Example
        -------
        .. code:: python

            total = I.getTotalFluxDensity()
            print total; print total.to('mJy')  # can be easily converted to other units
              0.0050459482673 Jy
              5.0459482673 mJy

        """

        val, unit = getValueUnit(units,UNITS_FLUXDENSITY)  # val is dummy here, since no actual value
        
        return (self.getBrightness('Jy/mas^2').sum() * self.pixelarea).to(unit)
    
    F = property(getTotalFluxDensity)


# HIGH-LEVEL HELPERS
def rotateImage(image,angle,direction='NE'):

    """Rotate an image around its central pixel by `angle` degrees.

    The parts of `image` which after rotation are outside the frame,
    will be truncated. Areas in the rotated image which are empty will
    be filled will zeros.

    The rotation is performed using high-quality cubic spline
    interpolation.

    Parameters
    ----------
    image : 2D array
        The square image array to be rotated.

    angle : float or int
        Image will be rotated by that many degrees.

    direction : str
        Rotation direction. Default is 'NE', i.e. from North to East,
        i.e. anti-clockwise, which is astronomical standard. To rotate
        clockwise instead, give direction='NW' (or negate the `angle`
        argument).

    Returns
    -------
    rotimage : 2D array
        The `image`, rotated by `angle` degrees towards `direction`.

    """

    checkImage(image,returnsize=False)

    angle = getQuantity(angle,recognized_units=UNITS_ANGULAR)
    
    if direction == 'NW':
        angle = -angle
    
    rotimage = ndimage.rotate(image,angle.to('deg').value,reshape=False)

    return rotimage


def resampleImage(image,resamplingfactor,conserve=True):

    """Resample an image by `resamplingfactor`.

    Parameters
    ----------
    image : 2D array
        The 2D image array to be resampled. npix must be odd-valued.

    resamplingfactor : float
        The factor by which the the image sampling (number of pixels)
        is increased (if >1) or decreased (if <1). Note that
        `resamplingfactor` may be adjusted because of the requirement
        that ``npix_new`` be integer and odd-valued.

    conserve : bool
        If True (default), the resampled image will be renormalized
        such that the total flux between the original and the
        resampled images is preserved.
    
    Returns
    -------
    newimage : 2d array
        The resampled image. Final size ``npix`` is adjusted to the
        nearest odd-valued integer.

    newfactor : float
        The actually used resampling factor. Adjusted, if necessary, by
        the requirement that ``npix_new`` be integer and odd-valued.

    npix : int
        Number of pixels along one axis of ``newimage``.

    """
    
    npix = checkImage(image,returnsize=True)
    total = image.sum()
    newsize_int, newfactor = computeIntCorrections(npix,resamplingfactor)
    newimage = ndimage.zoom(image,newfactor)
    npix = checkImage(newimage,returnsize=True)

    if conserve is True:
        newtotal = newimage.sum()
        newimage *= (total / newtotal)
    
    return newimage, newfactor, npix
    
# LOW-LEVEL HELPERS
def checkInt(x):
    
    """Check whether x is integer."""
    
    if not isinstance(x,int):
        raise TypeError('x is not integer.')

def checkOdd(x):

    """Check whether x is integer and odd."""
    
    checkInt(x)
        
    if x % 2 == 0:
        raise ValueError('x is not odd.')

def checkImage(image,returnsize=True,enforce2d=True):

    """Make a few common assertions for ``image``.

    Verifies that ``image`` is 2D, square with shape (npix,npix), and
    that npix is odd-valued. Raises various Exceptions if test fails.

    Parameters
    ----------
    image : 2D array
        The image (2D array) to be tested.

    returnsize : bool
        If ``True``, returns ``image.shape[0]``, i.e. the number of
        pixels along `image`, if all test pass.

    enforce2d : bool
        If True (default), the image array must be 2D. Otherwise,
        e.g. a multi-wavelength image array of shape (Nlambda,Nx,Ny)
        can be provided. Caution: some functions in hypercat can not
        yet handle multi-wavelength images.

    Examples
    --------
    .. code:: python

        # all tests pass, returns nothing
        image = N.ones(11,11)
        check(image,returnsize=False)

        # fails with Exception if image not square
        image = N.ones(11,10)
        check(image,returnsize=False)
          ...
          ValueError: 'image' must be square (nx=ny)

        # fails with Exception if size not odd
        image = N.ones(10,10)
        check(image,returnsize=False)
          'image' size in pixels (nx) must be odd and integer, but is nx =  10
          ...
          ValueError: x is not odd.

        # fails with Exception if image not 2D
        image = N.ones(11,11,3)
        check(image,returnsize=False)
          ...
          ValueError: 'image' must be 2D.

    """

    # image props assertions
    if enforce2d is True:
        if image.ndim != 2:
            raise ValueError("'image' must be 2D.")

    nx, ny = image.shape[-2:]
    
    if (nx != ny):
        raise ValueError("'image' must be square (nx=ny)")

    try:
        checkOdd(nx)
    except (TypeError, ValueError):
        logging.error("'image' size in pixels (nx) must be odd and integer, but is nx = ", nx)
        raise

    if returnsize:
        return nx


def computeIntCorrections(npix,factor):

    """Compute `newnpix` and `newfactor` such that `npix_new` is the
    nearest odd-valued integer to ``npix*factor``.

    Parameters
    ----------

    npix : int
        Odd-valued integer.

    factor : float
        Multiplicative factor.

    Returns
    -------
    newnpix : int
        Nearest odd integer to ``npix*factor``.

    newfactor : float
        `factor` but corrected such that ``npix*newfactor = newnpix``.

    Examples
    --------
    .. code:: python
    
        computeIntCorrections(3,2.)
          (7, 2.33333)  # factor was corrected

        computeIntCorrections(3,5)
          (15, 5.0)  # no corrections necessary

        computeIntCorrections(15,0.6)
          (9, 0.6)  # can also scale down

        computeIntCorrections(15,0.5)
          (7, 0.46667)  # factor was corrected

    """
    
    checkOdd(npix)
    newnpix = npix*factor
    newnpix = N.int((2*N.floor(newnpix/2)+1))  # rounded up or down to the nearest odd integer
    newfactor = newnpix/float(npix)

    return newnpix, newfactor
