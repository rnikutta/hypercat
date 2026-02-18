__version__ = '20210617'   #yyymmdd
__author__ = 'Robert Nikutta <robert.nikutta@gmail.com>'

"""Utilities for handling the CLUMPY image hypercube.

.. automodule:: imageops
"""

# std dev
import logging
from copy import copy

# 3rd party
import numpy as np
from scipy import ndimage
import astropy
from astropy import nddata  # todo: re-implement the functionality provided by nddata.extract_array() & remove this dependency

# hypercat
#from units import *
from .units import *

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
#        print("At init: self.data.value.std() =", self.data.value.std())


    def setPixelscale(self,pixelscale='1 arcsec',distance=None):

        """Set size and area of a pixel from user input.

        This does not affect the brightness per pixel.

        Parameters
        ----------
        pixelscale : str or 'Quantity' instance
            Angular or linear scale of one pixel. See docstring of
            :func:`getValueUnit` for the requirements.

            Recognized angular units: (see ``UNITS['ANGULAR']`` in :module:`units`)

            Recognized linear units: (see ``UNITS['LINEAR']`` in :module:`units`)

            If linear unit, then it is the linear size of a pixel at
            the distance of the source. Then the `distance` argument
            must be also specified (see below).

        distance : str | 'Quantity' instance | None
            If `pixelscale` was specified in linear units (e.g. pc),
            then the distance to the source must be not ``None``, but
            rather an argument akin to 'pixelscale' above, i.e. str or
            'Quantity' instance.

            Recognized linear units: (see ``UNITS['LINEAR']`` in :module:`units`)

        Examples
        --------
        .. code-block:: python

            I = hypercat.Image(raw_image,pixelscale='0.2 arcsec')  # definition of an 'Image' instance
            print(I.pixelscale); print(I.pixelarea)
              0.2 arcsec
              0.04 arcsec2

            I.setPixelscale(pixelscale='1 AU',distance='1 pc')
            print(I.pixelscale); print(I.pixelarea)
              1.0 arcsec
              1.0 arcsec2

        """

        self.pixelscale = getQuantity(pixelscale,UNITS['CUNITS'])

        if self.pixelscale.unit in UNITS['LINEAR'] and distance is not None:
            try:
                self.distance = getQuantity(distance,UNITS['LINEAR'])
            except AttributeError:
                logging.error("Must provide a value for 'distance' argument. Current value is: "+str(distance))
                raise

            self.pixelscale = np.arctan2(self.pixelscale,self.distance).to('arcsec')


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
        .. code-block:: python

            theta = (30,0,3,0,20,9.7) # (sig,i,N0,q,tauv,lambda)
            img = M.get_image(theta)  # raw model image

            # `Image` instance
            I = hypercat.Image(img,pixelscale='1 AU',distance='1 pc',peak_pixel_brightness='1.0 mJy/pix')

            print(", ".join([str(e) for e in (I.npix,I.pixelscale,I.pixelarea,I.FOV,I.data.max(),I.getBrightness('mJy/arcsec^2').max())]))
              221, 1.0 arcsec, 1.0 arcsec2 / pix, 221.0 arcsec, 1.0 mJy / pix, 1.0 mJy / arcsec2
            print(I.getTotalFluxDensity())
              5.35844 Jy  # total flux density

            # new instance, upsampled 2x
            I = hypercat.Image(img,pixelscale='1 AU',distance='1 pc',peak_pixel_brightness='1.0 mJy/pix')
            I.resample(2.0)
            print(", ".join([str(e) for e in (I.npix,I.pixelscale,I.pixelarea,I.FOV,I.data.max(),I.getBrightness('mJy/arcsec^2').max())]))
              443, 0.49887 arcsec, 0.24887 arcsec2 / pix, 221.0 arcsec, 0.26328 mJy / pix, 1.05787 mJy / arcsec2
            print(I.getTotaFluxDensity())
              5.38289 Jy  # total flux density preserved; difference due to interpolation errors

            # new instance, downsampled 0.5x
            I = hypercat.Image(img,pixelscale='1 AU',distance='1 pc',peak_pixel_brightness='1.0 mJy/pix')
            I.resample(0.5)
            print(", ".join([str(e) for e in (I.npix,I.pixelscale,I.pixelarea,I.FOV,I.data.max(),I.getBrightness('mJy/arcsec^2').max())]))
              111, 1.99099099 arcsec, 3.96405 arcsec2 / pix, 221.0 arcsec, 3.92935 mJy / pix, 0.99125 mJy / arcsec2
            print(I.getTotalFluxDensity())
              5.30354 Jy  # still preserved

        """

        newpixelscale = getQuantity(newpixelscale,recognized_units=UNITS['ANGULAR'])

        resamplingfactor = (self.pixelscale / newpixelscale).decompose().value
        newimage, newfactor, self.npix = resampleImage(self.data.value,resamplingfactor,conserve=True)

        self.data = newimage * self.data.unit

        self.setPixelscale(pixelscale=self.pixelscale/newfactor)
        if newfactor != resamplingfactor:
            logging.info("The requested resampling to pixel scale ({:g} {:s}) was slightly adjusted due to discretization (now {:g} {:s}). This is to preserve sizes on the sky.".format(\
                          newpixelscale.value,newpixelscale.unit,self.pixelscale.value,self.pixelscale.unit))


    def rotate(self,angle,direction='NE',returnimage=False):

        """Rotate ``self.data`` by `angle` degrees from North towards
        `direction` (either East or West).

        See docstring of :func:`rotateImage()`

        """

        self.data = rotateImage(self.data.value,angle=angle,direction=direction) * self.data.unit
        logging.info("Rotated image (see self.data) by {:s} in direction '{:s}'.".format(str(angle),direction))

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

            Recognized angular units: (see ``UNITS['ANGULAR']`` in :class:`Image`)

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

        FOV = getQuantity(fov,UNITS['ANGULAR'])
        self.pixelscale = FOV / np.float(self.npix)
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

        FOV = getQuantity(fov,UNITS['ANGULAR'])
        factor = (FOV/self.FOV).decompose().value
        newsize_int, newfactor = computeIntCorrections(self.npix,factor)
        cpix = self.npix//2

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
    def _getImageT(self):
        try:
            return self.data.value.T
        except AttributeError:
            return self.data.T
        except:
            raise

    I = property(_getImageT) #: Property alias for :func:`_getImageT`


class Image(ImageFrame):

    def __init__(self,image,pixelscale='1 arcsec',distance=None,\
                 total_flux_density='1 Jy',pa='0 deg',snr=None,brightness_units='Jy/mas^2'):

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
            print(I.pixelscale); print(I.data.max()); print(I.getTotalFluxDensity())
              30.0 mas
              0.525156676769 mJy / pix
              0.49999995972 Jy

            # init using linear pixel scale and source distance
            I = Image(image,pixelscale='1 AU',distance='1 pc',peak_pixel_brightness='3.0 mJy/pix')
            print(I.pixelscale); print(I.distance); print(I.data.max())
              1.0 arcsec
              1.0 pc
              3.0 mJy / pix

        """
        if not isinstance(pa,astropy.units.quantity.Quantity):
            pa = getQuantity(pa,recognized_units=UNITS['ANGULAR'])

        if pa.to('deg').value != 0.:
            self.rotate(pa)

        if total_flux_density is not None:
            self.setBrightness(total_flux_density)

#        # TEST
#        if brightness_units is not None:
#            self.data = self.getBrightness(brightness_units)
#        # TEST

        # add noise
        if snr is not None:
#            print("Before add_noise:: self.data.value.std() =", self.data.value.std())

            noisy_image, noise_pattern = add_noise(copy(self.data.value),snr)
            self.data = noisy_image * self.data.unit
#            print("After add_noise:: self.data.value.std() =", self.data.value.std())

            logging.info("SNR_meas = {:.2f}".format(measure_snr(noisy_image, noise_pattern)))

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

            Recognized brightness units: (see ``UNITS['FLUXDENSITY']`` in
            :module:`units`)

        Examples
        --------
        .. code-block:: python

            print(I.data.max()); print(I.getTotalFluxDensity())
            0.000383368023904 Jy / pix
            1.00000080509 Jy

            I.setBrightness('2 Jy')
            print(I.data.max()); print(I.getTotalFluxDensity())
            0.000766735291108 Jy / pix
            1.99999992838 Jy
        """

        total_flux_density = getQuantity(total_flux_density,recognized_units=UNITS['FLUXDENSITY'])
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

             print(I.pixelscale); print(I.data.max()); print(I.getBrightness('Jy/arcsec^2'))
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
        .. code-block:: python

            total = I.getTotalFluxDensity()
            print(total); print(total.to('mJy'))  # can be easily converted to other units
              0.0050459482673 Jy
              5.0459482673 mJy

        """

        val, unit = getValueUnit(units,UNITS['FLUXDENSITY'])  # val is dummy here, since no actual value

        return (self.getBrightness('Jy/mas^2').sum() * self.pixelarea).to(unit)

    F = property(getTotalFluxDensity) #: Property alias for :func:`getTotalFluxDensity`


# HIGH-LEVEL HELPERS

def add_noise(image,snr,fraction=1.0):

    """Add Gaussian noise to ``image`` such that SNR is ``snr`` at the peak pixel (or a ``fraction`` of it).

    Parameters
    ----------
    image : 2-d array
        Noise-free image.

    snr : float
        The desired SNR of image+noise_pattern at peak pixel value.

    fraction : float
        Fraction of peak pixel. Must be 0 > fraction >= 1.0. By
        default, the target SNR of the noisy image is estimated at the
        max (peak) pixel value, i.e. at fraction=1.0.

    Returns
    -------
    noisy_image, noise_pattern : 2-d arrays
        The noisy image (image + noise_pattern), and the noise_pattern
        image itself.

    See also
    --------
    :func:`measure_snr()`

    Examples
    --------
    .. code-block:: python

        img = cube(vec) # with appropriate parameter vector
        noisy_image, noise_pattern = add_noise(img,1.) # requesting SNR=1
        import plotting
        # source in noisy_image should be almost indiscernible
        plotting.multiplot((img,noisy_image,noise_pattern),titles=('image','image+noise','noise'))

    """

    if fraction <= 0. or fraction > 1.0:
        raise Exception("'fraction' must be > 0.0 and <= 1.0")

    # compute noise pattern with correct amplitude distribution
    mu = 0.0
    sigma = fraction*np.max(image) / float(snr)
    noise_pattern = np.random.normal(mu,sigma,size=image.shape)

    # noisy image
    noisy_image = image + noise_pattern

    # normalize

    return noisy_image, noise_pattern


def measure_snr(noisy_image,noise_pattern,fraction=1.0):

    """Measure the effective SNR of a noisy image.

    Assuming that noisy_image has had (Gaussian) noise _pattern added
    to the signal, measures the effective SNR at the peak pixel value
    (or a fraction of it).

    Parameters
    ----------
    noisy_image : 2-d array
        Image of signal + Gaussian noise.

    noise_pattern : 2-d array
        The noise image that was added to image to create noisy_image.

    fraction : float
        Fraction of peak pixel. Must be 0 > fraction >= 1.0. By
        default, the target SNR of the noisy image is estimated at the
        max (peak) pixel value, i.e. at fraction=1.0.

    Returns
    -------
    snr : float
        SNR = max(signal) / std(noise)
        If mode == 'mean': SNR = mean(signal) / std(noise)

    See also
    --------
    :func:`add_snr()`

    Examples
    --------
    .. code-block:: python
        noisy_image, noise_pattern = add_noise(img,10.) # requesting SNR=10
        measure_snr(noisy_image, noise_pattern)
          10.013

    """

    if fraction <= 0. or fraction > 1.0:
        raise Exception("'fraction' must be > 0.0 and <= 1.0")

    snr = fraction**2.*np.max(noisy_image) / np.std(noise_pattern)

    return snr




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

    angle = getQuantity(angle,recognized_units=UNITS['ANGULAR'])

    if direction == 'NW':
        angle = -angle

#    # remember circular zero-value area around torus, and after rotation, re-set everything in said area to zero again
#    # (rotation is a interpolation scheme and might introduce minor fluctuations)
    mask = (image <= 0.)
    rotimage = ndimage.rotate(image,angle.to('deg').value,reshape=False)
    rotimage[mask] = 0.

    return rotimage


def makepositive(image):

    res = copy(image)
    gtzero = (image>0.)
    MIN = image[gtzero].min()
    res[~gtzero] = MIN

    return res


def thresholding(image,where='below',thresh=0.,cfill=0.,):

    if where == 'below':
        mask = (image <= thresh)
    elif where == 'above':
        mask = (image >= thresh)
    else:
        raise Exception("'where' must be either 'above' or 'below'")

    image[mask] = cfill

    return image



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
    newimage = makepositive(newimage)
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


def checkEven(x):

    """Check whether x is integer and odd."""

    checkInt(x)

    if x % 2 != 0:
        raise ValueError('x is not even.')


def check2d(image):
    if image.ndim != 2:
        raise ValueError("'image' is not 2-d.")


def checkSquare(image):
    nx, ny = image.shape[-2:]
    if (nx != ny):
        raise ValueError("'image' is not square (nx=ny)")


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
    .. code-block:: python

        # all tests pass, returns nothing
        image = np.ones(11,11)
        check(image,returnsize=False)

        # fails with Exception if image not square
        image = np.ones(11,10)
        check(image,returnsize=False)
          ...
          ValueError: 'image' must be square (nx=ny)

        # fails with Exception if size not odd
        image = np.ones(10,10)
        check(image,returnsize=False)
          'image' size in pixels (nx) must be odd and integer, but is nx =  10
          ...
          ValueError: x is not odd.

        # fails with Exception if image not 2D
        image = np.ones(11,11,3)
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
    .. code-block:: python

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
    newnpix = np.int((2*np.floor(newnpix//2)+1))  # rounded up or down to the nearest odd integer
    newfactor = newnpix/float(npix)

    return newnpix, newfactor


def make_binary(img,eps=0.3):

    """Make an image strictly binary.

    Assuming that all pixels should be either 0 or 1, but that there
    might be small deviations from 0 and 1, bring all pixels up to
    ``eps`` away from 0 and 1 to exactly 0 and 1.

    Parameters
    ----------
    img : 2D array
        Image array to be 'binarified'. All values should be close to
        0 or 1.

    eps: float
        Maximal deviation from 0. and 1. values. Must be 0.0<eps<0.5
        (otherwise assignment to 0 or 1 is ambiguous.

    Returns
    -------
    img : 2D array
        The binariefied array.

    Examples
    --------
    .. code-block:: python

        img = np.array([[0.99,0.01],[-0.11,1.2]])
        img
          array([[ 0.99,  0.01],
                 [-0.11,  1.2 ]])

        make_binary(img,eps=0.1)
          array([[ 1.  ,  0.  ],
                 [-0.11,  1.2 ]])

        imageops.make_binary(img,eps=0.2)
          array([[1., 0.],
                 [0., 1.]])

    """

    if eps >= 0.5:
        raise Exception("eps>=0 can lead to ambiguous results when binarizing, since 0+0.5 <= 1-0.5")

    img[np.abs(img-1.)<eps] = 1.
    img[np.abs(img-0.)<eps] = 0.

    return img


def trim_square_odd(image):

    """Given a 2-d image, find the brightest pixel and trim the image to
       the largest possible square around it that has odd number of
       pixels along the axes.

    Parameters
    ----------
    image : 2-d array
        Input image. Need not be square, any rectangle is fine. Must
        have one brightest pixel.

    Returns
    -------
    trimmed : 2-d array
        A square 2-d array with odd npix (=nx=ny), with the brightest
        pixel from `image` in its center pixel. npix is the maximal
        size for a square inscribed within `image`.

    Examples
    --------

    # Start with an even-by-even square
    image = np.zeros((4,4))
    image[1,1] = 1
    print(image)
      [[0. 0. 0. 0.]
       [0. 1. 0. 0.]
       [0. 0. 0. 0.]
       [0. 0. 0. 0.]]
    trimmed = trim_square_odd(image)
    print(trimmed)
      [[0. 0. 0.]
       [0. 1. 0.]
       [0. 0. 0.]]

    # Try a rectangle, and have the brightest pixel at the edge
    image = np.zeros((3,4))
    image[0,1] = 1
    print(image)
      [[0. 1. 0. 0.]
       [0. 0. 0. 0.]
       [0. 0. 0. 0.]]
    trimmed = trim_square_odd(image)
    print(trimmed)
      [[1.]]

    """

    check2d(image) # test if image is indeed a 2-d array
    ny, nx = image.shape

    # find 2-index of the brightest pixel
    idxpeak1d = np.argmax(image)
    idx2d = np.unravel_index(idxpeak1d,image.shape)
    cy, cx = idx2d

    # find max margins around the central pixel
    dx = min((cx,nx-cx-1))
    dy = min((cy,ny-cy-1))
    d = min((dx,dy))

    # trim
    trimmed = image[cy-d:cy+d+1,cx-d:cx+d+1]

    return trimmed



def trim_square(img):

    """Trim image to the largest envelope square.

    Finds all zero-valued margins (left,right,top,bottom) and trims
    the image to the largest centered square. It will not cut into any
    non-zero pixels.

    Parameters
    ----------

    img : 2D array
        Image to be trimmed.

    Returns
    -------
    img : 2D array
        Trimmed image. If no trimming was performed, the original
        image is returned.

    """

    aux = np.argwhere(img>0.)
    xmin,ymin = np.min(aux,axis=0)
    xmax,ymax = np.max(aux,axis=0)

    MIN = min(xmin,ymin)
    MAX = max(xmax,ymax)

    img = img[MIN:MAX,MIN:MAX]

    return img

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile



def resample_image(img,factor=None,npixout=None):

    """Resample image either but a factor, or to a target npix value.

    This version of the function makes no checks of number of pixels etc.

    Parameters
    ----------
    img : 2D array
        Image array to be resampled.

    factor: float or None
        If not ``None``, the image will be resampled by ``factor``.

    npixout: int or None
        If not ``None``, this takes precedence over
        ``factor``. ``npixout`` is the desired image dimension after
        resampling.

    Returns
    -------
    res : 2D array
        The resampled image.

    Examples
    --------
    .. code-block:: python

        img = np.zeros((101,101))

        resample_image(img,factor=2.0).shape
          factor =  2.0
          (202,202)

        resample_image(img,factor=0.7).shape
          factor =  0.7
          (71,71) # uses round() function to get integer npix

        resample_image(img,npixout=202).shape
          npixin,npixout =  101 202
          factor =  2.0
          (202, 202)

        imageops.resample_image(img,npixout=71).shape
          npixin, npixout =  101 71
          factor =  0.7029702970297029
          (71, 71)
    """

    if factor is None:
        pass

    if npixout is not None:
        npixin = img.shape[0]
        factor = npixout / npixin
        logging.info("npixin, npixout = %d, %d" % (npixin,npixout))

    if factor is None and npixout is None:
        raise Exception("Specify either ``factor`` ot ``npixout``.")

    print("factor = ", factor)
    res = ndimage.zoom(img,factor)
    res = make_binary(res)

    return res
