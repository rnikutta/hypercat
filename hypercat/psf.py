from __future__ import print_function

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import numpy as N
from astropy.modeling.models import AiryDisk2D, Gaussian2D
from astropy.convolution import convolve, convolve_fft
import astropy.io.fits as pyfits
from imageops import *
from units import *
from imageops import checkOdd

__author__ = "Enrique Lopez-Rodriguez <enloro@gmail.com>, Robert Nikutta <robert.nikutta@gmail.com>"
__version__ = '20180207' #yyyymmdd

"""Utilities for the PSF analysis of the images created by hyperCAT
    
    .. automodule:: PSF_modeling
"""

class PSF(ImageFrame):

    def __init__(self,image,pixelscale):

        """
        Parameters
        ----------
        image : array
            The PSF image (raw 2d array)

        pixelscale : str | 'Quantity' instance
            The pixelscale of `image`.
        """

        ImageFrame.__init__(self,image,pixelscale=pixelscale)

        
    def convolve(self,image):
        """ Convolution of the model PSF with a CUMPY image
        
        Parameters
        ----------
        
        image : float array (2D)
            Array obtained from hypercat using get_image module
        
        psf : float array (2D)
            Array of the model PSF obtained with PSF_modeling()
        
        Example
        -------
        .. code-block:: python
        
            convolved_image = PSF_conv(image,psf)
        """

        result = convolve_fft(image,self.data,normalize_kernel=True,allow_huge=True)

        return result


#def getPSF(image,psf='model',diameter=None,strehl=None, hdukw=None,pixelscalekw=None):
#
#
#    """Model a PSF suitable for `image`, or load from a FITS file.
#
#    Parameters
#    ----------
#    image : instance
#        `Image` instance. The size (in pixels) of the PSF image, and
#        the pixel scale, will be taken from ``image``.
#
#    psf : str
#        Either 'model', or path to the FITS file containing the PSF image.
#
#        If ``psf='model'``, kwargs ``diameter`` and ``strehl`` must
#        not be None. The PSF will then be modeled as a superposition
#        of an Airy pattern (PSF core) and a broad Gaussian (PSF
#        wings).
#
#        If ``psf`` is the path to a FITS file, kwargs ``hdukw`` and
#        ``pixelscalekw`` must not be None.
#
#    diameter : str
#        Diameter of the aperture, e.g. ``'30 m'``. This argument must
#        not be None if ``psf='model'``.
#
#    strehl : float
#        Strehl ratio of the telescope, e.g. ``0.8``. This argument
#        must not be None if ``psf='model'``.
#
#    hdukw : str | int
#        When ``psf`` is the path to a FITS file, that file contains an
#        image of the PSF to be used. Then, ``hdukw`` specifies the
#        name or ID of the HDU in that FITS file where the PSF image
#        can be found. Example: ``hdukw='psf'`` or ``hdukw=1``.
#
#        This argument must not be None if ``psf`` is the path to a
#        FITS file.
#        
#    pixelscalekw : str
#        Same as with ``hdukw``. kwarg ``pixelscalekw`` specifies the
#        name of the pixel scale card in the HDU specified by ``hdukw``
#        in the FITS file given by ``psf``.
#
#        This argument must not be None if ``psf`` is the path to a
#        FITS file.
#
#    """


def getPSF(psfdict,image):


    """Model or load from file a PSF suitable for `image`.

    Parameters
    ----------
    image : instance
        `Image` instance.

    psfdict : dict
    
    """

    pixelscale = image.pixelscale

    psfobj = psfdict['psf']
    
    if psfobj == 'model':
        npix = image.data.shape[-1]
        wavelength = image.wave
        diameter = psfdict['diameter']
        strehl = psfdict['strehl']
        
        image_psf = modelPSF(npix,wavelength=wavelength,\
                             diameter=diameter,strehl=strehl,\
                             pixelscale=pixelscale)

    elif psfobj.endswith('.fits'): # PSF model from fits file; must have keyword PIXELSCL
        
        image_psf, pixelscale_psf = loadPSFfromFITS(psfobj,psfdict)
        
        if pixelscale_psf != pixelscale:
            image_psf, _newfactor, aux = resampleImage(image_psf,pixelscale_psf/pixelscale)

    return PSF(image_psf,str(image.pixelscale))


def loadPSFfromFITS(fitsfile,psfdict):

    hdukw = psfdict['hdukw']
    pixelscalekw = psfdict['pixelscalekw']
    
    header = pyfits.getheader(fitsfile,hdukw)
    pixelscale_psf = header[pixelscalekw]  # currently assuming that pixelscale in the FITS file is in arcsec
        
    image_psf = pyfits.getdata(fitsfile,hdukw)

    return image_psf, pixelscale_psf

        
def modelPSF(npix,wavelength='1.25 micron',diameter='30 m',strehl=0.8,pixelscale='0.01 arcsec'):

    """PSF modeling of CLUMPY images given a specific telescope and wavelength

    Parameters
    ----------
    npix : int
        Size of PSF image to produce (in pixels). The PSF image will
        be ``npix`` x ``npix`` large.

    wavelength: str | instance
        Wavelength at which to create PSF, e.g. '2.2 micron'. Can also
        be instance of :class:`astropy.units.quantity.Quantity`.

    diameter: array
        Diameter of the telescope, e.g. '30 m'. Can also be instance
        of :class:`astropy.units.quantity.Quantity`.

    strehl: float
        Strehl ratio of the model PSF.
        
    pixelscale: str | instance
        Angular scale of the instrument, per pixel, e.g. '0.1 arcsec'
        (the 'per pixel' is assumed implicitly). Can also be instance
        of :class:`astropy.units.quantity.Quantity`
        
    Example
    -------
    tbd

    """

    # checks and units conversion
    checkOdd(npix)
    
    wavelength = (getQuantity(wavelength,recognized_units=UNITS['WAVE']))
    diameter = (getQuantity(diameter,recognized_units=UNITS['LINEAR']))
    pixelscale = (getQuantity(pixelscale,recognized_units=UNITS['ANGULAR']))
    
    # Position of Gaussian at the center of the array. Coordinates, x_mean, y_mean,
    # must be loaded from Hypercat to be equal to the dimensions of the clumpy models
    y, x = N.mgrid[:npix, :npix] # this should be the dimension of the hypercat array

    # 2D AiryDisk: Halo of PSF
    radius = ((1.22 * wavelength/diameter)*u.radian) # Radius is the radius of the first zero 1.22 l/D in arcsec
    radius = (radius/pixelscale).decompose() # Core diameter in px
    
    sigma_p_dl = 0.0745                    # Typical diffraction-limited aberration level or Strehl = 0.8
    S_dl = 0.8                             # Diffraction-limited Strengthl of 0.8 to normalize
    sigma_p = strehl * (sigma_p_dl / S_dl) # normalization of the aberration level
    
    # Intensity of the 2D Airy Disk
    aI = N.exp(-sigma_p**2.)

#    a2D = models.AiryDisk2D(amplitude=aI,x_0=npix/2,y_0=npix/2,radius=radius)
    a2D = AiryDisk2D(amplitude=aI,x_0=npix//2,y_0=npix//2,radius=radius)
    a2D = a2D(x,y) # evaluate the 2D Airy disk

    # 2D Gaussian: Core of PSF
    C = get_normalization() # use defaults (r_0 = 0.15m at 0.5um) --> C = 5461692.609078237 m^(-1/5)
    r_o = C * wavelength**(6./5.) # r_o is now normalized assuming r_o = 0.15m at 0.5um

    rho_o = r_o * (1. + 0.37*(r_o/diameter)**(1./3.)) # rho_o for short exposures

    rad_H = ((1.22*(wavelength/diameter) * N.sqrt(1. + (diameter/rho_o)**2.))*u.radian)
    rad_H = (rad_H/pixelscale).decompose() # Halo diameter in px

    # Intensity of the 2D Gaussian
    gI = (1-aI) / (1. + (diameter/rho_o)**2)            

#    g2D = models.Gaussian2D(amplitude=gI,x_mean=npix/2,y_mean=npix/2,x_stddev=rad_H,y_stddev=rad_H,theta=0.)
    g2D = Gaussian2D(amplitude=gI,x_mean=npix//2,y_mean=npix//2,x_stddev=rad_H,y_stddev=rad_H,theta=0.)
    g2D = g2D(x,y) # evaluate the 2D Gaussian

    # Final PSF
    psf = a2D + g2D

    return psf


def get_normalization(r_0='0.15 m',wave='0.5 micron'):

    """Compute PSF normalization.

    Assumes r_o is a function of the wavelength such as r_o = C*lambda^(6/5).

    r_0 is in the range of 0.1-0.2m at 0.5um. Assuming r_0 = 0.15m at
    0.5um, the constant C = r_0/lambda^(6/5) =
    0.15/((0.5E-6)**(6./5.)) = 5461692.609 in meters.

    Parameters
    ----------
    r_0 : str | 'Quantity' instance
        Size of r_0.

    wave : str | 'Quantity' instance
        Wavelength at which r_o has its value.

    Returns
    -------
    C : float
        Normalization C = r_0/lambda^(6/5).

    """

    r_0 = getQuantity(r_0,recognized_units=UNITS['LINEAR'])
    wave = getQuantity(wave,recognized_units=UNITS['WAVE'])
    
    C = r_0 / (wave**(6./5.))  # with defaults: C = 5461692.609078237 m^(-1/5)

    return C


# OLD VERSION; will be removed soon
#def modelPSF(npix,wavelength=1.25,diameter=30.0,strehl=0.8,pixelscale=0.01):
#
#    """ PSF modeling of CLUMPY images given a specific telescope and wavelength
#
#    Parameters
#    ----------
#
#    image : float array (2D)
#        Array obtained from hypercat using get_image module
#
#    wavelength: array
#        List of wavelengths in microns to create PSF
#
#    diameter: array
#        Diameter of the telescope in meters
#
#    strehl: array
#        Strehl of the model PSF
#        
#    pxscale: array
#        Pixel-scale of the instrument in arcsec/px
#        
#    Example
#    -------
#    .. code-block:: python
#        
#        PSF = PSF_model()
#    """
#
#    ###   Position of Gaussian at the center of the array. Coordinates, x_mean, y_mean,
#    #must be loaded from Hypercat to be equal to the dimensions of the clumpy models
#    y, x = N.mgrid[:npix, :npix] #this should be the dimension of the hypercat array
#
#    wavelength = wavelength * 1E-6
#    ### 2D AiryDisk: Halo of PSF
#    radius = 1.22 * wavelength/diameter * 206265  #Radius is the radius of the first zero 1.22 l/D in arcsec
#    radius = radius / pixelscale                     #Core diameter in px
#    
#    sigma_p_dl = 0.0745                           #Typical diffraction-limited aberration level or Strehl = 0.8
#    S_dl       = 0.8                              #Diffraction-limited Strengthl of 0.8 to normalize
#    sigma_p = strehl * (sigma_p_dl / S_dl)        #normalization of the aberration level
#    
#    aI = N.exp(-sigma_p**2.)                      #Intensity of the 2D Airy Disk
#
#    a2D = models.AiryDisk2D(amplitude=aI,x_0=len(x)/2,y_0=len(y)/2,radius=radius)
#    a2D = a2D(x,y)                                # This is the 2D Airy disk
#
#    ### 2D Gaussian: Core of PSF
#    r_o = 5461692.609 * (wavelength)**(6./5.)           #Normalized assuming r_o = 0.15m at 0.5um, r_o in m
#    rho_o = r_o * (1. + 0.37*(r_o/diameter)**(1./3.))   #rho_o for short exposures
#    #rho_o = r_o                                         #rho_o for long exposures
#    rad_H = 1.22*(wavelength/diameter) * N.sqrt(1. + (diameter/rho_o)**2.) * 206265     #arcsec
#    rad_H = rad_H / pixelscale                             #Halo diameter in px
#    
#    gI = (1-aI) / (1. + (diameter/rho_o)**2)            #Intensity of the 2D Gaussian
#
#    g2D = models.Gaussian2D(amplitude =gI, x_mean=len(x)/2,y_mean=len(y)/2,x_stddev=rad_H,y_stddev=rad_H,theta=0.)
#    g2D = g2D(x,y)                                # This is the 2D Gaussian
#
#    ### Final PSF
#    psf = a2D + g2D
#
##    print('Telescope Diameter [m]=', diameter)
##    print('Angular Resolution ["] =',radius*pixelscale)
##    print('Angular resolution [px] =',radius)
#
#    return psf
