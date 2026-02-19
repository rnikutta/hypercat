__author__ = "Enrique Lopez-Rodriguez <enloro@gmail.com>, Robert Nikutta <robert.nikutta@gmail.com>"
__version__ = '20210615' #yyyymmdd

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
from copy import copy
import logging

import numpy as np
from astropy.modeling.models import AiryDisk2D, Gaussian2D
from astropy.convolution import convolve, convolve_fft
from astropy.io import fits
#import astropy.io.ascii as ascii
#from scipy import ndimage
from skimage import restoration
import json

#from imageops import *
#from units import *
#from utils import get_rootdir

from .imageops import *
from .units import *
from .utils import get_rootdir

#rootdir = get_rootdir()
#rootdir = '/Users/elopezro/Documents/GitHub/hypercat/'

rootdir = get_rootdir()

"""Utilities for the PSF analysis of the images created by hyperCAT

    .. automodule:: PSF_modeling
"""

def fft_pxscale(header,wave):

    """Compute conversion scale from telescope space to sky space.

       Parameters
       ----------
       ima : array
           2D Telescope pupil model.

       Returns
       -------
       fftscale : float
           The frequency scale in sky space.

       Example
       -------
       .. code-block:: python

          fftscale = fft_pxscale(ima)

    """

    #size of the image. This should be taken from the header.
    gridsize = header['NAXIS1']
    #pixel scale of the image. This should be taken from the header.
    pxscale_mod = header['PIXSCALE']    #in meters/px
    #1D FFT of the gridsize.
    fft_freq=np.fft.fftfreq(gridsize,pxscale_mod)
    #wavelength of the desires psf. This is a input of the user, wavelength in microns
    wave = (getQuantity(wave,recognized_units=UNITS['WAVE']))
    lam = wave.to(u.m)                 #in meters
    #re-orginizing the 1D FFT to match with the grid.
    roll=np.floor(gridsize//2).astype("int")
    freq = np.fft.fftshift(fft_freq)
    ##
    ## pxscale -> fftscale
    fftscale=np.diff(freq)[0]           ## cycles / mas per pixel in FFT image
    mas2rad=np.deg2rad(1./3600000.)     ## mas per rad
    fftscale = fftscale/mas2rad * lam   ## meters baseline per px in FFT image at a given wavelength
    logging.info("Pixel scale in PSF image is: %g mas per pixel" % fftscale.value)
    return fftscale.value


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

        result = convolve_fft(image,self.data/self.data.max(),normalize_kernel=True,allow_huge=True)

        return result


#    def deconvolve(self,image,niter):
#        ima = image.data/np.sum(image.data)
#        psf = self.data/np.sum(self.data)
#        result = restoration.richardson_lucy(ima, psf, iterations=niter)
#        return result[::-1,::-1]

#    def deconvolve(self,image,niter):
#        # TODO: refactor such that this func can return ndarray; package into an class::`Image` container outside
#        _image = copy(image)
##        print("_image.data.sum()", _image.data.sum())
#        _unit = _image.data.unit # save units for later...
#        ima = _image.data #/np.sum(_image.data)
##        print("ima.sum()", ima.sum())
#        psf = self.data/np.sum(self.data)
##        print("psf.sum()", psf.sum())
#        result = restoration.richardson_lucy(ima, psf, iterations=niter)
##        print("result.sum()", result.sum())
#        _image.data = result * _unit  # ... re-apply units
##        print("_image.data.sum()", _image.data.sum())
#        return _image


    def deconvolve(self,image,niter):
        # TODO: refactor such that this func can return ndarray; package into an class::`Image` container outside
        _image = copy(image)
        print("BEFORE: psf.PSF.deconvolve(): _image.data.max() = ", _image.data.max())
#        print("_image.data.sum()", _image.data.sum())
        _unit = _image.data.unit # save units for later...
        ima = _image.data.value #/np.sum(_image.data)
        print("MIDDLE: psf.PSF.deconvolve(): ima.max() = ", ima.max())
#        print("ima.sum()", ima.sum())
        psf = self.data/np.sum(self.data)
#        print("psf.sum()", psf.sum())
        result = restoration.richardson_lucy(ima, psf, iterations=niter)
        print("MIDDLE: psf.PSF.deconvolve(): result.max() = ", result.max())
#        print("result.sum()", result.sum())
        _image.data = result * _unit  # ... re-apply units
#        print("_image.data.sum()", _image.data.sum())
        print("AFTER: psf.PSF.deconvolve(): _image.data.max() = ", _image.data.max())
        return _image

    
    
def getPupil(psfdict):
    pupilfile = rootdir+'data/pupils.json'
    with open(pupilfile,'r') as f:
        pupildict = json.load(f)
    pupil_fitsfile = pupildict[psfdict['telescope']]['file']
    data, header = fits.getdata(rootdir+pupil_fitsfile,header=True)
    return data, header


def getPSF(psfdict):

    """Model or load from file a PSF suitable for `image`.

    Parameters
    ----------
    psfdict : dict

    """

    psfmode = psfdict['psfmode']

    #Model-PSF
    if psfmode == 'model':
        image_psf = modelPSF(npix = psfdict['npix'],\
                             wavelength=psfdict['wavelength'],\
                             diameter=psfdict['diameter'],\
                             strehl=psfdict['strehl'],\
                             pixelscale=psfdict['pixelscale'])
        pixelscale = (getQuantity(psfdict['pixelscale'],recognized_units=UNITS['ANGULAR']))
        pixelscale_psf = pixelscale.to(u.mas).value
        wave = psfdict['wavelength']

    #Pupil-PSF
    if psfmode == 'pupil':
        #Obtain Pupil
        data, header = getPupil(psfdict)
        #Compute PSF and obtain PSF pixelscale
        pixelscale_psf = fft_pxscale(header,psfdict['wavelength'])  #mas/px
        image_psf = np.abs(np.fft.fftshift(np.fft.fft2(data)))

        # trim image_psf array to be an odd-dimensioned square with the highest pixel value at the exact center pixel
        image_psf = trim_square_odd(image_psf)

        #PSF with the pixelscale
        image_psf = Image(image_psf,pixelscale=np.str(pixelscale_psf)+' mas')
        #Normalization of the PSF
        image_psf = image_psf.I / np.max(image_psf.I)
        wave = psfdict['wavelength']

    #Image-PSF
    if psfmode.endswith('.fits'): # PSF model from fits file; must have keyword PIXELSCALE
        # TODO: get the below to work
        image_psf, header = loadPSFfromFITS(psfobj,psfdict)
        #image_psf, _newfactor, aux = resampleImage(image_psf,pixelscale_psf/pixelscale)
        wave = header['wave']

    #image_psf = force_ood(image_psf)

    psf_ = PSF(image_psf,str(pixelscale_psf*u.mas))
    psf_.wave = wave

    return psf_


def loadPSFfromFITS(fitsfile,psfdict):

    hdukw = psfdict['hdukw']
    pixelscalekw = psfdict['pixelscale']
    image_psf, header = fits.getdata(fitsfile,hdukw,header=True)

    return image_psf, header


def modelPSF(npix=241,wavelength='2.2 micron',diameter='30 m',strehl=0.8,pixelscale='1 mas'):

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

    """

    # checks and units conversion
    checkOdd(npix)


    wavelength = (getQuantity(wavelength,recognized_units=UNITS['WAVE']))
    diameter = (getQuantity(diameter,recognized_units=UNITS['LINEAR']))
    pixelscale = (getQuantity(pixelscale,recognized_units=UNITS['ANGULAR']))

    wavelength = wavelength.to(u.m).value
    diameter   = diameter.to(u.m).value
    pixelscale = pixelscale.to(u.arcsec).value

    # Position of Gaussian at the center of the array. Coordinates, x_mean, y_mean,
    # must be loaded from Hypercat to be equal to the dimensions of the clumpy models
    y, x = np.mgrid[:npix, :npix] # this should be the dimension of the hypercat array

    # 2D AiryDisk: Halo of PSF
    radius = (206265*(wavelength/diameter)) # Radius is the radius of the first zero l/D in arcsec
    radius = (radius/pixelscale) # Core diameter in px

    sigma_p_dl = 0.0745                    # Typical diffraction-limited aberration level or Strehl = 0.8
    S_dl = 0.8                             # Diffraction-limited Strengthl of 0.8 to normalize
    sigma_p =   (sigma_p_dl / S_dl) / strehl # normalization of the aberration level

    # Intensity of the 2D Airy Disk
    aI = np.exp(-sigma_p**2.)

    a2D = AiryDisk2D(amplitude=aI,x_0=npix//2,y_0=npix//2,radius=radius)
    a2D = a2D(x,y) # evaluate the 2D Airy disk

    # 2D Gaussian: Core of PSF
    C = get_normalization().value # use defaults (r_0 = 0.15m at 0.5um) --> C = 5461692.609078237 m^(-1/5)
    r_o = C * wavelength**(6./5.) # r_o is now normalized assuming r_o = 0.15m at 0.5um

    rho_o = r_o * (1. + 0.37*(r_o/diameter)**(1./3.)) # rho_o for short exposures

    rad_H = (((wavelength/diameter) * np.sqrt(1. + (diameter/rho_o)**2.)))
    rad_H = (rad_H/pixelscale) # Halo diameter in px

    # Intensity of the 2D Gaussian
    gI = (1-aI) / (1. + (diameter/rho_o)**2)

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
