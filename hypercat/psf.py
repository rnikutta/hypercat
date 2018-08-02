from __future__ import print_function

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import numpy as np
from astropy.modeling.models import AiryDisk2D, Gaussian2D
from astropy.convolution import convolve, convolve_fft
import astropy.io.fits as pyfits
from imageops import *
from imageops import checkOdd
from units import *
import astropy.io.ascii as ascii
from scipy import ndimage



__author__ = "Enrique Lopez-Rodriguez <enloro@gmail.com>, Robert Nikutta <robert.nikutta@gmail.com>"
__version__ = '20180216' #yyyymmdd

"""Utilities for the PSF analysis of the images created by hyperCAT
    
    .. automodule:: PSF_modeling
"""

def fft_pxscale(ima,wave,telescope):
    
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
    gridsize = ima[0].header['NAXIS1']
    #pixel scale of the image. This should be taken from the header.
    if telescope == 'JWST':
            pxscale_mod = ima[0].header['PUPLSCAL']    #in meters
    if telescope != 'JWST':
            pxscale_mod = ima[0].header['PIXSCALE']    #in meters
    #1D FFT of the gridsize.
    fft_freq=np.fft.fftfreq(gridsize,pxscale_mod)
    #wavelength of the desires psf. This is a input of the user, wavelength in microns
    lam = wave.to(u.m)                 #in meters
    #re-orginizing the 1D FFT to match with the grid.
    roll=np.floor(gridsize//2).astype("int")
    freq = np.fft.fftshift(fft_freq)
    ##
    ## pxscale -> fftscale
    fftscale=np.diff(freq)[0]           ## cycles / mas per pixel in FFT image
    mas2rad=np.deg2rad(1./3600000.)     ## mas per rad
    fftscale = fftscale/mas2rad * lam   ## meters baseline per px in FFT image at a given wavelength
    #print("Pixel scale in PSF image is: ", fftscale, " mas per pixel")
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

        result = convolve_fft(image,self.data,normalize_kernel=True,allow_huge=True)

        return result


def getPSF(psfdict,image):


    """Model or load from file a PSF suitable for `image`.

    Parameters
    ----------
    image : instance
        `Image` instance.

    psfdict : dict
    
    """

    pixelscale = image.pixelscale
    wavelength = image.wave

    psfobj = psfdict['psf']

    #Model-PSF
    if psfobj == 'model':
        npix = image.data.shape[-1]
        diameter = psfdict['diameter']
        strehl = psfdict['strehl']
        image_psf = modelPSF(npix,wavelength=wavelength,\
                             diameter=diameter,strehl=strehl,\
                             pixelscale=pixelscale)
        
    #Pupil-PSF
    if psfobj == 'pupil':
        #Obtain Pupil
        DIR = '/Users/elopezro/Documents/GitHub/hypercat/'
        pupilfile = DIR+'data/pupils.csv'
        pupil_info = ascii.read(pupilfile)
        pupil_fitsfile = pupil_info['Pupil_Image'][pupil_info['Telescope'] == psfdict['telescope']][0]
        pupil_fits  = pyfits.open(DIR+pupil_fitsfile)
        #Compute PSF and obtain PSF pixelscale
        pixelscale_psf = fft_pxscale(pupil_fits,wavelength,psfdict['telescope'])  #mas/px
        image_psf = np.abs(np.fft.fftshift(np.fft.fft2(pupil_fits[0].data)))
        #Re-sample PSF to the sky image pixelscale
        image_psf = Image(image_psf,pixelscale=np.str(pixelscale_psf)+' mas')
        #image_psf.changeFOV(FOV)
        image_psf = image_psf.I
               
    #Image-PSF
    if psfobj.endswith('.fits'): # PSF model from fits file; must have keyword PIXELSCALE
        image_psf, pixelscale_psf = loadPSFfromFITS(psfobj,psfdict)
        #image_psf, _newfactor, aux = resampleImage(image_psf,pixelscale_psf/pixelscale)

    return PSF(image_psf,str(pixelscale_psf * u.mas))


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
        
    """

    # checks and units conversion
    checkOdd(npix)
    
    wavelength = (getQuantity(wavelength,recognized_units=UNITS['WAVE']))
    diameter = (getQuantity(diameter,recognized_units=UNITS['LINEAR']))
    pixelscale = (getQuantity(pixelscale,recognized_units=UNITS['ANGULAR']))
    
    # Position of Gaussian at the center of the array. Coordinates, x_mean, y_mean,
    # must be loaded from Hypercat to be equal to the dimensions of the clumpy models
    y, x = np.mgrid[:npix, :npix] # this should be the dimension of the hypercat array

    # 2D AiryDisk: Halo of PSF
    radius = ((1.22 * wavelength/diameter)*u.radian) # Radius is the radius of the first zero 1.22 l/D in arcsec
    radius = (radius/pixelscale).decompose() # Core diameter in px
    
    sigma_p_dl = 0.0745                    # Typical diffraction-limited aberration level or Strehl = 0.8
    S_dl = 0.8                             # Diffraction-limited Strengthl of 0.8 to normalize
    sigma_p =   (sigma_p_dl / S_dl) / strehl # normalization of the aberration level
    
    # Intensity of the 2D Airy Disk
    aI = np.exp(-sigma_p**2.)

    a2D = AiryDisk2D(amplitude=aI,x_0=npix//2,y_0=npix//2,radius=radius)
    a2D = a2D(x,y) # evaluate the 2D Airy disk

    # 2D Gaussian: Core of PSF
    C = get_normalization() # use defaults (r_0 = 0.15m at 0.5um) --> C = 5461692.609078237 m^(-1/5)
    r_o = C * wavelength**(6./5.) # r_o is now normalized assuming r_o = 0.15m at 0.5um

    rho_o = r_o * (1. + 0.37*(r_o/diameter)**(1./3.)) # rho_o for short exposures

    rad_H = ((1.22*(wavelength/diameter) * np.sqrt(1. + (diameter/rho_o)**2.))*u.radian)
    rad_H = (rad_H/pixelscale).decompose() # Halo diameter in px

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
