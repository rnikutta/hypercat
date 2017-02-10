import numpy as N
import astropy.io.fits as pyfits
from astropy.modeling import models, fitting
from astropy.convolution import convolve, convolve_fft
from imageops import *

__author__ = "Enrique Lopez-Rodriguez <enloro@gmail.com>, Robert Nikutta <robert.nikutta@gmail.com>"
__version__ = '20170207'  #yyyymmdd

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
#
#        if hasattr(image,'pa') and image.pa.value != 0.:
#            psfdata = rotateImage(psf.data,imaga.pa.to('deg').value)
#        else:
#            psfdata = self.data
            

        result = convolve_fft(image,self.data,normalize_kernel=True,allow_huge=True)
#        result = convolve_fft(image,psfdata,normalize_kernel=True,allow_huge=True)

        return result


def getPSF(psfdict,image):


    """Model or load from file a PSF suitable for `image`.

    Parameters
    ----------
    image : instance
        `Image` instance.

    psfdict : dict
    
    """

    pixelscale = image.pixelscale.to('arcsec').value

    psfobj = psfdict['psf']
    
    if psfobj == 'model':
        npix = image.data.shape[-1]
        wavelength = image.wave.to('micron').value
        pixelscale = image.pixelscale.to('arcsec').value
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
    

        
def modelPSF(npix,wavelength=1.25,diameter=30.0,strehl=0.8,pixelscale=0.01):

    """ PSF modeling of CLUMPY images given a specific telescope and wavelength

    Parameters
    ----------

    image : float array (2D)
        Array obtained from hypercat using get_image module

    wavelength: array
        List of wavelengths in microns to create PSF

    diameter: array
        Diameter of the telescope in meters

    strehl: array
        Strehl of the model PSF
        
    pxscale: array
        Pixel-scale of the instrument in arcsec/px
        
    Example
    -------
    .. code-block:: python
        
        PSF = PSF_model()
    """

    ###   Position of Gaussian at the center of the array. Coordinates, x_mean, y_mean,
    #must be loaded from Hypercat to be equal to the dimensions of the clumpy models
    y, x = N.mgrid[:npix, :npix] #this should be the dimension of the hypercat array

    wavelength = wavelength * 1E-6
    ### 2D AiryDisk: Halo of PSF
    radius = 1.22 * wavelength/diameter * 206265  #Radius is the radius of the first zero 1.22 l/D in arcsec
    radius = radius / pixelscale                     #Core diameter in px
    
    sigma_p_dl = 0.0745                           #Typical diffraction-limited aberration level or Strehl = 0.8
    S_dl       = 0.8                              #Diffraction-limited Strengthl of 0.8 to normalize
    sigma_p = strehl * (sigma_p_dl / S_dl)        #normalization of the aberration level
    
    aI = N.exp(-sigma_p**2.)                      #Intensity of the 2D Airy Disk

    a2D = models.AiryDisk2D(amplitude=aI,x_0=len(x)/2,y_0=len(y)/2,radius=radius)
    a2D = a2D(x,y)                                # This is the 2D Airy disk

    ### 2D Gaussian: Core of PSF
    r_o = 5461692.609 * (wavelength)**(6./5.)           #Normalized assuming r_o = 0.15m at 0.5um, r_o in m
    rho_o = r_o * (1. + 0.37*(r_o/diameter)**(1./3.))   #rho_o for short exposures
    #rho_o = r_o                                         #rho_o for long exposures
    rad_H = 1.22*(wavelength/diameter) * N.sqrt(1. + (diameter/rho_o)**2.) * 206265     #arcsec
    rad_H = rad_H / pixelscale                             #Halo diameter in px
    
    gI = (1-aI) / (1. + (diameter/rho_o)**2)            #Intensity of the 2D Gaussian

    g2D = models.Gaussian2D(amplitude =gI, x_mean=len(x)/2,y_mean=len(y)/2,x_stddev=rad_H,y_stddev=rad_H,theta=0.)
    g2D = g2D(x,y)                                # This is the 2D Gaussian

    ### Final PSF
    psf = a2D + g2D

#    print 'Telescope Diameter [m]=', diameter
#    print 'Angular Resolution ["] =',radius*pixelscale
#    print 'Angular resolution [px] =',radius

    return psf
