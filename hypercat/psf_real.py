__version__ = '20180129'   #yyymmdd
__author__ = 'Enrique Lopez-Rodriguez <enloro@gmail.com>'

"""Utilities for handling the real PSF of 30-m telescopes.
    
    .. automodule:: interferometry
"""

# IMPORTS

# 3rd party
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits

#HyperCAT
#import 

# HELPER FUNCTIONS

def fft_pxscale(ima,wave):
    
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
    pxscale_mod = ima[0].header['PIXSCALE']    #in meters
    #1D FFT of the gridsize.
    fft_freq=np.fft.fftfreq(gridsize,pxscale_mod)
    #wavelength of the desires psf. This is a input of the user, wavelength in microns
    lam = wave*1E-6                 #in meters
    #re-orginizing the 1D FFT to match with the grid.
    roll=np.floor(gridsize/2).astype("int")
    freq = np.fft.fftshift(fft_freq)
    ##
    ## pxscale -> fftscale
    fftscale=np.diff(freq)[0]           ## cycles / mas per pixel in FFT image
    mas2rad=np.deg2rad(1./3600000.)     ## mas per rad
    fftscale = fftscale/mas2rad * lam   ## meters baseline per px in FFT image at a given wavelength
    #print "Pixel scale in PSF image is: ", fftscale, " mas per pixel"
    
    return fftscale


def psf_real(telescope='TMT',wavelength=2.2):
    """ Compute PSF from 30-m telescope 
        
       Parameters
       ----------
       telescope : string
             Name of telescope, options are TMT and EELT.
       wavelength : float
             Wavelength of the PSF in microns.
        
       Returns
       -------
       pxscale : float
             Pixelscale of the PSF in mas per pixel.
       psf: 2D array
             PSF of the telescope at the given wavelength.
        
       Example
       -------
       .. code-block:: python
        
          pxscale, psf = psf_real('TMT',2.2)
        
    """

    if telescope == 'TMT':
        TMT_pupil_file = '../psf/TMT_Pupil_Amplitude_Gray_Pixel_Approximated_With_Obscuration.fits'
        TMT_pupil_fits = fits.open(TMT_pupil_file)
        TMT_pupil = TMT_pupil_fits[0].data
        pxscale = fft_pxscale(TMT_pupil_fits,wavelength) #in mas
        psf = np.abs(np.fft.fftshift(np.fft.fft2(TMT_pupil)))
        
        print 'The PSF of TMT was estimated using the pupil image of the TMT'
        print 'with a pixelscale of ',pxscale,' mas per pixel'
        
        
    if telescope == 'EELT':
        EELT_psf_file = '../psf/metis_psf_mag=03.00_seeing=0.80.fits'
        eelt_psf = fits.open(EELT_psf_file)
        eelt_wave = np.array([2.2,3.8,8.65,11.63,12.3,17.8])    #available wavelengths by the E-ELT
        #closest wavelength from that chosen byt he user
        n_wave = np.where(np.abs(eelt_wave-wavelength) == np.min(np.abs(eelt_wave-wavelength)))[0][0]
        wave    = eelt_psf[0].header[7+n_wave]
        pxscale = eelt_psf[0].header[13+n_wave]
        psf     = eelt_psf[0].data[n_wave][0]

        print 'At this time HyperCAT has the PSFs of E-ELT at 2.2, 3.8, 8.65, 11.63, 12.3, 17.8 microns'
        print 'The closest PSF to the ',wavelength,' microns is ',wave,' microns with a '
        print 'pixelscale of ',pxscale,' mas per pixel'

    return pxscale, psf
