#3rd party
import astropy.modeling
import numpy as np
import matplotlib.pyplot as plt
import os
import astropy.io.fits as fits
from astropy.convolution import convolve_fft
from scipy import ndimage
import matplotlib
from skimage import restoration

#HyperCAT
import sys
sys.path.append('/Users/elopezro/Documents/GitHub/hypercat/hypercat/')
import hypercat as hc
import plotting
import psf_real

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
    roll=np.floor(gridsize//2).astype("int")
    freq = np.fft.fftshift(fft_freq)
    ##
    ## pxscale -> fftscale
    fftscale=np.diff(freq)[0]           ## cycles / mas per pixel in FFT image
    mas2rad=np.deg2rad(1./3600000.)     ## mas per rad
    fftscale = fftscale/mas2rad * lam   ## meters baseline per px in FFT image at a given wavelength
    #print("Pixel scale in PSF image is: ", fftscale, " mas per pixel")
    
    return fftscale

def skyimage(ngc1068,sig,inc,Y,N,q,tauv,wave):
    vec = (sig,inc,Y,N,q,tauv,wave)
    sky = ngc1068(vec)
    return sky


def observations(sky,TMT_pupil,wave,dectector_pixel):
    psf_pxscale = fft_pxscale(TMT_pupil_fits,wave) #in mas
    psf = np.abs(np.fft.fftshift(np.fft.fft2(TMT_pupil)))

    #Re-sample PSF
    Dx = np.int(sky.FOV.value/psf_pxscale) 
    xy_p = np.where(psf == np.max(psf))
    psf_FOV = psf[xy_p[0][0]-Dx:xy_p[0][0]+Dx+1,xy_p[1][0]-Dx:xy_p[1][0]+Dx+1]
    resamplingfactor =  psf_pxscale / sky.pixelscale.value /2.
    psf_agnsampled = ndimage.zoom(psf_FOV,resamplingfactor)
    psf_agnsampled = psf_agnsampled/np.max(psf_agnsampled)
    
    #Convolve PSF x AGN
    sky_convolved = convolve_fft(sky.I,psf_agnsampled,normalize_kernel=True,allow_huge=True)

    #Pixelation
    newfactor = sky.pixelscale.value / dectector_pixel 
    image_TMT_MICHI = ndimage.zoom(sky_convolved,newfactor)
    psf_TMT_MICHI = ndimage.zoom(psf_agnsampled,newfactor)
    
    return image_TMT_MICHI,psf_agnsampled

def deconv(obs,psf):
    deconvolved_obs = restoration.richardson_lucy(obs, psf, iterations=50)
    return deconvolved_obs


### Input
figname = 'NGC1068_TMT_MICHI.png'
hypercube = '/Volumes/Seagate Backup Plus Drive/hypercat/hypercat_20170827.hdf5'
subcube   = 'subcube_ngc1068.json'

cube = hc.ModelCube(hdffile=hypercube,hypercube='imgdata',subcube_selection=subcube)

#NGC1068 
lum  = '1.6e45 erg/s'
dis  = '12.5 Mpc'
name = 'ngc1068'
pa   = '42 deg'

ngc1068 = hc.Source (cube, luminosity=lum, distance=dis, objectname=name, pa=pa)

#Best fit from LR+18
sig  = 43.
inc  = 75.
Y    = 18.
N    = 4.
q    = 0.08
tauv = 70.

wave = np.array([3.45,4.75,8.7,10.3,11.6])
dectector_pixel = np.array([11.9,11.9,27.5,27.5,27.5])

### PSF
DIR = '/Users/elopezro/Documents/GitHub/hypercat/psf/'
TMT_pupil_file = DIR+'TMT_Pupil_Amplitude_Gray_Pixel_Approximated_With_Obscuration.fits'
TMT_pupil_fits = fits.open(TMT_pupil_file)
TMT_pupil = TMT_pupil_fits[0].data

### Figures
#fig, axs = plt.subplots(3,len(wave),figsize=(8.,5),sharex=True,sharey=True)
fig, axs = plt.subplots(4,len(wave),figsize=(13.,10.5),sharex=True,sharey=True)
ax = axs.flatten()
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

for ii in range(len(wave)):
    sky = skyimage(ngc1068,sig,inc,Y,N,q,tauv,wave[ii])
    image_TMT_MICHI,psf_TMT_MICHI = observations(sky,TMT_pupil,wave[ii],dectector_pixel[ii])
    deconv_obs = deconv(image_TMT_MICHI,psf_TMT_MICHI)

    #FOV
    FOV = sky.I.shape[0]*sky.pixelscale.value/1000./2. #arcsec
    print FOV
    #Sky image
    sky.I = sky.I + 1E-7
    levels = sky.I.max()*np.array([0.05,0.1,0.2,0.3,0.5,0.7,0.9])
    ax[ii].imshow(np.log10(sky.I),origin='lower',interpolation='Nearest',extent=[-FOV, FOV, -FOV, FOV])
    ax[ii].contour(np.log10(sky.I),levels=np.log10(levels),\
                   colors='white',alpha=0.5,extent=[-FOV, FOV, -FOV, FOV])

    #PSF image
    ax[ii+len(wave)].imshow(np.log10(np.abs(psf_TMT_MICHI)),origin='lower',interpolation='Nearest',\
                            extent=[-FOV, FOV, -FOV, FOV],vmin=-3,vmax=0)

    #Final image
    ax[ii+2*len(wave)].imshow(np.log10(np.abs(image_TMT_MICHI)),origin='lower',interpolation='Nearest',\
                              extent=[-FOV, FOV, -FOV, FOV])
    ax[ii+2*len(wave)].contour(np.abs(np.log10(np.abs(image_TMT_MICHI))),9,colors='white',alpha=0.2,\
                              extent=[-FOV, FOV, -FOV, FOV])

    #Deconvolved image
    ax[ii+3*len(wave)].imshow(deconv_obs,origin='lower',interpolation='Nearest',\
                              extent=[-FOV, FOV, -FOV, FOV])
    ax[ii+3*len(wave)].contour(deconv_obs,9,colors='white',alpha=0.2,\
                              extent=[-FOV, FOV, -FOV, FOV])

    #xy labels
    ax[0].set_ylabel('Offset (")')
    ax[5].set_ylabel('Offset (")')
    ax[10].set_ylabel('Offset (")')
    ax[15].set_ylabel('Offset (")')
    ax[ii+3*len(wave)].set_xlabel('Offset (")')

    #labels
    ax[0].text(-0.15,-0.15,'Model',color='white',fontsize=15)
    ax[5].text(-0.15,-0.15,'TMT PSF',color='white',fontsize=15)
    ax[10].text(-0.15,-0.15,'Observation',color='white',fontsize=15)
    ax[15].text(-0.15,-0.12,'Observation',color='white',fontsize=15)
    ax[15].text(-0.15,-0.15,'Deconvolved',color='white',fontsize=15)

    #title
    ax[ii].set_title(np.str(wave[ii])+' $\\mu$m')
     
fig.tight_layout()
fig.savefig(figname,dpi=300)
os.system('open '+figname)
