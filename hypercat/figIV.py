#__version__ = '20180624' # yyyymmdd
#__author__ = 'ELR, KI, RN'

#### Who are are you?
user = 'ELR' #Options 'ELR', 'KI', 'RN'

#IMPORTS
import os as os    
import sys

if user == 'ELR':
    DIR = '/Users/elopezro/Documents/GitHub/hypercat/' #ELR local HyperCAT folder
if user == 'KI':
    DIR = '' #KI local HyperCAT folder
if user == 'RN':
    DIR = '/home/robert/dev/hypercat/' # RN local HyperCAT folder
    
sys.path.append(DIR+'/hypercat')  
    
# 3rd party
import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as fits
import pandas as pd
from scipy import ndimage
from astropy.convolution import convolve_fft
import astropy.io.ascii as ascii
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from skimage import restoration
import astropy.units as u
import os as os

# Hypercat
import hypercat as hc
import plotting

# CONSTANTS
if user == 'ELR':
    cubefile = '/Volumes/Seagate Backup Plus Drive/hypercat/hypercat_20170827.hdf5' #ELR local hdf5 file
if user == 'KI':
    cubefile = '' #KI local hdf5 file
if user == 'RN':
    cubefile = '/media/robert/ntoshiba3tb/work_takewithme/hypercat/hypercat_20170827.hdf5' # RN local hdf5 file


 ### INPUTS from user
#NGC1068 json file with parameters from LR18
subcube = DIR+'/examples/subcube_ngc1068.json' 
#Figure name
figname = 'figIV.pdf'

#NGC1068 physical parameters
lum  = '1.6e45 erg/s'
dis  = '12.5 Mpc'
objname = 'ngc1068'
pa   = '42 deg'

#Best inferred model from LR+18
sig  = 43.
inc  = 75.
Y    = 18.
N    = 4.
q    = 0.08
tauv = 70.

#Fluxes from LR+18
fluxes = np.array(['8.4 mJy','22 mJy','98 mJy','1000 mJy','2500 mJy','5600 mJy','6979 mJy','6193 mJy','11567 mJy'])

#Instrumental configuration
telescope = ['JWST','Keck','GMT','TMT','ELT']
wavelength = np.array([1.2,1.6,2.2,3.45,4.75,8.8,9.6,10.3,11.6])

#SNR 
snr = 40

ii = 3
jj = 3
### Automatic figure generator
#Load cube
print('--- Loading cube ---')
cube = hc.ModelCube(hdffile=cubefile,hypercube='imgdata',subcube_selection=subcube)
print('-- DONE')
print('--------------------')
#Generate AGN
print('-- Generating object ---')
agn = hc.Source (cube, luminosity=lum, distance=dis, objectname=objname, pa=pa)
print('-- DONE')
print('------------------------')
### HyperCAT ###
print('-- Generating sky image ---')
#Vector with best inferred model
vec = (sig,inc,Y,N,q,tauv,wavelength[ii])
#Generate image on the sky
sky = agn(vec,total_flux_density=fluxes[ii])
print('--------------------------')
#Generate Telescope configuration
print('-- Generating Telescope configuration ---')
Telescope = hc.Telescope(psfdict={'psf':'pupil','telescope':telescope[jj]},\
                   telescope=telescope[jj],instrument=None,pixelscale_detector='Nyquist')
print('--------------------------')
#Generate Synthetic observations
print('-- Generating synthetic observations ---')
obs_pupil, psf_pupil, psf_pupil_resample = Telescope.observe(sky,snr=None)
obs_conv = psf_pupil.convolve(sky.I)
print('--------------------------')
#Generate Synthetic observations
print('-- Generating synthetic observations with noise ---')
obs_pupil_noisy, psf_pupil, psf_pupil_resample = Telescope.observe(sky,snr=40)
print('--------------------------')
print('-- Deconvolve synchetic observations ---')
obs_deconv = psf_pupil_resample.deconvolve(obs_pupil)
print('--------------------------')
print('-- Deconvolve synthetic observations with noise ---')
obs_deconv_noisy = psf_pupil_resample.deconvolve(obs_pupil_noisy)
print('--------------------------')



fig, axs = plt.subplots(2,4,figsize=(22,9))
ax = axs.flatten()

FOV = sky.pixelscale.value * sky.I.shape[0] /2


### Model image
ima = sky.I/sky.pixelarea.value
levels = np.nanmax(ima)*np.arange(0.01,0.9,0.03)
cbar = ax[0].imshow(ima,origin='lower',interpolation='Nearest',
            extent=[-FOV,FOV,-FOV,FOV],vmin=0,vmax=0.1)
plt.colorbar(cbar,ax=ax[0],fraction=0.046, pad=0.04).set_label('Flux (mJy/sqmas)',fontsize=12)
ax[0].contour(ima,levels=levels,colors='white',alpha=0.5,\
            extent=[-FOV,FOV,-FOV,FOV])
ax[0].tick_params(labelsize=12)
ax[0].set_xlabel('Offsets (mas)',fontsize=15)
ax[0].set_ylabel('Offsets (mas)',fontsize=15)
ax[0].set_title('Model image',fontsize=20)
ax[0].text(-160,-155,np.str(wavelength[ii])+' $\\mu$m',color='white',weight='bold',fontsize=15)
ax[0].text(-160,145,'A',color='white',weight='bold',fontsize=15)


### PSF (model sampling)
psf_model = np.log10(np.abs(psf_pupil.I/np.max(psf_pupil.I)))
res = np.round((206265*wavelength[ii]*1E-6/30.)*1000.,1)
vmin =-3
vmax= 0
cbar = ax[1].imshow(psf_model,origin='lower',interpolation='Nearest',
            extent=[-FOV,FOV,-FOV,FOV],vmin=vmin,vmax=vmax)
plt.colorbar(cbar,ax=ax[1],fraction=0.046, pad=0.04).set_label('Normalized log(PSF)',fontsize=12)
ax[1].tick_params(labelsize=12)
ax[1].set_xlabel('Offsets (mas)',fontsize=15)
ax[1].get_yaxis().set_visible(False)
ax[1].set_title('PSF',fontsize=20)
ax[1].text(-160,-155,np.str(telescope[jj])+' ('+np.str(res)+' mas)',color='white',weight='bold',fontsize=15)
ax[1].text(-160,145,'B',color='white',weight='bold',fontsize=15)


###  Convolved image
obs_c = obs_conv/sky.pixelarea.value
levels = np.nanmax(obs_c)*np.arange(0.1,0.9,0.1)
cbar = ax[2].imshow(obs_c,origin='lower',interpolation='Nearest',
            extent=[-FOV,FOV,-FOV,FOV],vmin=0,vmax=0.05)
plt.colorbar(cbar,ax=ax[2],fraction=0.046, pad=0.04).set_label('Flux Density (mJy/sqmas)',fontsize=12)
ax[2].contour(obs_c,levels=levels,colors='white',alpha=0.5,\
            extent=[-FOV,FOV,-FOV,FOV])
ax[2].tick_params(labelsize=12)
ax[2].set_xlabel('Offsets (mas)',fontsize=15)
ax[2].get_yaxis().set_visible(False)
ax[2].set_title('Convolved image',fontsize=20)
ax[2].text(-160,145,'C',color='white',weight='bold',fontsize=15)


### Pixelated convolved image
obs = obs_pupil.I/obs_pupil.pixelarea.value
levels = np.nanmax(obs)*np.arange(0.1,0.9,0.1)
cbar = ax[3].imshow(obs,origin='lower',interpolation='Nearest',
            extent=[-FOV,FOV,-FOV,FOV],vmin=0,vmax=0.05)
plt.colorbar(cbar,ax=ax[3],fraction=0.046, pad=0.04).set_label('Flux Density (mJy/sqmas)',fontsize=12)
ax[3].contour(obs,levels=levels,colors='white',alpha=0.5,\
            extent=[-FOV,FOV,-FOV,FOV])
ax[3].tick_params(labelsize=12)
ax[3].set_xlabel('Offsets (mas)',fontsize=15)
ax[3].get_yaxis().set_visible(False)
ax[3].set_title('Pixelated convolved image',fontsize=20)
ax[3].text(-160,145,'D',color='white',weight='bold',fontsize=15)

### Noisy image
ima_noisy = obs_pupil_noisy.I/obs_pupil_noisy.pixelarea.value
levels = np.nanmax(ima_noisy)*np.arange(0.1,0.9,0.1)
cbar = ax[4].imshow(ima_noisy,origin='lower',interpolation='Nearest',
            extent=[-FOV,FOV,-FOV,FOV],vmin=0,vmax=0.05)
plt.colorbar(cbar,ax=ax[4],fraction=0.046, pad=0.04).set_label('Flux Density (mJy/sqmas)',fontsize=12)
ax[4].contour(ima_noisy,levels=levels,colors='white',alpha=0.5,\
            extent=[-FOV,FOV,-FOV,FOV])
ax[4].tick_params(labelsize=12)
ax[4].set_xlabel('Offsets (mas)',fontsize=15)
ax[4].set_ylabel('Offsets (mas)',fontsize=15)
ax[4].set_title('Noisy image',fontsize=20)
ax[4].text(-160,145,'E',color='white',weight='bold',fontsize=15)
ax[4].text(-160,-155,'SNR = '+np.str(snr),color='white',weight='bold',fontsize=15)

### PSF (pixelated sampling)
psf_model = np.log10(np.abs(psf_pupil_resample.I/np.max(psf_pupil_resample.I)))
vmin =-3
vmax= 0
cbar = ax[5].imshow(psf_model,origin='lower',interpolation='Nearest',
            extent=[-FOV,FOV,-FOV,FOV],vmin=vmin,vmax=vmax)
plt.colorbar(cbar,ax=ax[5],fraction=0.046, pad=0.04).set_label('Normalized log(PSF)',fontsize=12)
ax[5].tick_params(labelsize=12)
ax[5].set_xlabel('Offsets (mas)',fontsize=15)
ax[5].get_yaxis().set_visible(False)
ax[5].set_title('Pixelated PSF',fontsize=20)
ax[5].text(-160,145,'F',color='white',weight='bold',fontsize=15)


### Deconvolved image
ima_deconv = obs_deconv/obs_pupil_noisy.pixelarea.value
levels = np.nanmax(ima_deconv)*np.array([0.01,0.05,0.1,0.3,0.5,0.7,0.9])
cbar = ax[6].imshow(ima_deconv,origin='lower',interpolation='Nearest',
            extent=[-FOV,FOV,-FOV,FOV],vmin=0,vmax=0.005)
plt.colorbar(cbar,ax=ax[6],fraction=0.046, pad=0.04).set_label('Flux Density (mJy/sqmas)',fontsize=12)
ax[6].contour(ima_deconv,levels=levels,colors='white',alpha=0.5,\
            extent=[-FOV,FOV,-FOV,FOV])
ax[6].tick_params(labelsize=12)
ax[6].set_xlabel('Offsets (mas)',fontsize=15)
ax[6].set_ylabel('Offsets (mas)',fontsize=15)
ax[6].set_title('Deconvolded pixelated image',fontsize=20)
ax[6].text(-160,145,'G',color='white',weight='bold',fontsize=15)


### Deconvolved noisy image
ima_deconv_noisy = obs_deconv_noisy/obs_pupil_noisy.pixelarea.value
levels = np.nanmax(ima_deconv_noisy)*np.array([0.01,0.05,0.1,0.3,0.5,0.7,0.9])
cbar = ax[7].imshow(ima_deconv_noisy,origin='lower',interpolation='Nearest',
            extent=[-FOV,FOV,-FOV,FOV],vmin=0,vmax=0.005)
plt.colorbar(cbar,ax=ax[7],fraction=0.046, pad=0.04).set_label('Flux Density (mJy/sqmas)',fontsize=12)
ax[7].contour(ima_deconv_noisy,levels=levels,colors='white',alpha=0.5,\
            extent=[-FOV,FOV,-FOV,FOV])
ax[7].tick_params(labelsize=12)
ax[7].set_xlabel('Offsets (mas)',fontsize=15)
ax[7].set_ylabel('Offsets (mas)',fontsize=15)
ax[7].set_title('Deconvolved noisy image',fontsize=20)
ax[7].text(-160,145,'H',color='white',weight='bold',fontsize=15)


fig.tight_layout()
fig.savefig(figname,dpi=300)
os.system('open '+figname)

