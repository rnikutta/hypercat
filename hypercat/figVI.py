#__version__ = '20180626' # yyyymmdd
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
figname = 'figVI.pdf'

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
telescope = ['JWST','KECK','GMT','TMT','ELT']
wavelength = np.array([1.2,1.6,2.2,3.45,4.75,8.8,9.6,10.3,11.6])

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
fontsize_label = 40
fontsize_title = 40
labelsize_ticks = 20
labelsize_cbar = 20

#fig, axes = plt.subplots(len(wavelength)+1,len(telescope)+1,figsize=(37,9*len(wavelength)))
fig, axes = plt.subplots(len(wavelength)+1,len(telescope)+1,figsize=(9.*(len(telescope)+1),9.1*len(wavelength)))
ax = axes.flatten()

for jj in range(len(telescope)):
    for ii in range(len(wavelength)):
        ### HyperCAT ###
        print('-- Generating sky image ---')
        #Vector with best inferred model
        vec = (sig,inc,Y,N,q,tauv,wavelength[ii])
        #Generate image on the sky
        sky = agn(vec,total_flux_density=fluxes[ii])
        print('--------------------------')
        #Generate Pupil-PSF: This is only needed because we want to plot 
        #the pupil images and PSF. if not desired, then go to next step.
        print('-- Generating Pupil-PSF---')
        PupilPSF = hc.PupilPSF(psfdict={'psf':'pupil','telescope':telescope[jj]},\
                   telescope=telescope[jj],instrument=None,pixelscale_detector='Nyquist')
        pupil_image, psf_image, pixelscale_pupil, pixelscale_psf = \
        PupilPSF.pupilpsf(wavelength[ii] * u.um)
        #Generate Telescope configuration
        print('-- Generating Telescope configuration ---')
        Telescope = hc.Telescope(psfdict={'psf':'pupil','telescope':telescope[jj]},\
                   telescope=telescope[jj],instrument=None,pixelscale_detector='Nyquist')
        print('--------------------------')
        #Generate Synthetic observations
        print('-- Generating synthetic observations ---')
        if ((PupilPSF.telescope == 'JWST') & (sky.wave_ == 11.6)):
            obs_pupil, psf_pupil, psf_pupil_resample = Telescope.observe(sky,snr=None)
        else:
            obs_pupil, psf_pupil, psf_pupil_resample = Telescope.observe(sky,snr=40)
        print('--------------------------')
        print('-- Deconvolve synthetic observations ---')
        obs_deconv = psf_pupil_resample.deconvolve(obs_pupil)
        print('--------------------------')

        #################
        
        ### Figures ###
        ## Pupil images
        FOV = pixelscale_pupil * pupil_image.shape[0]/2
        ax[jj+1].imshow(pupil_image,origin='lower',interpolation='Nearest',\
                 vmin=0,vmax=1,extent=[-FOV,FOV,-FOV,FOV])
        ax[jj+1].set_xlabel('meters',fontsize=fontsize_label)
        ax[jj+1].set_ylabel('meters',fontsize=fontsize_label)
        ax[jj+1].set_title(telescope[jj],fontsize=fontsize_title)
        ax[jj+1].tick_params(labelsize=labelsize_ticks)
        
        
        ## Sky images
        sky_Jysqmas = sky.I/sky.pixelarea.value
        levels = np.nanmax(sky_Jysqmas)*np.array([0.05,0.1,0.3,0.5,0.7,0.9])
        FOV = sky.pixelscale.value * sky.I.shape[0]/2
        if jj == 0:
            cbar = ax[((len(telescope)+1)*ii)+(len(telescope)+1)].imshow(sky_Jysqmas,origin='lower',\
                                                                  interpolation='Nearest',extent=[-FOV,FOV,-FOV,FOV])
            cbar = fig.colorbar(cbar,ax=ax[((len(telescope)+1)*ii)+(len(telescope)+1)],\
                        fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel('Flux (mJy/sqmas)',size=fontsize_label)
            cbar.ax.tick_params(labelsize=labelsize_cbar)
            
            ax[((len(telescope)+1)*ii)+(len(telescope)+1)].contour(sky_Jysqmas,levels=levels,colors='white',alpha=0.5,\
                                                                 extent=[-FOV,FOV,-FOV,FOV])
            
            ax[((len(telescope)+1)*ii)+(len(telescope)+1)].text(-120,120,np.str(wavelength[ii])+' $\mu$m',\
                                                                color='white',fontsize=fontsize_label,weight='bold')

            if ii < len(wavelength)-1:
                ax[((len(telescope)+1)*ii)+(len(telescope)+1)].get_xaxis().set_visible(False)
                ax[((len(telescope)+1)*ii)+(len(telescope)+1)].set_ylabel('Offsets (mas)',\
                                                                          fontsize=fontsize_label)
                ax[((len(telescope)+1)*ii)+(len(telescope)+1)].tick_params(labelsize=labelsize_ticks)
        #title
        ax[len(telescope)+1].set_title('Model Image',fontsize=fontsize_title)
        
        ### Synthetic observations
        obs_Jysqmas = obs_deconv/obs_pupil.pixelarea.value
        levels = np.nanmax(obs_Jysqmas)*np.array([0.05,0.1,0.3,0.5,0.7,0.9])
        cbar = ax[((len(telescope)+1)*ii)+(len(telescope)+jj+2)].imshow(obs_Jysqmas,origin='lower',
                                                                 interpolation='Nearest',extent=[-FOV,FOV,-FOV,FOV])
        cbar = fig.colorbar(cbar,ax=ax[((len(telescope)+1)*ii)+(len(telescope)+jj+2)],\
                    fraction=0.046, pad=0.04)
        cbar.set_label('Flux (mJy/sqmas)',fontsize=fontsize_label)
        cbar.ax.tick_params(labelsize=labelsize_cbar)
        
        res = np.str(np.round(206265*wavelength[ii]*1E-6/(pixelscale_pupil * pupil_image.shape[0])*1000.,1))
        ax[((len(telescope)+1)*ii)+(len(telescope)+jj+2)].text(-140,-140,res+' mas',\
                                                                color='white',fontsize=fontsize_label,weight='bold')
        
        if len(obs_Jysqmas) > 2:
            ax[((len(telescope)+1)*ii)+(len(telescope)+jj+2)].contour(obs_Jysqmas,levels=levels,\
                                                                  colors='white',alpha=0.5,\
                                                                 extent=[-FOV,FOV,-FOV,FOV])
        if ii < len(wavelength)-1:
                ax[((len(telescope)+1)*ii)+(len(telescope)+jj+2)].get_xaxis().set_visible(False)
                ax[((len(telescope)+1)*ii)+(len(telescope)+jj+2)].get_yaxis().set_visible(False)
        #title
        ax[len(telescope)+jj+2].set_title('Synthetic Observation',fontsize=fontsize_title) 
        
    #Bottom row labels        
    ax[((len(telescope)+1)*ii)+(len(telescope)+1)].set_ylabel('Offsets (mas)',fontsize=fontsize_label)
    ax[((len(telescope)+1)*ii)+(len(telescope)+1)].set_xlabel('Offsets (mas)',fontsize=fontsize_label)
    ax[((len(telescope)+1)*ii)+(len(telescope)+1)].tick_params(labelsize=labelsize_ticks)
    ax[((len(telescope)+1)*ii)+(len(telescope)+jj+2)].set_xlabel('Offsets (mas)',fontsize=fontsize_label)
    ax[((len(telescope)+1)*ii)+(len(telescope)+jj+2)].get_yaxis().set_visible(False)
    ax[((len(telescope)+1)*ii)+(len(telescope)+jj+2)].tick_params(labelsize=labelsize_ticks)
        
#(0,0) element is blanck
ax[0].axis('off')

fig.tight_layout()
fig.savefig(figname,dpi=250)
os.system('open '+figname)
print('-- DONE')
