#3rd party
from astropy.modeling import models
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
from copy import copy
import os as os

#HyperCAT
import sys
sys.path.append('../hypercat/')
import hypercat as hc
import interferometry as inter
import ioops as io
import ndiminterpolation


#For more information about to select subcubes and make a source check the hypercat_workflow.ipynb notebook.

# Load a sub-cube into RAM
hypercube = '/Volumes/Seagate Backup Plus Drive/hypercat/hypercat_20170827.hdf5'
subcube = '/Users/elopezro/Documents/Projects/NGC1068_SOFIA/Science/HyperCAT/subcube_ngc1068.json'
cube = hc.ModelCube(hdffile=hypercube,hypercube='imgdata',subcube_selection=subcube)


#NGC 1068 physical parameters
lum  = '1.65e45 erg/s'
dis  = '12.5 Mpc'
name = 'ngc1068'
pa   = '45 deg'

#Make a source
ngc1068 = hc.Source(cube,luminosity=lum,distance=dis,objectname=name,pa=pa)  

#Select a clumpy torus image and wavelength. Assuming a flux density of 10105 mJy at 10 microns.
#clumpy models
wave = 12.05

sig = 43.
Y   = 18.
N   = 4.
q   = 0.08
tv  = 70.
inc   = 75.
flux = '16.8 Jy'

vec = (sig,inc,Y,N,q,tv,wave)
sky = ngc1068(vec,total_flux_density=flux)

#Interferometric observations
DIR = '/Users/elopezro/Documents/GitHub/agn-imaging/data/oifits_Burtscher13/'
uvfilename = DIR+'NGC1068.oifits'
ori_fft,fftscale,u,v,corrflux,BL,Phi,ori_ifft = inter.interferometry(sky,uvfilename)


##gaussian
C = 16.

fwhm_g = 46.7 * (1./206265.) * (1./1000.) # mas * rad/arcsec
xx = np.linspace(0,2000,100)/wave

n1 = np.pi*fwhm_g*xx
n2 = 4.*np.log(2.)
F_g = C * np.exp(-(n1**2/n2))

fig, ax = plt.subplots(1,1,figsize=(7,5))

cbar = ax.scatter(BL,corrflux/1000.,c = Phi,s=50,linewidths=0)
plt.colorbar(cbar,ax=ax).set_label('PA ($^{\circ}$)',fontsize=20)
ax.set_ylabel('F$_{corr}$ [Jy]',fontsize=20)
ax.set_xlabel('Projected Baseline length [m]',fontsize=20)
ax.tick_params(labelsize=20)
ax.set_xlim([0,140])
ax.set_ylim([0,20])

ax.plot(xx,F_g,'-',linewidth=1)

fig.savefig('NGC1068_interferometry_HyperCAT.png')
os.system('open NGC1068_interferometry_HyperCAT.png')

