# IMPORTS
import hypercat as hc
import interferometry as inter
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import ioops as io

#For more information about to select subcubes and make a source check the hypercat_workflow.ipynb notebook.

# Load a sub-cube into RAM
hdf5file = '/Volumes/Seagate Backup Plus Drive/hypercat/hypercat_20170714.hdf5'
cube = hc.ModelCube(hdffile=hdf5file,hypercube='imgdata',subcube_selection='../examples/Circinus.json')

#Make a source
circinus = hc.Source(cube,luminosity='6e43 erg/s',distance='4.2 Mpc',name='Circinus',pa='-37 deg')

#Select a clumpy torus image and wavelength. Assuming a flux density of 2500 mJy at 8.5 microns.
i = 61
q = 0.9
tauv = 54
wave = 12.
theta = (i,q,tauv,wave)
sky = circinus(theta)

#plt.imshow(sky.data.T,origin='lower',interpolation='Nearest')

#Create 2D FFT of clumpy torus image
ori_fft = inter.ima2fft(sky)
#Obtain pixel scale
fftscale = inter.fft_pxscale(sky)
#Obtain uv points
filename = '../examples/Circinus_clean.oifits'
u,v = inter.uvload(filename)
#obtain baseline and position angles in 1D
BL, Phi = inter.baseline_phase_1D(u,v)
#Obtain uv plane in pixels
u_px,v_px = inter.uvfreq2uvpixel(u,v,fftscale)
#Obtain correlated flux
corrflux, BL, Phi = inter.correlatedflux(ori_fft,u,v)
#obtain image fom fft
ori_ifft = inter.ima_ifft(ori_fft,u,v)

inter.plot_inter(sky,ori_fft,ori_ifft,u,v,fftscale,corrflux,BL,Phi)