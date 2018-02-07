from __future__ import print_function

__version__ = '20170202'   #yyymmdd
__author__ = 'Enrique Lopez-Rodriguez <enloro@gmail.com>'

"""Utilities for handling the interferometric mode of HyperCAT.
    
    .. automodule:: interferometry
"""

# IMPORTS

# 3rd party
import numpy as np
from astropy.modeling import models
import matplotlib.pyplot as plt
from copy import copy
import numpy.ma as ma

#HyperCAT
import ioops as io
import ndiminterpolation

# HELPER FUNCTIONS

def interferometry(sky,uvfilename):

    #Create 2D FFT of clumpy torus image
    ori_fft = ima2fft(sky)
    #Obtain pixel scale
    fftscale = fft_pxscale(sky)
    #Obtain uv points
    u,v = uvload(uvfilename)
    #Obtain correlated flux
    corrflux, BL, Phi = correlatedflux(ori_fft,u,v)
    #obtain image fom fft
    ori_ifft = ima_ifft(ori_fft,u,v)
    #Plots
    plot_inter(sky,ori_fft,ori_ifft,u,v,fftscale,corrflux,BL,Phi)

    return ori_fft,fftscale,u,v,corrflux,BL,Phi,ori_ifft



def uvload(filename):
    
    """Read uv points from a iofits file.
        
       Parameters
       ----------
       filename : str
           Name and direction of file to be load containgin the uv points
        
       Returns
       -------
       u, v: uv points
        
       Example
       -------
       .. code-block:: python
        
          filename = '/your/folder/file.oifits'
          u, v = uvload(filename)
        
    """
    
    ff = io.FitsFile(filename)
    
    #get uv points
    v = ff.getdata(4,'vcoord')
    u = ff.getdata(4,'ucoord')
    
    # create the center-symmetric points
    u_rev=-u
    v_rev=-v
    
    #combine the data set of uv points
    u = np.concatenate([u_rev,u])
    v = np.concatenate([v_rev,v])
    
    return u,v


def ima2fft(ima):
    
    """Compute 2-d FFT of an image.
        
       Parameters
       ----------
       ima : array
           2D clumpy model image
            
       abs : bool
           If True (default), only the module of the FFT is returned.
           If False, the complex values of the FFT is returned.
        
       Returns
       -------
       ima_fft : array
           The absolute value of the 2D FFT.
        
       Example
       -------
       .. code-block:: python
        
           ima_fft = ima2fft(ima)
        
    """
    
    #The 2D FFT is shifted to reconstruct the image at the central position of the array.
    ima = ima.data
    
    ima_fft = np.fft.fftshift(np.fft.fft2(ima))

    return ima_fft


def fft_pxscale(ima):
    
    """Compute conversion scale from sky space to FFT space.
        
       Parameters
       ----------
       ima : array
           2D clumpy model image.
        
       Returns
       -------
       fftscale : float
           The frequency scale in FFT space.
        
       Example
       -------
       .. code-block:: python
        
          fftscale = fft_pxscale(ima)
        
    """
    
    gridsize = ima.data.shape[0]
    #pixel scale of the image. This should be taken from the header of the clumpy torus image
    pxscale_mod = ima.pixelscale.value    #in mas
    #1D FFT of the gridsize.
    fft_freq=np.fft.fftfreq(gridsize,pxscale_mod)
    #wavelength of the clumpy torus image. This should be taken from the header of the clumpy torus image
    lam = ima.wave.value*1E-6                 #in m
    #lam = ima.wavelength
    #re-orginizing the 1D FFT to match with the grid.
#Py2    roll=np.floor(gridsize/2).astype("int")
    roll=np.floor(gridsize//2).astype("int")
    freq = np.fft.fftshift(fft_freq)
    ##
    ## pxscale -> fftscale
    fftscale=np.diff(freq)[0]           ## cycles / mas per pixel in FFT image
    mas2rad=np.deg2rad(1./3600000.)     ## mas per rad
    fftscale = fftscale/mas2rad * lam   ## meters baseline per px in FFT image at a given wavelength
    print("Pixel scale in FFT image is: ", fftscale, " m (Baseline) per pixel")
    
    return fftscale


def correlatedflux(ima_fft,u,v):
    
    """Compute 2D correlated flux, baseline and position angle map from a 2-d FFT map.
        
       Parameters
       ----------
       ima_fft : array
           2D FFT of the clumpy torus model from ima2fft
            
       u_px, v_px : array
           u and v planes in pixels
            
        Returns:
        --------
        corrflux : array
            Correlated flux in 2D uv plane
        
        BL : array
        Baseline estimated from uvplanes
        
        Phi : array
        Position angles estimated from uvplanes


        Example
        -------
        .. code-block:: python
        
           corrflux2D = correlatedflux2D(ori_fft,u_px,v_px,BL,Phi)
        
    """
    
    ima_fft = np.abs(ima_fft)
    
    x  = np.arange(ima_fft.shape[0])
    ip = ndiminterpolation.NdimInterpolation(ima_fft,[x,x])
    uu = u + ima_fft.shape[0]//2
    vv = v + ima_fft.shape[0]//2
    corrflux = ip(np.dstack((uu,vv)))

    BL = np.sqrt(u**2+v**2)
    Phi = np.rad2deg(np.arctan(u/v))

    return corrflux, BL, Phi



def ima_ifft(ima_fft,u,v):
    
    uu = u.astype(int) + ima_fft.shape[0]//2
    vv = v.astype(int) + ima_fft.shape[0]//2
    
    a = np.zeros((len(ima_fft),len(ima_fft)),dtype=np.complex_)
    for ii in range(len(uu)):
        a[uu[ii],vv[ii]] = ima_fft[uu[ii],vv[ii]]

    am = ma.array(a,mask=(a==0.))

    ima_ifft = np.abs(np.fft.ifft2(a))

    return ima_ifft


################ Plotting functions ###########

def plot_inter(ima,ima_fft,ima_ifft,u,v,fftscale,corrflux,BL,Phi):
    fig1, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(17,10))
    
    #Sky image
    ax1.imshow(ima.data.T,origin='lower',interpolation='Nearest')
    
    #FFT Image
    ax2.imshow(np.log10(np.abs(ima_fft)),origin='lower',interpolation='Nearest')

    #iFFT image
    ax3.imshow(ima_ifft,origin='lower',interpolation='Nearest')
    
    #uvplane and correlated flux
    cbar = ax4.scatter(u,v,c = corrflux,s=50,linewidths=0)
    plt.colorbar(cbar,label='F$_{corr}$ [Jy]',ax=ax4)
    ax4.tick_params(labelsize=20)
    ax4.set_xlim([np.max(u),np.min(u)])
    ax4.set_ylabel('v [m]',fontsize=20)
    ax4.set_xlabel('u [m]',fontsize=20)
    ax4.axvline(0,linestyle='--',color='black')
    ax4.axhline(0,linestyle='--',color='black')
    
    #uvplane and Phi
    cbar = ax5.scatter(u,v,c = Phi,s=50,linewidths=0)
    plt.colorbar(cbar,label='PA ($^{\circ}$)',ax=ax5)
    ax5.tick_params(labelsize=20)
    ax5.set_xlim([np.max(u),np.min(u)])
    ax5.set_ylabel('v [m]',fontsize=20)
    ax5.set_xlabel('u [m]',fontsize=20)
    ax5.axvline(0,linestyle='--',color='black')
    ax5.axhline(0,linestyle='--',color='black')
    
    #Correlated flux vs baseline
    cbar = ax6.scatter(BL,corrflux,c = Phi,s=20,linewidths=0)
    plt.colorbar(cbar,label='PA ($^{\circ}$)',ax=ax6)
    ax6.set_ylim([0,np.max(corrflux)])
    ax6.set_ylabel('F$_{corr}$ [Jy]',fontsize=20)
    ax6.set_xlabel('Baseline [m]',fontsize=20)

