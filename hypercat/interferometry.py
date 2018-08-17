from __future__ import print_function

__version__ = '20170813'   #yyymmdd
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
from astropy import units as u

#HyperCAT
#import ioops as io
#import ndiminterpolation

import ioops as io
import ndiminterpolation
import imageops
import units

# HELPER FUNCTIONS

def load_uv(oifilename,hdu=4):

    ff = io.FitsFile(oifilename)

    # get uv points
    v = ff.getdata(hdu,'vcoord')
    u = ff.getdata(hdu,'ucoord')

#    # create the center-symmetric points
#    u_rev = -u
#    v_rev = -v
#
#    #combine the data set of uv points
#    u = np.concatenate([u_rev,u])
#    v = np.concatenate([v_rev,v])

    return u, v


def get_BLPhi(u,v):
    
    BL = np.sqrt(u**2 + v**2)
    Phi = -np.rad2deg(np.arctan2(u,v)) # minus for physical/astronomical way of measuring angles (positive in anti-clockwise direction)

    return BL, Phi







def uvload(filename,hdu=4):

    """Read uv points from a iofits file.

       Parameters
       ----------
       filename : str
           Name and direction of file to be load containgin the uv points

       Returns
       -------
       u, v: uv points
       cf, cferr: correlated flux and errors
       pa, paerr: PA and errors
       amp, amperr: visibility amplitude and errors

       Example
       -------
       .. code-block:: python

          filename = '/your/folder/file.oifits'
          u, v = uvload(filename)

    """

    ff = io.FitsFile(filename)

    ###get uv points
    v = ff.getdata(hdu,'vcoord')
    u = ff.getdata(hdu,'ucoord')

#TMP    # create the center-symmetric points
#TMP    u_rev = -u
#TMP    v_rev = -v
#TMP
#TMP    #combine the data set of uv points
#TMP    u = np.concatenate([u_rev,u])
#TMP    v = np.concatenate([v_rev,v])

    ##get correlated flux from observations
    cf = ff.getdata(hdu,'CFLUX')
    cferr = ff.getdata(hdu,'CFLUXERR')

    ## get PA from observations
    pa = ff.getdata(4,'VISPHI')
    paerr = ff.getdata(hdu,'VISPHIERR')

    ## get PA from observations
    amp = ff.getdata(hdu,'VISAMP')
    amperr = ff.getdata(hdu,'VISAMPERR')

    ##get wavelength
    wave = ff.getdata(3,'EFF_WAVE')

    return u, v, cf, cferr, pa, paerr, amp, amperr, wave


def ima2fft(ima,fliplr=True):

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

    # The 2D FFT is shifted to reconstruct the image at the central position of the array.


    
#    if ima.__class__ is imageops.Image:
#        ima = ima.data.T
#    elif .__class__ is numpy.ndarray:
#        ima = ima
#    else:
#        raise Exception("The image provided to ima2fft must be either
#        a 2-d array or an instance of the `Image` class")

    # do 2-d FFT
    units = ima.data.unit
    imafft = np.fft.fftshift(np.fft.fft2(ima.data))

    if fliplr is True:
        imafft = np.fliplr(imafft)

    return imafft*units


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
    roll=np.floor(gridsize//2).astype("int")
    freq = np.fft.fftshift(fft_freq)
    ##
    ## pxscale -> fftscale
    fftscale=np.diff(freq)[0]           ## cycles / mas per pixel in FFT image
    mas2rad=np.deg2rad(1./3600000.)     ## mas per rad
    fftscale = fftscale/mas2rad * lam   ## meters baseline per px in FFT image at a given wavelength
    print("Pixel scale in FFT image is: ", fftscale, " m (Baseline) per pixel")

    return fftscale



#def gauss2cf():
#
#    import imageops
#    import obsmodes
#    import pylab as plt
#    import matplotlib
#    from astropy.modeling import models
#    
#    npix = 241
#    x = np.linspace(-npix//2,npix//2,npix)
#    X,Y = np.meshgrid(x,x,indexing='xy')
#    G = models.Gaussian2D
#    g = G.evaluate(X,Y,1,0,0,15,15,0)
#    g = 16*u.Jay * (g/g.max())
#    gsky = imageops.Image(g,pixelscale='1 mas',total_flux_density='1.5 Jy')
#    gsky.wave = wave
#    cf,bl,fftscale = vlti.observe(gsky,oifilename='../docs/notebooks/NGC1068.oifits')
#    u,v = vlti.u, vlti.v
##    u = np.concatenate((u,[-1,1]))
##    v = np.concatenate((v,[1,-1]))
#    cf,bl,fftscale = vlti.observe(gsky,uv=(u,v))
#    FOVm = gsky.npix * fftscale
#    ex = FOVm/2
#    
#    ax1.imshow(gsky.data.value.T,origin='lower',extent=[-ex,ex,-ex,ex])
#    
#    ax2.imshow(np.abs(vlti.imafft.value),extent=[-ex,ex,-ex,ex],origin='lower',norm=matplotlib.colors.LogNorm())
#    ax2.plot(vlti.u,vlti.v,marker='o',ls='none',color='orange',ms=2)
#    
#    idx1 = np.argsort(bl)
#    ax3.plot(bl[idx1],cf[idx1],'b-')


#def sky2cf():
#    import hypercat
#    import obsmodes
#    import pylab as plt
#    import matplotlib
#
#    # get cube and sky
#    cube = hypercat.ModelCube('/home/robert/data/hypercat/hypercat_20180417.hdf5', hypercube='imgdata', subcube_selection='onthefly')
#    ngc1068 = hypercat.Source(cube,luminosity='1.6e45 erg/s',distance='14.4 Mpc',pa='42 deg',objectname='ngc1068')                          
##    ngc1068 = hypercat.Source(cube,luminosity='3e44 erg/s',distance='14.4 Mpc',pa='42 deg',objectname='ngc1068')                          
##    vec = (43,75,18,4,0.08,70,12)
#
##    sig, i, Y, N0, q, tv, wave = 30,80,8,4,0.08,70,12
#    sig, i, Y, N0, q, tv, wave = 28.7,57.6,9.5,1.39,0.022,10.95,12.
#    vec = (sig, i, Y, N0, q, tv, wave)
#    sky = ngc1068(vec,total_flux_density='16 Jy')
#
#    # get cf from oifiltsfile
#    u_, v_, cf_, cferr_, pa_, paerr_, amp_, amperr_, wave_ = uvload('../docs/notebooks/NGC1068.oifits')
#    wave_ *= 1e6
#    sel = (wave_ > 11.5) & (wave_ < 12.5)
#    cfs = cf_[:,sel].mean(axis=1)
#    cferrs = cferr_[:,sel].mean(axis=1)
#    
#    # siumlate interferometric observations
#    vlti = obsmodes.Interferometry()
#    cf,bl,fftscale = vlti.observe(sky,uv='../docs/notebooks/NGC1068.oifits') # use u,v points from oifitsfile
#    print("XXXXXXXXXXXXXXXXXX cf = ", cf)
##    bl = bl[:len(bl)//2]
#    
#    # plotting
#    fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,3))
#    ex = sky.FOV.value/2
#    ax1.imshow(sky.data.value.T,origin='lower',extent=[-ex,ex,-ex,ex],cmap=matplotlib.cm.viridis)
#    ax1.axvline(0);ax1.axhline(0)
#    ax1.set_xlabel('mas')
#    ax1.set_ylabel('mas')
#
#    ex = (fftscale*u.m * sky.npix).value/2
#    ax2.imshow(np.abs(vlti.imafft.value).T,origin='lower',extent=[ex,-ex,-ex,ex],norm=matplotlib.colors.LogNorm(),cmap=matplotlib.cm.jet)
#    ax2.axvline(0); ax2.axhline(0)
#    ax2.plot(vlti.u,vlti.v,marker='o',ls='none',color='k',ms=3)
#    ax2.set_xlim(130,-130)
#    ax2.set_ylim(-130,130)
#    ax2.set_xlabel('m')
#    ax2.set_ylabel('m')
#    ax2.set_title("sig, i, Y, N0, q, tv, wave = " + vec.__repr__())
#
#    idx1 = np.argsort(bl)
##    print("idx1 ",idx1,len(idx1))
##    print("bl ",bl,len(bl))
##    print("bl.shape,cfs.shape,cferrs.shape = ", bl.shape,cfs.shape,cferrs.shape )
#
#    # Find unresolved point-source flux at long baselines, following Lopez-Gonzaga+2016, Section 3.1
#    selps = (bl>70.)
#    cfsps = cfs[selps]
#
#    Fps = np.mean(cfsps)*u.Jy
#    print("Fps = ", Fps)
#    Ftot = sky.getTotalFluxDensity()
#
#    cf = cf * u.pix
#    
##    cf = (Ftot-Fps) * (cf/cf.max()) + Fps
#
#    print("=============== Ftot,Fps,cf = ",Ftot,Fps,cf)
#    cf = (Ftot-Fps) * (cf/Ftot) + Fps
#
#    ax3.plot(bl[idx1],cf[idx1],'bo-',ms=2,lw=1)
#    ax3.errorbar(bl[idx1],cfs[idx1],yerr=cferrs[idx1],marker='o',ms=2,color='orange',ls='none')
#    ax3.set_xlim(0,130)
#    ax3.set_xlabel('BL (m)')
#    ax3.set_ylabel('corr. flux (Jy)')
#    
#    return vlti, fig


def sky2cf(vec=None):
    import hypercat
    import obsmodes
    import pylab as plt
    import matplotlib

    # get cube and sky
    cube = hypercat.ModelCube('/home/robert/data/hypercat/hypercat_20180417.hdf5', hypercube='imgdata', subcube_selection='onthefly')
    ngc1068 = hypercat.Source(cube,luminosity='1.6e45 erg/s',distance='14.4 Mpc',pa='42 deg')

    # obs data
    u_, v_, cf_, cferr_, pa_, paerr_, amp_, amperr_, wave_ = uvload('../docs/notebooks/NGC1068.oifits')
    bl = np.sqrt(u_**2 + v_**2)
    wave_ *= 1e6
    sel = (wave_ > 11.5) & (wave_ < 12.5)
    cfs = cf_[:,sel].mean(axis=1)
    cferrs = cferr_[:,sel].mean(axis=1)
    
    # Find unresolved point-source flux at long baselines, following Lopez-Gonzaga+2016, Section 3.1
    selps = (bl>70.)
    cfsps = cfs[selps]
    Fps = np.mean(cfsps)*u.Jy
    Ftot = units.getQuantity('16 Jy',recognized_units=units.UNITS['FLUXDENSITY'])  #sky.getTotalFluxDensity()
    S = (Ftot-Fps)/Ftot  # use this to scale modeled CFs

    
#    cfs = cfs*t.vector('cfs')
#    cferrs = cfs*t.vector('cferrs')

    
    # instrument
    vlti = obsmodes.Interferometry(uv='../docs/notebooks/NGC1068.oifits')


    if vec is None:
#    sig, i, Y, N0, q, tv, wave = 28.7,57.6,9.5,1.39,0.022,10.95,12.
        sig, i, Y, N0, q, tv, wave = 28.7,57.6,9.5,1.39,0.022,10.95,12.
        vec = (sig, i, Y, N0, q, tv, wave)
        
    sky = ngc1068(vec,total_flux_density='16 Jy')

    CF,BL,FFTSCALE = vlti.observe(sky,fliplr=True) # use u,v points from oifitsfile
    CF = S * CF*u.pix + Fps
    CF = CF.value.astype('float64')

#    bl = bl[:len(bl)//2]
    
    # plotting
    fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,3))
    ex = sky.FOV.value/2
    ax1.imshow(sky.data.value.T,origin='lower',extent=[-ex,ex,-ex,ex],cmap=matplotlib.cm.viridis)
    ax1.axvline(0);ax1.axhline(0)
    ax1.set_xlabel('mas')
    ax1.set_ylabel('mas')

    ex = (FFTSCALE*u.m * sky.npix).value/2
    ax2.imshow(np.abs(vlti.imafft.value).T,origin='lower',extent=[ex,-ex,-ex,ex],norm=matplotlib.colors.LogNorm(),cmap=matplotlib.cm.jet)
    ax2.axvline(0); ax2.axhline(0)
    ax2.plot(vlti.u,vlti.v,marker='o',ls='none',color='k',ms=3)
    ax2.set_xlim(130,-130)
    ax2.set_ylim(-130,130)
    ax2.set_xlabel('m')
    ax2.set_ylabel('m')
    ax2.set_title("sig, i, Y, N0, q, tv, wave = " + ", ".join(["%.4f"%_ for _ in vec]))
#                  vec.__repr__())

    idx1 = np.argsort(BL)
#    print("idx1 ",idx1,len(idx1))
#    print("bl ",bl,len(bl))
#    print("bl.shape,cfs.shape,cferrs.shape = ", bl.shape,cfs.shape,cferrs.shape )

    ax3.plot(BL[idx1],CF[idx1],'bo-',ms=2,lw=1)
    ax3.errorbar(bl[idx1],cfs[idx1],yerr=cferrs[idx1],marker='o',ms=2,color='orange',ls='none')
    ax3.set_xlim(0,130)
    ax3.set_xlabel('BL (m)')
    ax3.set_ylabel('corr. flux (Jy)')

    print("bl", bl)
    print("BL", BL)
    
    return vlti, fig



def fft_pixelscale(image):

    fftscale = image.npix * image.pixelscale # mas 
    fftscale = image.wave.to('m') / fftscale.to('rad').value

    return fftscale


def correlatedflux(imafft,fftscale,u,v):

    """Compute 2D correlated flux, baseline and position angle map from a 2-d FFT map.

       Parameters
       ----------
       imafft : array
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

    imafft = np.abs(imafft)

    npix = imafft.shape[0]
    size = npix * fftscale # total image plae size in meters
    x = np.linspace(-size//2,size//2,npix)
    unit = imafft.unit
    ip = ndiminterpolation.NdimInterpolation(imafft.value,[x,x])
    corrflux = ip(np.dstack((u,v)))

    return corrflux*unit



def ima_ifft(ima_fft,u,v):

    uu = u.astype(int) + ima_fft.shape[0]//2
    vv = v.astype(int) + ima_fft.shape[0]//2

    a = np.zeros((len(ima_fft),len(ima_fft)),dtype=np.complex_)
    for ii in range(len(uu)):
        a[uu[ii],vv[ii]] = ima_fft[uu[ii],vv[ii]]

    am = ma.array(a,mask=(a==0.))

    ima_ifft = np.abs(np.fft.ifft2(a))

    return ima_ifft

def getObsPerWave(u,v,mag,magerr,wavearray,wave):

    m = np.zeros(len(u))
    merr = np.zeros(len(v))
    for ii in range(len(u)):
        m[ii] = mag[ii][np.where(wavearray <= wave*1E-6)[0][0]]
        merr[ii] = magerr[ii][np.where(wavearray <= wave*1E-6)[0][0]]
    return m, merr



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
