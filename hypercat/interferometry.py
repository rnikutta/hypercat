__version__ = '20200913'   #yyymmdd
__author__ = 'Enrique Lopez-Rodriguez <enloro@gmail.com>'

"""Utilities for handling the interferometric mode of HyperCAT.

    .. automodule:: interferometry
"""

# IMPORTS
from itertools import product

# 3rd party
import numpy as np
from astropy.modeling import models
import matplotlib.pyplot as plt
from copy import copy
import numpy.ma as ma
from astropy import units as u

from matplotlib.ticker import MaxNLocator

#HyperCAT
#import ioops as io
#import ndiminterpolation

#import ioops as io
#import ndiminterpolation
#import imageops
#import units

from . import ioops as io
from . import ndiminterpolation
from . import imageops
from . import units

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


def get_uv(phi,blmax=115,n=100):
    BL = np.linspace(-blmax,blmax,n)

    tan = np.tan(np.radians(phi))
    t2 = np.sqrt(tan**2+1)

    u = BL*tan / t2
    v = BL / t2

    return u, v
    

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


#UNDER_DEVELOPMENTdef sky2cf_many(vecs):
#UNDER_DEVELOPMENT    import hypercat
#UNDER_DEVELOPMENT    import obsmodes
#UNDER_DEVELOPMENT    import pylab as plt
#UNDER_DEVELOPMENT    import matplotlib
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    # get cube and sky
#UNDER_DEVELOPMENT    cube = hypercat.ModelCube('/home/robert/data/hypercat/hypercat_20180417.hdf5', hypercube='imgdata', subcube_selection='onthefly')
#UNDER_DEVELOPMENT    ngc1068 = hypercat.Source(cube,luminosity='1.6e45 erg/s',distance='14.4 Mpc',pa='42 deg')
#UNDER_DEVELOPMENT#    ngc1068 = hypercat.Source(cube,luminosity='1.6e45 erg/s',distance='12.5 Mpc',pa='42 deg')
#UNDER_DEVELOPMENT#    ngc1068 = hypercat.Source(cube,luminosity='1.6e45 erg/s',distance='14.4 Mpc',pa='55 deg')
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    # obs data
#UNDER_DEVELOPMENT    u_, v_, cf_, cferr_, phi_, phierr_, amp_, amperr_, wave_ = uvload('../docs/notebooks/NGC1068.oifits')
#UNDER_DEVELOPMENT    bl = np.sqrt(u_**2 + v_**2)
#UNDER_DEVELOPMENT    wave_ *= 1e6
#UNDER_DEVELOPMENT#    sel = (wave_ > 11.5) & (wave_ < 12.5)
#UNDER_DEVELOPMENT    sel = (wave_ > 11.5) & (wave_ < 12.5)
#UNDER_DEVELOPMENT    cfs = cf_[:,sel].mean(axis=1)
#UNDER_DEVELOPMENT    print("cfs selected = ", cfs.size)
#UNDER_DEVELOPMENT    cferrs = cferr_[:,sel].mean(axis=1)
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    pa = np.degrees(np.arctan2(-u_,v_))
#UNDER_DEVELOPMENT    
#UNDER_DEVELOPMENT    # Find unresolved point-source flux at long baselines, following Lopez-Gonzaga+2016, Section 3.1
#UNDER_DEVELOPMENT    selps = (bl>70.)
#UNDER_DEVELOPMENT    cfsps = cfs[selps]
#UNDER_DEVELOPMENT    Fps = np.mean(cfsps)*u.Jy
#UNDER_DEVELOPMENT    Ftot = units.getQuantity('16 Jy',recognized_units=units.UNITS['FLUXDENSITY'])  #sky.getTotalFluxDensity()
#UNDER_DEVELOPMENT    S = (Ftot-Fps)/Ftot  # use this to scale modeled CFs
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    # instrument
#UNDER_DEVELOPMENT    vlti = obsmodes.Interferometry(uv='../docs/notebooks/NGC1068.oifits')
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    CFS = []
#UNDER_DEVELOPMENT    for vec in vecs:
#UNDER_DEVELOPMENT    
#UNDER_DEVELOPMENT        sky = ngc1068(vec,total_flux_density='16 Jy')
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT        CF,BL,FFTSCALE = vlti.observe(sky,fliplr=True) # use u,v points from oifitsfile
#UNDER_DEVELOPMENT        CF = S * CF*u.pix + Fps
#UNDER_DEVELOPMENT        CF = CF.value.astype('float64')
#UNDER_DEVELOPMENT        CFS.append(CF)
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    fig,(ax3,ax4) = plt.subplots(1,2,figsize=(7,3))
#UNDER_DEVELOPMENT    idx1 = np.argsort(BL)
#UNDER_DEVELOPMENT    ax3.errorbar(bl[idx1],cfs[idx1],yerr=cferrs[idx1],marker='o',ms=2,color='orange',ls='none')
#UNDER_DEVELOPMENT    for CF_ in CFS:
#UNDER_DEVELOPMENT        ax3.plot(BL[idx1],CF_[idx1],'b-',ms=2,lw=0.5,alpha=0.7)
#UNDER_DEVELOPMENT    ax3.set_xlim(0,130)
#UNDER_DEVELOPMENT    ax3.set_xlabel('BL (m)')
#UNDER_DEVELOPMENT    ax3.set_ylabel('corr. flux (Jy)')
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    idx2 = np.argsort(pa)
#UNDER_DEVELOPMENT    ax4.errorbar(pa[idx2],cfs[idx2],yerr=cferrs[idx2],marker='o',ms=2,color='orange',ls='none')
#UNDER_DEVELOPMENT    for CF_ in CFS:
#UNDER_DEVELOPMENT        ax4.plot(pa[idx2],CF_[idx2],'bo-',ms=2,lw=1)
#UNDER_DEVELOPMENT    ax4.set_xlabel('PA (deg)')
#UNDER_DEVELOPMENT    ax4.set_ylabel('corr. flux (Jy)')
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    fig.subplots_adjust(left=0.06,right=0.98,bottom=0.15)
#UNDER_DEVELOPMENT    
#UNDER_DEVELOPMENT    return vlti, fig
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT        
#UNDER_DEVELOPMENT        
#UNDER_DEVELOPMENT#    bl = bl[:len(bl)//2]
#UNDER_DEVELOPMENT    
#UNDER_DEVELOPMENT    # plotting
#UNDER_DEVELOPMENT    fig,(ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(13,3))
#UNDER_DEVELOPMENT    ex = sky.FOV.value/2
#UNDER_DEVELOPMENT    ax1.imshow(sky.data.value.T,origin='lower',extent=[-ex,ex,-ex,ex],cmap=matplotlib.cm.viridis)
#UNDER_DEVELOPMENT    ax1.axvline(0);ax1.axhline(0)
#UNDER_DEVELOPMENT    ax1.set_xlabel('mas')
#UNDER_DEVELOPMENT    ax1.set_ylabel('mas')
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    ex = (FFTSCALE*u.m * sky.npix).value/2
#UNDER_DEVELOPMENT    ax2.imshow(np.abs(vlti.imafft.value).T,origin='lower',extent=[ex,-ex,-ex,ex],norm=matplotlib.colors.LogNorm(),cmap=matplotlib.cm.jet)
#UNDER_DEVELOPMENT    ax2.axvline(0); ax2.axhline(0)
#UNDER_DEVELOPMENT    ax2.plot(vlti.u,vlti.v,marker='o',ls='none',color='k',ms=3)
#UNDER_DEVELOPMENT    ax2.set_xlim(130,-130)
#UNDER_DEVELOPMENT    ax2.set_ylim(-130,130)
#UNDER_DEVELOPMENT    ax2.set_xlabel('m')
#UNDER_DEVELOPMENT    ax2.set_ylabel('m')
#UNDER_DEVELOPMENT    ax2.set_title("sig, i, Y, N0, q, tv, wave = " + ", ".join(["%.4f"%_ for _ in vec]))
#UNDER_DEVELOPMENT#                  vec.__repr__())
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    idx1 = np.argsort(BL)
#UNDER_DEVELOPMENT#    print("idx1 ",idx1,len(idx1))
#UNDER_DEVELOPMENT#    print("bl ",bl,len(bl))
#UNDER_DEVELOPMENT#    print("bl.shape,cfs.shape,cferrs.shape = ", bl.shape,cfs.shape,cferrs.shape )
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    ax3.plot(BL[idx1],CF[idx1],'bo-',ms=2,lw=1)
#UNDER_DEVELOPMENT    ax3.errorbar(bl[idx1],cfs[idx1],yerr=cferrs[idx1],marker='o',ms=2,color='orange',ls='none')
#UNDER_DEVELOPMENT    ax3.set_xlim(0,130)
#UNDER_DEVELOPMENT    ax3.set_xlabel('BL (m)')
#UNDER_DEVELOPMENT    ax3.set_ylabel('corr. flux (Jy)')
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    print("bl", bl)
#UNDER_DEVELOPMENT    print("BL", BL)
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    
#UNDER_DEVELOPMENT    idx2 = np.argsort(pa)
#UNDER_DEVELOPMENT    ax4.plot(pa[idx2],CF[idx2],'bo-',ms=2,lw=1)
#UNDER_DEVELOPMENT#    ax4.errorbar(pas[idx2],cfs[idx2],yerr=cferrs[idx2],marker='o',ms=2,color='orange',ls='none')
#UNDER_DEVELOPMENT    ax4.errorbar(pa[idx2],cfs[idx2],yerr=cferrs[idx2],marker='o',ms=2,color='orange',ls='none')
#UNDER_DEVELOPMENT#    ax4.plot(pas,CF,'bo-',ms=2,lw=1)
#UNDER_DEVELOPMENT#    ax4.set_xlim(0,130)
#UNDER_DEVELOPMENT    ax4.set_xlabel('PA (deg)')
#UNDER_DEVELOPMENT    ax4.set_ylabel('corr. flux (Jy)')
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    fig.subplots_adjust(left=0.06,right=0.98,bottom=0.15)
#UNDER_DEVELOPMENT    
#UNDER_DEVELOPMENT    return vlti, fig
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENTdef sky2cf(vec=None,distance='14.4 Mpc',luminosity='1.6e45 erg/s',posangle='42 deg',cmap='CMRmap'):
#UNDER_DEVELOPMENT    import hypercat
#UNDER_DEVELOPMENT    import obsmodes
#UNDER_DEVELOPMENT    import pylab as plt
#UNDER_DEVELOPMENT    import matplotlib
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    # get cube and sky
#UNDER_DEVELOPMENT    cube = hypercat.ModelCube('/home/robert/data/hypercat/hypercat_20181031_all.hdf5', hypercube='imgdata', subcube_selection='onthefly')
#UNDER_DEVELOPMENT#    ngc1068 = hypercat.Source(cube,luminosity='1.6e45 erg/s',distance='14.4 Mpc',pa='42 deg')
#UNDER_DEVELOPMENT#    ngc1068 = hypercat.Source(cube,luminosity='1.6e45 erg/s',distance='12.5 Mpc',pa='42 deg')
#UNDER_DEVELOPMENT    ngc1068 = hypercat.Source(cube,luminosity=luminosity,distance=distance,pa=posangle)
#UNDER_DEVELOPMENT#    ngc1068 = hypercat.Source(cube,luminosity='1.6e45 erg/s',distance='14.4 Mpc',pa='55 deg')
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    # obs data
#UNDER_DEVELOPMENT    u_, v_, cf_, cferr_, phi_, phierr_, amp_, amperr_, wave_ = uvload('../docs/notebooks/NGC1068.oifits')
#UNDER_DEVELOPMENT    bl = np.sqrt(u_**2 + v_**2)
#UNDER_DEVELOPMENT    pa = np.degrees(np.arctan2(-u_,v_))
#UNDER_DEVELOPMENT    
#UNDER_DEVELOPMENT    wave_ *= 1e6
#UNDER_DEVELOPMENT    sel = (wave_ > 11.5) & (wave_ < 12.5)
#UNDER_DEVELOPMENT    cfs = cf_[:,sel].mean(axis=1)
#UNDER_DEVELOPMENT    print("cfs selected = ", cfs.size)
#UNDER_DEVELOPMENT    cferrs = cferr_[:,sel].mean(axis=1)
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    phis = phi_[:,sel].mean(axis=1)
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    
#UNDER_DEVELOPMENT#    # Find unresolved point-source flux at long baselines, following Lopez-Gonzaga+2016, Section 3.1
#UNDER_DEVELOPMENT#    selps = (bl>70.)
#UNDER_DEVELOPMENT#    cfsps = cfs[selps]
#UNDER_DEVELOPMENT#    Fps = np.mean(cfsps)*u.Jy
#UNDER_DEVELOPMENT#    Ftot = units.getQuantity('16 Jy',recognized_units=units.UNITS['FLUXDENSITY'])  #sky.getTotalFluxDensity()
#UNDER_DEVELOPMENT#    S = (Ftot-Fps)/Ftot  # use this to scale modeled CFs
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    
#UNDER_DEVELOPMENT#    cfs = cfs*t.vector('cfs')
#UNDER_DEVELOPMENT#    cferrs = cfs*t.vector('cferrs')
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    
#UNDER_DEVELOPMENT    # instrument
#UNDER_DEVELOPMENT#    x = np.linspace(-100,100,20)
#UNDER_DEVELOPMENT#    UV = list(product(x,x))
#UNDER_DEVELOPMENT#    U = np.array([UV[j][0] for j in range(len(UV))])
#UNDER_DEVELOPMENT#    V = np.array([UV[j][1] for j in range(len(UV))])
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    U = u_
#UNDER_DEVELOPMENT    V = v_
#UNDER_DEVELOPMENT#    return U,V
#UNDER_DEVELOPMENT    
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT#    U = np.random.uniform(-115,115,100)
#UNDER_DEVELOPMENT#    V = np.random.uniform(-115,115,100)
#UNDER_DEVELOPMENT##    U = np.arange(-115,115.1,10)
#UNDER_DEVELOPMENT##    V = np.arange(-115,115.1,10)
#UNDER_DEVELOPMENT#    aux = np.array(list(product(U,V)))
#UNDER_DEVELOPMENT#    U = aux[:,0]
#UNDER_DEVELOPMENT#    V = aux[:,1]
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    U, V = [], []
#UNDER_DEVELOPMENT    for phi in range(0,181,10):
#UNDER_DEVELOPMENT        U_, V_ = get_uv(phi,129,100)
#UNDER_DEVELOPMENT        U = U + U_.tolist()
#UNDER_DEVELOPMENT        V = V + V_.tolist()
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    U = np.array(U)
#UNDER_DEVELOPMENT    V = np.array(V)
#UNDER_DEVELOPMENT            
#UNDER_DEVELOPMENT    
#UNDER_DEVELOPMENT    PA = np.degrees(np.arctan2(-U,V))
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    
#UNDER_DEVELOPMENT    
#UNDER_DEVELOPMENT#    vlti = obsmodes.Interferometry(uv='../docs/notebooks/NGC1068.oifits')
#UNDER_DEVELOPMENT    vlti = obsmodes.Interferometry(uv=(U,V))
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    if vec is None:
#UNDER_DEVELOPMENT#    sig, i, Y, N0, q, tv, wave = 28.7,57.6,9.5,1.39,0.022,10.95,12.
#UNDER_DEVELOPMENT#        sig, i, Y, N0, q, tv, wave = 28.7,57.6,9.5,1.39,0.022,10.95,12.
#UNDER_DEVELOPMENT        sig, i, Y, N0, q, tv, wave = 28.7,57.6,10,1.39,0.022,10.95,12.
#UNDER_DEVELOPMENT        vec = (sig, i, Y, N0, q, tv, wave)
#UNDER_DEVELOPMENT        
#UNDER_DEVELOPMENT    sky = ngc1068(vec,total_flux_density='16 Jy')
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    CF,BL,FFTSCALE = vlti.observe(sky,fliplr=True) # use u,v points from oifitsfile
#UNDER_DEVELOPMENT#    CF = (CF*u.pix + 0.8*u.Jy).value.astype('float64')
#UNDER_DEVELOPMENT    
#UNDER_DEVELOPMENT#    CF = S * CF*u.pix + Fps
#UNDER_DEVELOPMENT#    CF = CF.value.astype('float64')
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT#    bl = bl[:len(bl)//2]
#UNDER_DEVELOPMENT    
#UNDER_DEVELOPMENT    ccube = hypercat.ModelCube('/home/robert/data/hypercat/hypercat_20181031_all.hdf5', hypercube='clddata', subcube_selection='onthefly')
#UNDER_DEVELOPMENT    cvec = np.array(vec)[np.array((0,1,2,4))]
#UNDER_DEVELOPMENT    cimg = ccube(cvec) * vec[3]
#UNDER_DEVELOPMENT    cimg = imageops.rotateImage(cimg,posangle)
#UNDER_DEVELOPMENT    
#UNDER_DEVELOPMENT    # plotting
#UNDER_DEVELOPMENT    fontsize = 10
#UNDER_DEVELOPMENT    plt.rcParams['font.size'] = fontsize
#UNDER_DEVELOPMENT    
#UNDER_DEVELOPMENT    fig,((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(10,6.3))
#UNDER_DEVELOPMENT    
#UNDER_DEVELOPMENT    ex = sky.FOV.value/2
#UNDER_DEVELOPMENT    ax1.imshow(sky.data.value.T,origin='lower',extent=[-ex,ex,-ex,ex],cmap=cmap,interpolation='bicubic')
#UNDER_DEVELOPMENT    ax1.xaxis.set_major_locator(MaxNLocator(7))
#UNDER_DEVELOPMENT    ax1.yaxis.set_major_locator(MaxNLocator(7))
#UNDER_DEVELOPMENT    ax1.contour(cimg.T,origin='lower',extent=[-ex,ex,-ex,ex],levels=(1,3,5),linewidths=0.2,colors='w',alpha=0.5)
#UNDER_DEVELOPMENT    ax1.axvline(0,lw=0.5);
#UNDER_DEVELOPMENT    ax1.axhline(0,lw=0.5)
#UNDER_DEVELOPMENT    ax1.set_xlabel('x-offset (mas)')
#UNDER_DEVELOPMENT    ax1.set_ylabel('y-offset (mas)')
#UNDER_DEVELOPMENT    xt, yt = 0.02, 0.03
#UNDER_DEVELOPMENT    ax1.text(xt,yt,'(a)',color='w',transform=ax1.transAxes)
#UNDER_DEVELOPMENT#    ax1.add_compass(loc=1,c='w')
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    
#UNDER_DEVELOPMENT    
#UNDER_DEVELOPMENT    exf = (FFTSCALE*u.m * sky.npix).value/2
#UNDER_DEVELOPMENT    ax2.imshow(np.abs(vlti.imafft.value).T,origin='lower',extent=[exf,-exf,-exf,exf],norm=matplotlib.colors.LogNorm(),cmap='jet')
#UNDER_DEVELOPMENT    ax2.xaxis.set_major_locator(MaxNLocator(6))
#UNDER_DEVELOPMENT    ax2.yaxis.set_major_locator(MaxNLocator(6))
#UNDER_DEVELOPMENT    ax2.axvline(0,lw=0.5);
#UNDER_DEVELOPMENT    ax2.axhline(0,lw=0.5)
#UNDER_DEVELOPMENT    ax2.plot(vlti.u,vlti.v,marker='o',ls='none',color='k',ms=3)
#UNDER_DEVELOPMENT    ax2.set_xlim(130,-130)
#UNDER_DEVELOPMENT    ax2.set_ylim(-130,130)
#UNDER_DEVELOPMENT    ax2.set_xlabel('U (m)')
#UNDER_DEVELOPMENT    ax2.set_ylabel('V (m)')
#UNDER_DEVELOPMENT    ax2.text(xt,yt,'(b)',color='k',transform=ax2.transAxes)
#UNDER_DEVELOPMENT#    ax2.set_title("sig, i, Y, N0, q, tv, wave = " + ", ".join(["%.4f"%_ for _ in vec]))
#UNDER_DEVELOPMENT#                  vec.__repr__())
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    limg = sky.data.value
#UNDER_DEVELOPMENT    levels = np.array((0.1,0.5,0.9))*np.max(limg)
#UNDER_DEVELOPMENT    ax3.imshow(cimg.T,origin='lower',extent=[-ex,ex,-ex,ex],cmap='gray_r',interpolation='bicubic')
#UNDER_DEVELOPMENT    ax3.xaxis.set_major_locator(MaxNLocator(7))
#UNDER_DEVELOPMENT    ax3.yaxis.set_major_locator(MaxNLocator(7))
#UNDER_DEVELOPMENT    ax3.contour(limg.T,origin='lower',extent=[-ex,ex,-ex,ex],levels=levels,linewidths=0.2,colors='w',alpha=0.5)
#UNDER_DEVELOPMENT    ax3.axvline(0,lw=0.5);
#UNDER_DEVELOPMENT    ax3.axhline(0,lw=0.5)
#UNDER_DEVELOPMENT    ax3.set_xlabel('x-offset (mas)')
#UNDER_DEVELOPMENT    ax3.set_ylabel('y-offset (mas)')
#UNDER_DEVELOPMENT    xt, yt = 0.02, 0.03
#UNDER_DEVELOPMENT    ax3.text(xt,yt,'(c)',color='k',transform=ax3.transAxes)
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    idxBL = np.argsort(BL)
#UNDER_DEVELOPMENT#    print("idx1 ",idx1,len(idx1))
#UNDER_DEVELOPMENT#    print("bl ",bl,len(bl))
#UNDER_DEVELOPMENT#    print("bl.shape,cfs.shape,cferrs.shape = ", bl.shape,cfs.shape,cferrs.shape )
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    idxbl = np.argsort(bl)
#UNDER_DEVELOPMENT#    ax4.errorbar(bl[idxbl],cfs[idxbl],yerr=cferrs[idxbl],marker='o',ms=3,zorder=1,color='orange',ls='none',label='data')
#UNDER_DEVELOPMENT    im = ax4.scatter(bl[idxbl],cfs[idxbl],c=pa[idxbl],marker='o',s=10,zorder=1,cmap='rainbow',label='data')
#UNDER_DEVELOPMENT    cb = plt.colorbar(im,ax=ax4)
#UNDER_DEVELOPMENT#    ax4.errorbar(bl[idxbl],cfs[idxbl],yerr=cferrs[idxbl],marker=None,mew=0,zorder=1,color='orange',ls='none')
#UNDER_DEVELOPMENT    idxBL = np.argsort(BL)
#UNDER_DEVELOPMENT    ax4.plot(BL[idxBL],CF[idxBL],marker='o',ms=1,color='b',ls='none',zorder=2,lw=1,alpha=0.8,label='model')
#UNDER_DEVELOPMENT    ax4.xaxis.set_major_locator(MaxNLocator(6))
#UNDER_DEVELOPMENT    ax4.yaxis.set_major_locator(MaxNLocator(7))
#UNDER_DEVELOPMENT    ax4.set_xlim(0,130)
#UNDER_DEVELOPMENT    ax4.xaxis.set_ticks((0,30,60,90,120),minor=False)
#UNDER_DEVELOPMENT    ax4.xaxis.set_ticks((0,10,20,30,40,50,60,70,80,90,100,110,120,130),minor=True)
#UNDER_DEVELOPMENT    ax4.set_ylim(0,15)
#UNDER_DEVELOPMENT    ax4.yaxis.set_ticks((0,5,10,15),minor=False)
#UNDER_DEVELOPMENT    ax4.yaxis.set_ticks((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),minor=True)
#UNDER_DEVELOPMENT    ax4.set_xlabel('BL (m)')
#UNDER_DEVELOPMENT    ax4.set_ylabel('corr. flux (Jy)')
#UNDER_DEVELOPMENT    ax4.legend(loc='upper right',fontsize=8,frameon=False,title='@12 micron')
#UNDER_DEVELOPMENT    ax4.text(xt,yt,'(d)',color='k',transform=ax4.transAxes)
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    print("bl", bl)
#UNDER_DEVELOPMENT    print("BL", BL)
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    idxPA = np.argsort(PA)
#UNDER_DEVELOPMENT#    sel = (BL > 85) & (BL < 90)
#UNDER_DEVELOPMENT#    ax4.plot(PA[idxPA][sel],CF[idxPA][sel],'b-',ms=2,lw=1)
#UNDER_DEVELOPMENT    ax5.plot(PA[idxPA],CF[idxPA],'b-',lw=0.1,ms=3,zorder=2,alpha=0.2)
#UNDER_DEVELOPMENT#    ax4.errorbar(pas[idx2],cfs[idx2],yerr=cferrs[idx2],marker='o',ms=2,color='orange',ls='none')
#UNDER_DEVELOPMENT    idxpa = np.argsort(pa)
#UNDER_DEVELOPMENT    ax5.errorbar(pa[idxpa],cfs[idxpa],yerr=cferrs[idxpa],marker='o',ms=3,zorder=1,color='orange',ls='none')
#UNDER_DEVELOPMENT#    ax4.plot(pas,CF,'bo-',ms=2,lw=1)
#UNDER_DEVELOPMENT#    ax4.set_xlim(0,130)
#UNDER_DEVELOPMENT    ax5.set_xlabel('PA (deg)')
#UNDER_DEVELOPMENT    ax5.set_ylabel('corr. flux (Jy)')
#UNDER_DEVELOPMENT    ax5.set_xlim(-120,180)
#UNDER_DEVELOPMENT    ax5.xaxis.set_ticks((-90,0,90,180),minor=False)
#UNDER_DEVELOPMENT    ax5.xaxis.set_ticks((-120,-90,-60,-30,0,30,60,90,120,150,180),minor=True)
#UNDER_DEVELOPMENT    ax5.set_ylim(0,15)
#UNDER_DEVELOPMENT    ax5.yaxis.set_ticks((0,5,10,15),minor=False)
#UNDER_DEVELOPMENT    ax5.yaxis.set_ticks((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),minor=True)
#UNDER_DEVELOPMENT    ax5.text(xt,yt,'(e)',color='k',transform=ax5.transAxes)
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT#    vec2 = list(res.x[:2]) + [int(np.round(res.x[2]))] + list(res.x[3:6]) + [tuple(wave_[sel].tolist())]
#UNDER_DEVELOPMENT    mirwaves = np.linspace(8,13,20)
#UNDER_DEVELOPMENT    vec2 = vec[:-1] + [tuple(mirwaves)]
#UNDER_DEVELOPMENT    aux = cube(vec2)
#UNDER_DEVELOPMENT    sed = [aux[j,...].sum() for j in range(mirwaves.size)]
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    ax6.plot(mirwaves,sed,'b-',ms=1,label='model SED',zorder=2)
#UNDER_DEVELOPMENT    sel = (wave_>8) & (wave_<13)
#UNDER_DEVELOPMENT    for j in range(cf_.shape[0]):
#UNDER_DEVELOPMENT        if j == 0:
#UNDER_DEVELOPMENT            label = 'observations (per uv point)'
#UNDER_DEVELOPMENT        else:
#UNDER_DEVELOPMENT            label = ''
#UNDER_DEVELOPMENT            
#UNDER_DEVELOPMENT        ax6.plot(wave_[sel],cf_[j,sel],'0.2',lw=0.5,alpha=0.4,label=label)
#UNDER_DEVELOPMENT    ax6.legend(loc='upper left',fontsize=8,frameon=False)
#UNDER_DEVELOPMENT    ax6.set_xlabel('wavelength (micron)')
#UNDER_DEVELOPMENT    ax6.set_ylabel('corr. flux (Jy)')
#UNDER_DEVELOPMENT    ax6.set_xlim(8,13)
#UNDER_DEVELOPMENT    ax6.xaxis.set_ticks((8,9,10,11,12,13),minor=False)
#UNDER_DEVELOPMENT    ax6.xaxis.set_ticks(np.linspace(8,13,5*5+1),minor=True)
#UNDER_DEVELOPMENT    ax6.set_ylim(0,15)
#UNDER_DEVELOPMENT    ax6.yaxis.set_ticks((0,5,10,15),minor=False)
#UNDER_DEVELOPMENT    ax6.yaxis.set_ticks((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),minor=True)
#UNDER_DEVELOPMENT    ax6.text(xt,yt,'(f)',color='k',transform=ax6.transAxes)
#UNDER_DEVELOPMENT    
#UNDER_DEVELOPMENT    
#UNDER_DEVELOPMENT    title = "sig, i, Y, N0, q, tv, wave = " + ", ".join(["%.4f"%_ for _ in vec]) + "; D = %s, L = %s, PA = %s" % (distance,luminosity,posangle)
#UNDER_DEVELOPMENT    print(title)
#UNDER_DEVELOPMENT#    fig.suptitle(title)
#UNDER_DEVELOPMENT
#UNDER_DEVELOPMENT    fig.subplots_adjust(left=0.07,right=0.99,top=0.99,bottom=0.08,wspace=0.3,hspace=0.25)
#UNDER_DEVELOPMENT    plt.savefig('ngc1068_vlti_12mic_test1.pdf')
#UNDER_DEVELOPMENT    
#UNDER_DEVELOPMENT    return vlti, fig, sky.data.value, cimg



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
    ip = ndiminterpolation.NdimInterpolation(imafft.value,[x,x],mode='log')
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
