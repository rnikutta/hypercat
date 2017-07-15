from copy import copy
import numpy as N
from scipy import ndimage, integrate
import scipy.signal as signal

import ndiminterpolation as ndi
import pylab as p

def rot90ccw(v):

    """Rotate 2-d vector v 90 degrees CCW."""
    
    r = copy(v)
    r[0] = -v[1]
    r[1] =  v[0]

    return r
    
def whichside(a,b):

    sig = N.sign(N.dot(a,rot90ccw(b)))

    if sig > 0:
        print "b to the right of a"
    elif sig < 0:
        print "b to the left of a"
    else:
        print "b parallel/antiparallel to a"

    return sig


def gaussian2d_circular(npix=101,sigma=5.,x0=0,y0=0,theta=0.):
    x = N.arange(npix) - npix/2
    print x
    X, Y = N.meshgrid(x,x,indexing='ij')

    norm = 1. / (2.*N.pi*sigma**2.)
#    norm = 1.
    
    Z = norm * N.exp( -((X-x0)**2.+(Y-y0)**2.)/(2.*sigma**2.) )
#    Z = norm * N.exp(-((X-x0)**2./(2*float(sx)**2.) + (Y-y0)**2./(2*float(sy)**2.)))
    
    if theta != 0.:
        Z = ndimage.rotate(Z,float(theta),reshape=False)
        
    return Z


def gaussian(npix=101,sx=5.,sy=5.,x0=0,y0=0,theta=0.):
    x = N.arange(npix) - npix/2
    print x
    X, Y = N.meshgrid(x,x,indexing='ij')

    norm = 1. / (2.*N.pi*sx*sy)
#    norm = 1.
    
    Z = norm * N.exp( -( (X-x0)**2./(2.*float(sx)**2.) + (Y-y0)**2./(2.*float(sy)**2.)))
#    Z = norm * N.exp(-((X-x0)**2./(2*float(sx)**2.) + (Y-y0)**2./(2*float(sy)**2.)))
    
    if theta != 0.:
        Z = ndimage.rotate(Z,float(theta),reshape=False)
        
    return Z

def gaussian(npix=101,sx=5.,sy=5.,x0=0,y0=0,theta=0.):
    x = N.arange(npix) - npix/2
    X, Y = N.meshgrid(x,x,indexing='ij')

    norm = 1. / (2.*N.pi*sx*sy)
    Z = norm * N.exp( -( (X-x0)**2./(2.*float(sx)**2.) + (Y-y0)**2./(2.*float(sy)**2.)))
    
    if theta != 0.:
        Z = ndimage.rotate(Z,theta,reshape=False)
        
    return Z


def imageToEigenvectors(image):

    thresh = 0.001 # pixels as dim as this (in percent) of peak pixel brightness will be considered
#    y, x = N.argwhere(image>thresh*image.max()).T
    xind, yind = N.argwhere(image>thresh*image.max()).T
#    print "x.sum(), y.sum() = ", x.sum(), y.sum()
#    y, x = N.nonzero(image)
    x = xind - xind.mean()
    y = yind - yind.mean()
    coords = N.vstack((x,y))
    cov = N.cov(coords)
    evals, evecs = N.linalg.eig(cov)
    print "evals = ", evals
#    idmax = N.argmin(evals)
    idmax = N.argmax(evals)
    evec1 = evecs[idmax]
    print "evec1 = ", evec1

#    sig = whichside(evec1,N.array([0.,1.]))
    sig = whichside(N.array([0.,1.]),evec1)
    sig2 = whichside(evec1,N.array([0.,1.]))
#    sig = whichside(evec1,N.array([1.,0.]))
    print "sig = ", sig, sig2
    
#    if sig == -1.:
##        evec1[0] = -evec1[0]
#        evec1 = -evec1
    
    return evec1, xind, yind
    
#    sort_indices = N.argsort(evals)[::-1]
#    evec1, evec2 = evecs[:,sort_indices]
#
#    return evec1, evec2
    

def getUnitVector(axis=1,ndim=2):

    uvec = N.eye(ndim)[:,axis]

    return uvec


#def angle1(v1,v2):
def angle1(a,b):

    """Compute angle between two vectors a and b.

    Uses formula:

    .. math::

       \\cos\\theta = \\frac{a \\cdot b}{|a||b|}

    """

#    a = N.dot(v1,v2)
#    b = N.linalg.norm(v1)*N.linalg.norm(v2)
#
#    angle = N.degrees(N.arccos(a/b))
#    
#    print angle
#
##    if angle > 45.:
##        angle = 90 - abs(N.array(angle)-90)
###        angle = abs(N.array(angle)-90)
#
##    if angle > 135.:
##        angle = angle + 90
###        angle = 90 - abs(N.array(angle)-90)
#
#
##    delta = u1*v2 - u2*11
#    delta = v1[0]*v2[1] - v1[1]*v2[0]
#    print "delta = ", delta
#    
#        
#    return angle


    aux1 = N.dot(a,b)
    aux2 = N.linalg.norm(a)*N.linalg.norm(b)

    angle = N.degrees(N.arccos(aux1/aux2))
    
    print angle

#    if angle > 45.:
#        angle = 90 - abs(N.array(angle)-90)
##        angle = abs(N.array(angle)-90)

#    if angle > 135.:
#        angle = angle + 90
##        angle = 90 - abs(N.array(angle)-90)


#    delta = u1*v2 - u2*11
    delta = a[0]*b[1] - a[1]*b[0]
    print "delta = ", delta
        
    return angle


#def angle2(v1,v2):
#
#    """Compute angle between vectors v1 and v2.
#
#    Uses formula:
#
#    .. math::
#
#       \\tan \\theta = \\frac{|v1 \\times b2|}{a \\cdot b}
#
#    Slower than :func:`angle1`, but possibly more accurate for very
#    small angles.
#
#    """
#    
#    a = N.linalg.norm(N.cross(v1,v2))
#    b = N.dot(v1,v2)
#
#    angle = N.degrees(N.arctan2(a,b))
#
#    if angle > 45.:
#        angle = 90 - abs(N.array(angle)-90)
#    
#    return angle 


def angle2(a,b):

    """Compute angle between two vectors a and b.

    Uses formula:

    .. math::

       \\tan \\theta = \\frac{|a \\times b|}{a \\cdot b}

    Slower than :func:`angle1`, but possibly more accurate for very
    small angles.

    """
    
    aux1 = N.linalg.norm(N.cross(a,b))
    aux2 = N.dot(a,b)

    angle = N.degrees(N.arctan2(aux1,aux2))

    if angle > 45.:
        angle = 90 - abs(N.array(angle)-90)
    
    return angle 
    

def get_moment(image,radius=1.,angular='a',m=1):

    """Compute m-th moment of image.

    Parameters:
    -----------
    image : array (or instance of Image?)
       2d array (or instance of Image?)
    """
    
    npix = image.shape[0]
    x = N.linspace(-radius,radius,npix)
    theta = (x,x)
    ip = ndi.NdimInterpolation(image,theta,mode='lin')

    angulars = {'a':N.cos,'b':N.sin}
    ang = angulars[angular]


    def get_xy(r,phi):
        x_ = r*N.cos(phi)
        y_ = r*N.sin(phi)
        return (x_,y_)


    def getI(phi,r):
        x_, y_ = get_xy(r,phi)
        I = ip((x_,y_))
        res = I * ang(m*phi) * r**m
        return res

    
    res = integrate.dblquad(getI, 0., radius, lambda x:0., lambda x:2*N.pi,epsabs=1e-03, epsrel=1e-03)[0]
#    integrate.dblquad(func, -pi/2, pi/2, lambda x:-pi/2, lambda x:pi/2)[0]

#    return ip, theta
    return res


def get_power(image,m,r=1.):

    a0 = get_moment(image,radius=r,angular='a',m=0)

    if m == 0:
        P = (a0*N.log(r))**2
    elif m > 0:
        am = get_moment(image,radius=r,angular='a',m=m)
        bm = get_moment(image,radius=r,angular='b',m=m)
        P = (am**2+bm**2) / (2.*m**2*r**(2*m))

    return P


def get_wavelet(npix,a):
    print a
    x = N.arange(-npix/2,1+npix/2,1)
    print x
    X, Y = N.meshgrid(x,x)
    g = (2.-(X**2.+Y**2.)/a**2.)*N.exp(-(X**2.+Y**2.)/(2.*a**2.))
    return g


def get_wavelet_elliptical_mexh(npix,a,b):
    print a, b
    xa = N.arange(-npix/2+1,1+npix/2,1)
    xb = N.arange(-npix/2+1,1+npix/2,1)
    X,Y = N.meshgrid(xa,xb)
    g = (1./(2*N.pi*a**3*b**3)) * (a**2+b**2-(X**2/(a/b)**2)-(Y**2/(b/a)**2)) * N.exp(-0.5*((X**2/a**2)+(Y**2/b**2)))
    return g

#def get_wavelet_elliptical_mexh_vuong(npix,s=10.,sigma=1.,a=0,b=0):
#def get_wavelet_elliptical_mexh_vuong(npix,s=5.,sigma=2.):
def get_wavelet_elliptical_mexh_vuong(npix,sx=5.,sy=5.):
#    print a, b
    sigma = sx/float(sy)
    xa = N.arange(-npix/2+1,1+npix/2,1)
    xb = N.arange(-npix/2+1,1+npix/2,1)
    X,Y = N.meshgrid(xa,xb)

#    g = (1./(2*N.pi*a**3*b**3)) * (a**2+b**2-(X**2/(a/b)**2)-(Y**2/(b/a)**2)) * N.exp(-0.5*((X**2/a**2)+(Y**2/b**2)))
    K = sigma**2./(16*N.pi**3.)
#    g = (1./(s*N.sqrt(K)) * (2. - (X-a)**2./(s*sigma)**2 - (Y-a)**2./s**2) * N.exp(-0.5*((X-a)**2/(s*sigma)**2 - (Y-b)**2/s**2)))
#    g = (1./(s*N.sqrt(K)) * (2. - X**2./(s*sigma)**2. - Y**2./s**2.) * N.exp(-0.5*(X**2/(s*sigma)**2. + Y**2/s**2.)))
#
#    g = (1./(sx*N.sqrt(K)) * (2. - X**2./sx**2. - Y**2./sy**2.) * N.exp(-0.5*(X**2/sx**2. + Y**2/sy**2.)))
    g = (2. - X**2./sx**2. - Y**2./sy**2.) * N.exp(-0.5*(X**2/sx**2. + Y**2/sy**2.))
    return g

def get_wavelet_elliptical_mexh_vuong_fast(npix,sx=5.,sy=5.):
#    sigma = sx/float(sy)
    x_ = N.arange(-npix/2+1,1+npix/2,1)
    X, Y = N.meshgrid(x_,x_,indexing='ij')
    AUX = (X**2./sx**2. + Y**2./sy**2.)
#    K = sigma**2./(16*N.pi**3.)
#    g = (1./(sx*N.sqrt(K)) * (2. - X**2./sx**2. - Y**2./sy**2.) * N.exp(-0.5*(X**2/sx**2. + Y**2/sy**2.)))
    g = (2. - AUX) * N.exp(-0.5*AUX)
    return g


def get_wavelet_elliptical_mexh_gaillot(npix,sx=5.,sigma=1.):
    x_ = N.arange(-npix/2+1,1+npix/2,1)
    X, Y = N.meshgrid(x_,x_)
    AUX = (X**2. + (sigma*Y)**2.) / sx**2.
    g = (2. - AUX) * N.exp(-0.5*AUX)
    return g

def get_wavelet_elliptical_mexh_gaillot_full(npix,x0=0,y0=0,sx=5.,sigma=1.):
    x_ = N.arange(-npix/2+1,1+npix/2,1)
    X, Y = N.meshgrid(x_,x_)
    AUX = ((X-x0)**2. + sigma**2*(Y-y0)**2.) / sx**2.
    g = (2. - AUX) * N.exp(-0.5*AUX)
    return g


def plot():
    a_ = 17
    cmap=p.cm.viridis
    IMG = img
    npix=IMG.shape[0]
    wavelet = get_wavelet(IMG.shape[0],a_)
    conv = convolve_fft(IMG,wavelet/a_)
    ax1.cla()
    ax1.imshow(IMG.T,origin='lower',interpolation='none',cmap=cmap)
    ax1.contour(IMG.T,5,linestyles='-',colors='w')
    ax2.cla()
    ax2.imshow(wavelet.T,origin='lower',interpolation='none',cmap=cmap)
    ax3.cla()
    ax3.imshow(conv.T,origin='lower',interpolation='none',cmap=cmap)
    ax3.contour(conv.T,5,linestyles='-',colors='w')
    ax3.plot((npix/2-133/2,npix/2+133/2),(npix/2+133,npix/2+133),ls='-',lw=3,c='w')
    ax4.cla()
    ax4.semilogx(a,results,'b.-',ms=5)
    ax4.axvline(a_)


def work(a):
    return convolve_fft(img,get_wavelet(img.shape[0],a)/a).mean()/a

    
#
#
#
#    
#    X, Y = N.meshgrid(x,x)
#    R = N.sqrt(X**2+Y**2)
#
#
#    def func(x,y):
#
#        def getI(r,phi)
#        
#        return cos(x) + cos(y)
#
##    integrate.dblquad(func, -pi/2, pi/2, lambda x:-pi/2, lambda x:pi/2)[0]
#    integrate.dblquad(func, -pi/2, pi/2, lambda x:-pi/2, lambda x:pi/2)[0]

    
    
    

def ratio_fluxdensity_upper_over_lower():
    waves = N.linspace(2.2,18.5,40)
    angles = N.linspace(0,90,40)
    res = N.zeros((waves.size,angles.size))
    for iw,w in enumerate(waves):
        print w
        for ia,a in enumerate(angles):
            print a
            img = cube.get_image((a,5,0,w))
            ratio = img[:,idx+1:].sum()/img[:,:idx].sum()
            res[iw,ia] = ratio

            
def fluxdensity_i_wave(cube):
    waves = N.linspace(2.2,18.5,40)
    angles = N.linspace(0,90,40)
    res = N.zeros((waves.size,angles.size))
    for iw,w in enumerate(waves):
        print w
        for ia,a in enumerate(angles):
            print a
            img = cube.get_image((a,5,0,w))
            aux = img.sum()
            res[iw,ia] = aux

    return waves, angles, res


def getGaussian(npix,sig,xoff=0.,yoff=0.,Imax=1.):
    x = N.arange(npix)-npix/2
    Y,X = N.meshgrid(x,x)
    G = Imax * N.exp(-((X-xoff)**2. + (Y-yoff)**2.)/(2.*sig**2)) / (2.*N.pi*sig**2.)
    return G

def get_wavelet(npix,a):
    x = N.arange(-npix/2,1+npix/2,1)
    X,Y = N.meshgrid(x,x)
    g = (2.-(X**2+Y**2)/a**2)*N.exp(-(X**2+Y**2)/(2.*a**2))
    return g

def eq11(I,sig,a):
    return 2*(I/a) * (1.+(sig**2/a**2))**(-2.)


# y = eq11(1.,10,a)

def test1():
    G1 = getGaussian(101,10,-5, 5,1.)
    G2 = getGaussian(101,10, 5,-5,1.)
    G3 = G1+G2

    a = N.linspace(1,50)
    wmaxes = [signal.convolve2d(img,get_wavelet(npix,a_)/a_,mode='same').max()/a_ for a_ in a]

    fig = p.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(G3.T,origin='lower',interpolation='none',cmap=p.cm.inferno)
    ax2.plot(a,wmaxes,'b.-')
    ax2.set_xlim(min(a),max(a))

# TEST CASES WITH FIGURES FOR PAPER

def symmetric1(npix=51,savefile='symmertric1.npz'):

    x = N.arange(1.,npix+1)
    print x.size
#    x = N.arange(npix)

    print "Computing images"

#    G0  = gaussian2d_circular(npix,5.,0.,0.,0.) # symmetric Gaussian, nx=ny=10
    
    nx = ny = 8.
    G  = gaussian(npix,nx,ny,0.,0.,0.) # symmetric Gaussian, nx=ny=10
    wlG = get_wavelet_elliptical_mexh_vuong_fast(npix,nx,ny)

    
#    nx = ny = 7.
#    G1  = gaussian(npix,nx,ny,0.,0.,0.) # symmetric Gaussian, nx=ny=10
    
    nx, ny = 8., 5.
    Gh = gaussian(npix,nx,ny,0.,0.,0.)  # horizontally elongated, nx=10, ny=5
    wlGh = get_wavelet_elliptical_mexh_vuong_fast(npix,nx,ny)
    
    nx, ny = 8., 2. 
    Gv = gaussian(npix,nx,ny,0.,0.,0.)  # verticaly elongated, nx=5, ny=10
    wlGv = get_wavelet_elliptical_mexh_vuong_fast(npix,nx,ny)
    
    images = [G,Gh,Gv]
    wavelets = [wlG,wlGh,wlGv]
#    images = [G0,G,G1]
#    images = [G0]

    print "Computing convolutions"
    convomaxvals = N.zeros((x.size,len(images)))
    for jimg, img in enumerate(images):
        wl = wavelets[jimg]
        for jx, x_ in enumerate(x):
            if jx % 5 == 0: print x_
#        wl = get_wavelet(npix,x_)
            C = signal.convolve2d(img,wl/x_) #/ float(x_)
#            C = signal.convolve2d(img,wl/float(x_))
            convomaxvals[jx,jimg] = C.max() / x_
    
    if savefile is not None:
        print "Storing results in file '%s'" % savefile
        with open(savefile,'w') as f:
#            N.savez(savefile,G=G,Gh=Gh,Gv=Gv)
            N.savez(savefile,x=x,images=images,wavelets=wavelets,convomaxvals=convomaxvals)
        print "Done."

    
def plot_symmetric1(loadfile='symmetric1.npz'):

    data = N.load(loadfile)
    x = data['x']
    images = data['images']
    print "len(images) = ", len(images)
    wavelets = data['wavelets']
    convomaxvals = data['convomaxvals']

    print "images max vals = "
    print [img.max() for img in images]
    print [img.sum() for img in images]
    
    fig = p.figure(figsize=(6,6))
#    for j in xrange(0,3):
    for j in xrange(0,len(images)):
        ax = fig.add_subplot(3,3,j+1)
        ax.imshow(images[j].T,origin='lower')

    for j in xrange(0,len(images)):
        ax = fig.add_subplot(3,3,j+1+3)
        ax.imshow(wavelets[j].T,origin='lower')
        
#    for j in xrange(3,6):
    for j in xrange(0,len(images)):
        ax = fig.add_subplot(3,3,j+1+6)
#        C = convomaxvals[j-3]
        C = convomaxvals[:,j]
#        C /= N.array(x).astype('float')
        ax.semilogx(x,C,'b-')
        xmax = x[N.argmax(C)]
        ax.axvline(xmax)
        ax.set_title("xmax = %g" % xmax)

    fig.subplots_adjust(left=0.1,right=0.98,top=0.99,bottom=0.08,hspace=0.3,wspace=0.3)
    return fig
    
    
