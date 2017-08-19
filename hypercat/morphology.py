from copy import copy
import numpy as N
from scipy import ndimage, integrate
import scipy.signal as signal

import ndiminterpolation as ndi
import pylab as p

import math

from astropy.modeling.functional_models import Gaussian2D


def rotateVector(vec,deg=90.):

    """Rotate 2-d vector by deg degrees.

    Parameters
    ----------
    vec : array
        2d-d vector.

    deg : float
        Rotation angle in degrees. If positive, rotates
        counter-clockwise, otherwise counter-clockwise.

    Returns
    -------
    rvec : arrau\y
        Rotated vector.

    Examples
    --------

    .. code-block:: python

       vec = N.array([0.,1.])  # unit vector along postive y-axis
       rotateVector(vec,deg=90.)
         array([ -1., 0.])

       rotateVector(vec,deg=-90.)
         array([ 1., 0.])

       rotateVector(vec,deg=45.)
         array([-0.70710678,  0.70710678])

    """

    angle = N.radians(deg)
    s = math.sin(angle)
    c = math.cos(angle)
    matrix = N.array([[ c, s],
                      [-s, c]])

    # transposing matrix to make positive angles rotate
    # counter-clockwise, i.e. as is convention in mathematics
    rvec = N.dot(matrix.T,vec)

    return rvec
    
    
def rot90ccw(v):

    """Rotate 2-d vector v 90 degrees CCW."""
    
    r = copy(v)
    r[0] = -v[1]
    r[1] =  v[0]

    return r


def whichside(a,b,verbose=False):

    """Compute relation sense of vectors a relative to vector b.

    When computing and angle between two vectors, it's not always
    clear in which direction of vector lies of another (left or right,
    i.e. counter- or clock-wise). This function answers the question.

    Parameters
    ----------

    a, b : array

        Two 2-d vectors.

    Returns
    -------
    sig : float
        Number -1., 0., or +1.
        If 0., vectors a and b are parallel or antiparallel to each other. 
        If -1., b lies to the left of a (b counter-clockwise from a).
        If +1., b lies to the right of a (b clockwise from a).

    Examples
    --------

    """

#    sig = N.sign(N.dot(a,rot90ccw(b)))
    sig = N.sign(N.dot(rot90ccw(a),b))
#    sig = N.sign(N.dot(a,rotateVector(b,90.)))

    if verbose is True:
        if sig > 0:
            print "b to the right of a"
        elif sig < 0:
            print "b to the left of a"
        else:
            print "b parallel/antiparallel to a"

    return sig


#def gaussian(npix=101,sx=5.,sy=5.,x0=0,y0=0,theta=0.):
#    
#    x = N.arange(npix) - npix/2
#    X, Y = N.meshgrid(x,x,indexing='ij')
#
#    norm = 1. / (2.*N.pi*sx*sy)
#    Z = norm * N.exp( -( (X-x0)**2./(2.*float(sx)**2.) + (Y-y0)**2./(2.*float(sy)**2.)))
#    
#    if theta != 0.:
#        Z = ndimage.rotate(Z,theta,reshape=False)
#        
#    return Z

def gaussian(npix=101,sx=5.,sy=5.,x0=0,y0=0,theta=0.,norm=None):
    
    x = N.arange(npix) - npix/2
    X, Y = N.meshgrid(x,x,indexing='ij')

    if norm is None:
        norm = 1. / (2.*N.pi*sx*sy)

    g = Gaussian2D(norm,x0,y0,sx,sy,N.radians(theta))
    Z = g(X,Y)

    return Z


#def getImageEigenvectors2D(image,thresh=0.001):
#
#    """Compute eigenvectors of image covariance matrix.
#
#    Parameters
#    ----------
#    image : 2-d array
#        Image.
#
#    thresh : float
#        Pixels as dim as ``'thresh`` (fraction) of peak pixel
#        brightness will be considered. Default 0.001 (= 0.1 %).
#
#    """
#    
#    xind, yind = N.argwhere(image>thresh*image.max()).T
#    coords = N.vstack((xind,yind))
#    cov = N.cov(coords)
#    evals, evecs = N.linalg.eig(cov)
#    idmax = N.argmax(evals)
#    evec1 = evecs[idmax]
#
#    return evals, evecs


def getImageEigenvectors(image,thresh=0.001,sortdescending=True):

    """Compute eigenvectors of image covariance matrix.

    Parameters
    ----------
    image : 2-d array
        Image.

    thresh : float
        Pixels as dim as ``'thresh`` (fraction) of peak pixel
        brightness will be considered. Default 0.001 (= 0.1 %).

    sortdescending : bool
        If True (the default), returns the eigenvalues and
        eigentvectors sorted by the eigenvalues in descending order
        (i.e. the first eigenvector is the largest). If False, keeps
        the original order.

    Returns
    -------

    evals : array
        1-d array of eigenvalues vor each axis in ``image``.

    evecs : aray
    """
    
    idxes = N.argwhere(image>thresh*image.max()).T
    coords = N.vstack(idxes)
    cov = N.cov(coords)
    eigenvalues, eigenvectors = N.linalg.eig(cov)

    if sortdescending is True:
        idxsort = N.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxsort]
        eigenvectors = eigenvectors[idxsort]
        
#    idmax = N.argmax(evals)
#    evec1 = evecs[idmax]

    return eigenvalues, eigenvectors


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
    
    return evec1, xind, yind, sig2
    
#    sort_indices = N.argsort(evals)[::-1]
#    evec1, evec2 = evecs[:,sort_indices]
#
#    return evec1, evec2





def getUnitVector(axis=1,ndim=2):

    """Get n-dimensional unit vector.

    Parameters
    ----------
    axis : int
        Direction of the vector, counting axes from 0 (the pythonic way) up to ``ndim``-1.

    ndim : int
        Dimensionality of the unit vector. Default 2.

    Returns
    -------
    uvec : array
        1-dim array of length ``ndim``, representing a unit vector
        along the given ``axis``, in ``ndim`` dimensions.


    Examples
    --------

    .. code-block:: python

       getUnitVector(axis=0,ndim=2)
         array([ 1.,  0.])  # 2-d unit vector along x-axis

       getUnitVector(1,2)
         array([ 0.,  1.])  # 2-d unit vector along y-axis

       getUnitVector(1,5)
         array([ 0.,  1.,  0.,  0.,  0.])  # 5-d unit vector along 2nd axis

    """
    
    uvec = N.eye(ndim)[axis]

    return uvec


def getAngle(a,b,pa=True):

    """Compute angle between two vectors a and b.

    Uses formula:

    .. math::

       \\tan \\theta = \\frac{|a \\times b|}{a \\cdot b}

    Parameters
    ----------
    a, b : array
        Vectors to measure angle between.

    pa : bool
        Position angle flag. If True, the measured angle will be
        measured as angle between vector ``a`` counter-clockwise from
        vector ``b``, and angle always non-negative. If False, the
        measured angle can be positive (i.e. ``a'' is ``angle``
        degrees counter-clockwise from ``b``), or negative
        (i.e. ``a`` is ``angle`` degrees clockwise from ``b``).

    Returns
    -------
    angle : float
        Angle in degrees between vectors ``a`` and ``b``. See
        description of ``pa`` for more details.

    """

    # compute angle using cross product formula
    aux1 = N.linalg.norm(N.cross(a,b))
    aux2 = N.dot(a,b)
    angle = N.degrees(N.arctan2(aux1,aux2))
    
    sign = whichside(a,b)
    angle = angle * sign

    if pa is True and angle < 0:
        angle += 180.

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
    

def findEmissionCentroid(img):

    """Return (fractional) pixel coordinates of the emission centroid in ``image``.


    Parameters
    ----------
    img : 2-d array
        2-d image.

    Returns
    -------
    x0, y0 : float
        Fractional pixel coordinates of the emission centroid in
        ``image``

    """
    
    x0, y0 = ndimage.center_of_mass(image)

    return x0, y0

    
def findEmissionPA(image):

    """Find position angle of dominant emission feature via cov. matrix analysis.

    Parameters
    ----------
    image : 2-d array

    Returns
    -------
    pa : float
        Postion angle (PA) of the dominant emission feature in
        ``image``, measured in degrees counter-clockwise from North
        (i.e. from the positive y-axis.
    """

#    x0,y0 = ndimage.center_of_mass(image)
    npix = image.shape[0]
    cpix = npix/2
#    p.imshow(image.T,origin='lower',interpolation='none')
    evec1,x_,y_,sig2 = imageToEigenvectors(image)
    measured = getAngle(evec1,ey)
#        p.axvline(cpix,c='w',lw=1)
#        scale = 20
#        m = N.tan(N.radians(measured-90.))
#        b = y0-m*x0
#        xl = 0.
#        yl = m*xl+b
#        xr = npix
#        yr = m*xr+b
#        p.plot((xl,xr),(yl,yr),ls='-',lw=1,c='b')
#        p.xlim(0,npix-1)
#        p.ylim(0,npix-1)
#        p.title('measured PA = %.2f, with x-axis = %.2f' % (measured,measured-90.))
#        p.gca().get_xaxis().set_visible(False)
#        p.gca().get_yaxis().set_visible(False)
#        p.waitforbuttonpress()
#        p.draw()


def findOrientation_loop(I):

    ey = getUnitVector(axis=1,ndim=2)
    angles = N.arange(0,181,10)

    for angle in angles:
        print angle
        p.clf()
        image = ndimage.rotate(I,angle,reshape=False)
        #x0,y0 = N.unravel_index(N.argmax(image),image.shape)
        x0,y0 = ndimage.center_of_mass(image)
        npix = image.shape[0]
        cpix = npix/2
        p.imshow(image.T,origin='lower',interpolation='none')
        #p.scatter(x0,y0,marker='x',s=25,c='b')
        evec1,x_,y_,sig2 = imageToEigenvectors(image)
        #p.plot(x_[:1],y_[:1],c='w',marker='o')
        measured = getAngle(evec1,ey)
        p.axvline(cpix,c='w',lw=1)
        scale = 20
        m = N.tan(N.radians(measured-90.))
        b = y0-m*x0
        xl = 0.
        yl = m*xl+b
        xr = npix
        yr = m*xr+b
        p.plot((xl,xr),(yl,yr),ls='-',lw=1,c='b')
        p.xlim(0,npix-1)
        p.ylim(0,npix-1)
        p.title('measured PA = %.2f, with x-axis = %.2f' % (measured,measured-90.))
        p.gca().get_xaxis().set_visible(False)
        p.gca().get_yaxis().set_visible(False)
        p.waitforbuttonpress()
        p.draw()
        print