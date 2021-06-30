__author__ = 'Robert Nikutta <robert.nikutta@gmail.com>'
__version__ = '20200913' # yyyymmdd

# std lib
import math
import itertools
import warnings
from copy import copy

# 3rd party
import numpy as np
from scipy import ndimage, integrate, optimize
import scipy.signal as signal
import pylab as plt
from astropy.modeling.functional_models import Gaussian2D

# own
#import ndiminterpolation as ndi
from . import ndiminterpolation as ndi


def halflight_radius(img,center=''):

    """
    
    Parameters
    ----------

    center : str
        One of: peak, centroid, origin

    """

    npix = img.shape[0]
    cpix = npix // 2
    x = np.linspace(-cpix,cpix,npix)
    X,Y = np.meshgrid(x,x,indexing='ij')

    if center == 'origin':
#        R = np.sqrt(X**2+Y**2)
        x0, y0 = 0, 0
    elif center == 'centroid':
        Ftot, xbar, ybar = get_centroid(img)
#        Rcentroid = np.sqrt((X-xbar+cpix)**2+(Y-ybar+cpix)**2)
        x0 = xbar - cpix
        y0 = ybar - cpix
    elif center == 'peak':
        xpeak, ypeak = np.unravel_index(np.argmax(img[:,cpix:]),img[:,cpix:].shape)
        ypeak += cpix
#        R = np.sqrt((X-cpix+cpix)**2+(Y-ypeak+cpix)**2)
        x0 = 0
        y0 = ypeak - cpix

    Ftot = img.sum()
    R = np.sqrt((X-x0)**2+(Y-y0)**2)
    
    def halflight(r,R_):
        mask = (R_<=r)
        F = img[mask].sum()
        aux = F/Ftot - 0.5
        
        return aux

#    Rhl = optimize.brentq(halflight,0,100,args=(R)) # find root of halflight function
    Rhl = optimize.brentq(halflight,0,cpix,args=(R)) # find root of halflight function

    return Rhl


def mask_outside_square(img,frac=0.1,x0=0,y0=0):

    
    I = copy(img)

    npix = I.shape[0]
    cpix = npix//2
    a = int(np.ceil(npix*frac))

    x = np.linspace(-cpix,cpix,npix)
    X,Y = np.meshgrid(x,x,indexing='ij')
    
    mask = (np.abs(X-x0) <= a//2) & (np.abs(Y-y0) <= a//2)
    I[~mask] = 0.

    return I
    
    

def get_cutout(img,frac=0.1):

    npix = img.shape[0]
    cpix = npix//2

    a = int(np.ceil(npix*frac))
#    l = cpix - a//2
#    r = cpix + a//2
#    cutout = img[l:r,l:r]

#    cutout = img[cpix-a//2,cpix+1+a//2]
#    cutout = img[cpix+1-a//2:cpix+a//2]
    cutout = img[cpix-a//2:cpix+1+a//2,cpix-a//2:cpix+1+a//2]

    return cutout
    

def get_moment_raw_matmul(img,orders):

    r"""Raw image moment via two matrix multiplications.

    This is 14 times faster than the direct summation method.

    Exploits that:

    .. math::

       \sum_i^n \sum_j^m i^p j^q f_{ij} = \left(\vec i^p\right)^T \cdot \vec f \cdot \vec j^q

    with :math:`\vec i^p = (1,2^p,3^p,...,n^p)`, :math:`\vec j^q =
    (1,2^q,3^q,...,m^q)` and :math:`\cdot` are dot products on the
    RHS.

    Parameters
    ----------
    img : 2-d array
        (NxM) shaped image array.

    orders : 2-tuple of integers
        Image moment order to compute. (p,q) type tuple.

    Returns
    -------
    Mpq : float
        Geometric image moment of order p+q.

    Example
    -------
    .. code-block:: python

       img = np.eye(2)
       array([[1., 0.],
              [0., 1.]])

       # M00, i.e. the sum / mass of the image
       get_moment_raw_matmul(img,(0,0))
       2.0

    """

    p, q = orders
    n, m = img.shape
    i = np.arange(n)**p
    j = np.arange(m)**q

    Mpq = np.dot(np.dot(i.T,img),j)

    return Mpq


def get_all_moments_raw_matmul(img,pmax=2):

    """All raw image moments up to oder (pmax,pmax).

    Compute all (p+q) orders at once, with :math:`p,q \in [0,pmax]`.

    Parameters
    ----------
    img : 2-d array
        (nxm) shaped image array.

    pmax : int
        Max p,q moment orders.

    Returns
    -------
    M : 2-d array
        2-d array with shape (pmax+1,pmax+1). Contains all geometric
        moments from (0,0) up to (pmax,pmax). To access moment (i,j),
        just index the result array (see Examples).

    Examples
    --------
    .. code-block:: python

       img = gaussian(201,5,20) # npix,sigmax,sigmay
       M = get_all_moments_raw_matmul(img,pmax=2):
       M
         array([[9.88438828e-01, 4.94219414e+01, 2.83323273e+03],
                [4.94219414e+01, 2.47109707e+03, 1.41661637e+05],
                [2.49580804e+03, 1.24790402e+05, 7.15391265e+06]])

       M[0,0]
         0.999999497938336

       M[2,0]
         10024.994966831815

       M[0,2]
         10399.989518069591
    """
    
    n, m = img.shape
    maxorder = pmax+1
    
    I = np.zeros((n,maxorder))
    J = np.zeros((m,maxorder))

    for p in range(maxorder):
        I[:,p] = np.arange(n)**p
        J[:,p] = np.arange(m)**p

    M = np.dot(np.dot(I.T,img),J)

    return M


def get_moment_central_matmul(img,orders):

    r"""Central image moment via two matrix multiplications.

    This is 14 times faster than the direct summation method.

    Exploits that:

    .. math::

       \mu_{pq} = \sum_i^n \sum_j^m (i-\bar i)^p (j-\bar j)^q f_{ij} = \left((\vec i - \bar i)^p\right)^T \cdot \vec f \cdot (\vec j - \bar j)^q

    with :math:`\vec i^p = (1,2^p,3^p,...,n^p)`, :math:`\vec j^q =
    (1,2^q,3^q,...,m^q)`, :math:`\bar i = M_{01}/M_{00}`, :math:`\bar
    j = M_{10}/M_{00}`, and :math:`\cdot` are dot products on the RHS.

    Parameters
    ----------
    img : 2-d array
        (NxM) shaped image array.

    orders : 2-tuple of integers
        Image moment order to compute. (p,q) type tuple.

    Returns
    -------
    mupq : float
        Central image moment of order p+q.

    Example
    -------
    .. code-block:: python

       img = np.eye(2)
         array([[1., 0.],
                [0., 1.]])

       # mu00, i.e. the sum / mass of the image
       get_moment_central_matmul(img,(0,0))
         2.0

       # mu20
       get_moment_central_matmul(img,(2,0))
         0.5

       # mu02
       get_moment_central_matmul(img,(0,2))
         0.5

    """

    M00 = get_moment_raw_matmul(img,(0,0))
    ibar = get_moment_raw_matmul(img,(1,0)) / M00
    jbar = get_moment_raw_matmul(img,(0,1)) / M00
    
    p, q = orders
    n, m = img.shape
    i = (np.arange(n)-ibar)**p
    j = (np.arange(m)-jbar)**q

    mupq = np.dot(np.dot(i.T,img),j)

    return mupq





def get_cov_from_moments(img):
    mu00 = get_moment_central(img,(0,0))
    mu11p = get_moment_central(img,(1,1)) / mu00
    mu20p = get_moment_central(img,(2,0)) / mu00
    mu02p = get_moment_central(img,(0,2)) / mu00
    cov = np.zeros((2,2))
    cov[0,0] = mu20p
    cov[0,1] = mu11p
    cov[1,0] = mu11p
    cov[1,1] = mu02p

    return cov


def get_centroid(img):
    M00 = get_moment_raw(img,(0,0))
    xbar = get_moment_raw(img,(1,0)) / M00
    ybar = get_moment_raw(img,(0,1)) / M00

    return M00, xbar, ybar


def get_rgyr(img):

    m00 = get_moment_central(img,(0,0))
    m20 = get_moment_central(img,(2,0))
    m02 = get_moment_central(img,(0,2))

    rgx = np.sqrt(m20/m00)
    rgy = np.sqrt(m02/m00)

    return rgx, rgy


def get_eigenvalues(cov):
    eigenvals = np.linalg.eigvals(cov)
    
    return eigenvals


def get_angle(img=None,cov=None):

    """"Compute image position angle.

    At least one of `img' or `cov` must be provided. Works for PA
    between 0 and 180 degrees (measured E from N),
    i.e. counter-clockwise from North.

    Parameters
    ----------
    img : array or None
        The image of which to compute the position angle PA.

    cov : 2x2 array or None
        The covariance matrix of img, of any. If cov os provided, it
        will be used to compute the PA, even if img is provided as
        well.

    Returns
    -------
    angle : float
        The position angle of `img` (or of the image represented by
        `cov`, in degrees East from North.

    """

    if cov is None:
        if img is not None:
            cov = get_cov_from_moments(img)
        else:
            raise Exception("`cov` is None and `img` is None. One of the two must be provided.")

    else:
        if img is not None:
            warnings.warn("Both `img` and `cov` were provided. The position angle will be computed from `cov`.")

    angle = 0.5 * np.arctan2((2*cov[0,1]),(cov[0,0]-cov[1,1])) # TODO: check formula

    angle = 90 - np.degrees(-angle)

    return angle
    

# TODO: allow for axes switching (auto, or by user)
def get_elongation(cov):
    eigenvals = get_eigenvalues(cov)
#    elong = np.sqrt(eigenvals[0]/eigenvals[1])
    elong = np.sqrt(eigenvals[1]/eigenvals[0])

    return elong


def get_elongation(cov):

    """Get elongation (or aspect ratio) y/x of input image.

    If the shape of the input image is (2,2), then it is assumed to be
    the covariance matrix of the image. From such a covariance matrix
    the elongation can be computed directly.

    If the shape is not (2,2), the input is assumed to be the image
    itself, and the covariance matrix is computed first.
    """
    
    if cov.shape != (2,2):
        cov = get_cov_from_moments(cov)
    
    eigenvals = get_eigenvalues(cov)
#    elong = np.sqrt(eigenvals[0]/eigenvals[1])
    elong = np.sqrt(eigenvals[1]/eigenvals[0])

    return elong


def circle(npix,r,x0=0,y0=0):

    I = np.zeros((npix,npix))
    cpix = npix//2
    x = np.linspace(-cpix,cpix,npix)
    X,Y = np.meshgrid(x,x,indexing='ij')
    mask = np.sqrt((X-x0)**2 + (Y-y0)**2) <= r
    I[mask] = 1.

    return I

def square(npix,a,x0=0,y0=0):

    I = np.zeros((npix,npix))
    cpix = npix//2
    x = np.linspace(-cpix,cpix,npix)
    X,Y = np.meshgrid(x,x,indexing='ij')
    mask = (np.abs(X-x0) <= a//2) & (np.abs(Y-y0) <= a//2)
    I[mask] = 1.

    return I

def rectangle(npix,a,b,x0=0,y0=0,theta=0.):

    I = np.zeros((npix,npix))
    cpix = npix//2
    x = np.linspace(-cpix,cpix,npix)
    X,Y = np.meshgrid(x,x,indexing='ij')
    mask = (np.abs(X-x0) <= a//2) & (np.abs(Y-y0) <= b//2)
    I[mask] = 1.

    return I


#def moment_raw_upto(img,maxorder=7):
#    n = maxorder + 1
#    res = np.zeros(n)
#    for j in range(n):
#        res[j] = moment_raw(img,order=j)
#
#    return res

def get_hu_moment(img,orders):
    pass

def get_moment(img,orders,central=False,scaleinvariant=False):

    p, q = orders
    
    if central is True:
        mu00 = get_moment(img,(0,0))
        xbar = get_moment(img,(1,0)) / mu00  # these seem correct, as the give the correct
        ybar = get_moment(img,(0,1)) / mu00  # coords of the center of gravity
    else:
        xbar = 0.
        ybar = 0.
        
    lattice = np.arange(img.shape[0])
    X, Y = np.meshgrid(lattice,lattice,indexing='ij')
    mu = np.sum((X-xbar)**p * (Y-ybar)**q * img)

#    if scaleinvariant is True and p>=1 and q>=1:
    if scaleinvariant is True and p+q>=2:
        mu00 = get_moment(img,(0,0))
        aux = mu00**(1+(p+q)/2.)
        mu = mu / aux
    
    return mu


def get_moment_raw(img,orders,xbar=0.,ybar=0.):
    
    p, q = orders
    lattice = np.arange(img.shape[0])
    X, Y = np.meshgrid(lattice,lattice,indexing='ij')
    M = np.sum((X-xbar)**p * (Y-ybar)**q * img)

    return M


def get_moment_central(img,orders):

    p, q = orders
    M00 = get_moment_raw(img,(0,0))
    xbar = get_moment_raw(img,(1,0)) / M00  # these seem correct, as the give the correct
    ybar = get_moment_raw(img,(0,1)) / M00  # coords of the center of gravity
    mu = get_moment_raw(img,orders,xbar=xbar,ybar=ybar)

    return mu


def get_moment_scaleinvariant(img,orders):

    p, q = orders
    if p+q < 2:
        raise Exception('Moment orders must satisfy p+q>=2')

    mu = get_moment_central(img,orders)
    mu00 = get_moment_raw(img,(0,0))
    aux = mu00**(1+(p+q)/2.)
    eta = mu / aux

    return eta
        
    

class Moment:

    def __init__(self,img):
        self.img = img
        lattice = np.arange(self.img.shape[0])
        self.X, self.Y = np.meshgrid(lattice,lattice,indexing='ij')
        self.M00, self.xbar, self.ybar = get_centroid(self.img)

    def __call__(self,p,q):
        self.p, self.q = p, q
        self.raw = get_moment_raw(self.img,(self.p,self.q))
        self.central = get_moment_central(self.img,(self.p,self.q)) #,xbar=self.xbar,ybar=self.ybar)
        
        try:
            self.scaleinvariant = get_moment_scaleinvariant(self.img,(self.p,self.q))
        except:
            warnings.warn("Can't compute scale-invariant moments. p+q>=2 must hold.")
            
        self.cov = get_cov_from_moments(self.img)
        self.eigenvals = get_eigenvalues(self.cov)
        self.elongation = get_elongation(self.cov)
        self.angle = get_angle(self.cov) # angle(largest EV, closest axis)
        

class MomentAnalytics:

    def __init__(self):
        ord = (0,1,2,3,4) #,3,4)
#        self.orders = list(itertools.product(ord,ord))
        self.orders = ((0,0),(2,0),(0,2),(3,0),(0,3),(4,0),(0,4))

        
    def __call__(self,img=None,a=20,b=10,xoff=0.,yoff=0.,theta=0.,model='gaussian',npix=241,norm=1.,eps=0.3):

        if img is None:
            self.a, self.b, self.xoff, self.yoff, self.theta =  a, b, xoff, yoff, theta
            func = globals()[model]
            self.npix = npix
            self.img = func(self.npix,self.a,self.b,self.xoff,self.yoff,self.theta)
#circle            self.img = func(self.npix,self.a,self.xoff,self.yoff) #circle
#            self.img = norm * self.img/self.img.max()
        else:
            self.img = img
            
        dummy, self.xbar, self.ybar = get_centroid(self.img)
        
        for (p,q) in self.orders:
#slow            self.set_moment_name(p,q,get_moment_raw(self.img,(p,q)),prefix='M')
#slow            self.set_moment_name(p,q,get_moment_central(self.img,(p,q)),prefix='m')
            self.set_moment_name(p,q,get_moment_raw_matmul(self.img,(p,q)),prefix='M') #fast
            self.set_moment_name(p,q,get_moment_central_matmul(self.img,(p,q)),prefix='m') #fast
#eta            if (p+q>=2):
#eta                self.set_moment_name(p,q,get_moment_scaleinvariant(self.img,(p,q)),prefix='eta')

        self.cov = get_cov_from_moments(self.img)
        print("self.cov ", self.cov)
        self.eigenvals = get_eigenvalues(self.cov)
        self.elongation = get_elongation(self.cov)
#        self.angle = get_angle(self.cov) # angle(largest EV, closest axis)
        self.angle = get_angle(cov=self.cov) # angle(largest EV, closest axis)
        self.skewx = self.m03 / np.sqrt(self.m02**3.)
        self.skewy = self.m30 / np.sqrt(self.m20**3.)
        self.gini = gini(self.img)

        npix = self.img.shape[0]
        cpix = npix//2
        self.fluxasym = np.sum(self.img[:,cpix:])/np.sum(self.img)
                
    def set_moment_name(self,p,q,val,prefix):
        setattr(self, '%s%d%d' % (prefix,p,q), val)
        

#def moment(img,orders=(0,),central=False):
#
#    if not isinstance(orders,(list,tuple,np.ndarray)):
#        orders = (orders,)
#        
#    moments = np.zeros(len(orders))
#
#    if central is True:
#        pass
#    
#    lattice = np.arange(img.shape[0])
#    X, Y = np.meshgrid(lattice,lattice,indexing='ij')
#    for j,order in enumerate(orders):
#        print(order)
#        moments[j] = np.sum(X**order * Y**order * img)
#    
#    return moments


def gini(arr):
    """Compute Gini coefficient of array `arr`.

    In mathematical notation, where argument ``arr`` is a discrete
    array with values :math:`I_i`:

    .. math::

       G = \\frac{\\sum_i (2 i - n - 1)\\cdot I_i}{n \\sum_i I_i} 

    where the array values :math:`I_i` are sorted in ascending order,
    and the :math:`i` are the array indices of the sorted array.

    """
    
    arr = arr.flatten()
    n = arr.size
    idx = np.arange(1,n+1)
    eps = 1.e-10

    # offset minor negative vales # TODO: should probably yell if MIN is too far away from 0.
    MIN = np.min(arr)
    if MIN < 0.:
        arr -= MIN

    arr = arr + eps  # make all pixels non-zero
    arr = np.sort(arr)
    
    G = np.sum((2*idx-n-1)*arr) / (n*np.sum(arr))

    return G
    

def gini_pure(arr):
    """Compute Gini coefficient of array `arr`.

    In mathematical notation, where argument ``arr`` is a discrete
    array with values :math:`I_i`:

    .. math::

       G = \\frac{\\sum_i (2 i - n - 1)\\cdot I_i}{n \\sum_i I_i} 

    where the array values :math:`I_i` are sorted in ascending order,
    and the :math:`i` are the array indices of the sorted array.

    """
    
    arr = arr.flatten()
    n = arr.size
    print(n)
    idx = np.arange(1,n+1)
    print(n,idx)
    eps = 1.e-10

    # offset minor negative vales # TODO: should probably yell if MIN is too far away from 0.
    MIN = np.min(arr)
    if MIN < 0.:
        arr -= MIN

#    arr = arr + eps  # make all pixels non-zero
    arr = np.sort(arr)
    
    print(n,idx,MIN,eps,arr)
#    mean = np.sum(arr)/n
#    G = np.sum((2*idx-n-1)*arr) / (n*n*mean)

    G = np.sum((2*idx-n-1)*arr) / (n*np.sum(arr))

    # make it an unbiased estimator
    G = G * n / (n-1)

    return G
    

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
    rvec : array
        Rotated vector.

    Examples
    --------

    .. code-block:: python

       vec = np.array([0.,1.]) # unit vector along postive y-axis
       rotateVector(vec,deg=90.)
         array([ -1., 0.])

       rotateVector(vec,deg=-90.)
         array([ 1., 0.])

       rotateVector(vec,deg=45.)
         array([-0.70710678,  0.70710678])

    """

    angle = np.radians(deg)
    s = math.sin(angle)
    c = math.cos(angle)
    matrix = np.array([[ c, s],
                       [-s, c]])

    # transposing matrix to make positive angles rotate
    # counter-clockwise, i.e. as is convention in mathematics
    rvec = np.dot(matrix.T,vec)

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

#    sig = np.sign(np.dot(a,rot90ccw(b)))
    sig = np.sign(np.dot(rot90ccw(a),b))
#    sig = np.sign(np.dot(a,rotateVector(b,90.)))

    if verbose is True:
        if sig > 0:
            print("b to the right of a")
        elif sig < 0:
            print("b to the left of a")
        else:
            print("b parallel/antiparallel to a")

    return sig


#def gaussian(npix=101,sx=5.,sy=5.,x0=0,y0=0,theta=0.):
#    
#    x = np.arange(npix) - npix/2
#    X, Y = np.meshgrid(x,x,indexing='ij')
#
#    norm = 1. / (2.*np.pi*sx*sy)
#    Z = norm * np.exp( -( (X-x0)**2./(2.*float(sx)**2.) + (Y-y0)**2./(2.*float(sy)**2.)))
#    
#    if theta != 0.:
#        Z = ndimage.rotate(Z,theta,reshape=False)
#        
#    return Z

def gaussian(npix=101,sx=5.,sy=5.,x0=0,y0=0,theta=0.,norm=None):

    x = np.arange(npix) - npix//2
    X, Y = np.meshgrid(x,x,indexing='ij')

    if norm is None:
        norm = 1. / (2.*np.pi*sx*sy)

    g = Gaussian2D(norm,x0,y0,sx,sy,np.radians(theta))
    Z = g(X,Y)

    return Z

def gaussian_asymmetric(npix=101,sx=5.,sy=5.,x0=0,y0=0,theta=0.,norm=None):

    """Construct a two-component Gaussian"""
    
    x = np.arange(npix) - npix//2
    X, Y = np.meshgrid(x,x,indexing='ij')

    if norm is None:
        norm = 1. / (2.*np.pi*sx*sy)

    g1 = Gaussian2D(norm,x0,x0,sx,sy,np.radians(theta))
    g2 = Gaussian2D(norm,x0,y0,sx,sx,np.radians(theta))
#    g1 = Gaussian2D(norm,x0,x0,sx,sy,np.radians(theta))
#    g2 = Gaussian2D(norm,x0,y0,sx,sy,np.radians(theta))
    g = g1 + g2
    Z = g(X,Y)

    return Z


def skew_normal_2d_r(npix,sx=10,sy=10,x0=0,y0=0,xlambda=0,ylambda=0):

    # import rpy2 and sn lib
    from rpy2.robjects.packages import importr
    import rpy2.robjects as ro
    sn = importr('sn')

    # create and evaluate 2d skew normal

    # 1. Create symmetric 2d Gaussian and get its cov
    img = gaussian(npix=npix,sx=sx,sy=sy,x0=0,y0=0,theta=0,norm=None)
    cov = get_cov_from_moments(img)
    xx, yy = cov.diagonal()
    print("xx, yy = ", xx, yy)
    
    # create and evaluate 2d skew normal
    ro.r('alpha <-  c(%f, %f)' % (xlambda,ylambda))  # skewness parameter; the second value controls y skewness
    ro.r('Omega <-  matrix(c(%f, 0., 0., %f), 2, 2)' % (xx,yy)) # cov of a Gaussian created like this: g = morphology.gaussian(101,10,20,0,0,0)
    npix2 = npix // 2
    xran = ro.r('xran  <-  seq(-%d, %d, length=%d)' % (npix2,npix2,npix))
    yran = ro.r('yran  <-  seq(-%d, %d, length=%d)' % (npix2,npix2,npix))
    z = ro.r('z <-  outer(xran, yran, FUN=Vectorize( function(x, y) dmsn(c(x, y), c(%d, %d), Omega, alpha) )  )' % (x0,y0)) # c(xoff,yoff) in pixels
    
    Z = np.asarray(z) # convert R array 'z' to numpy array 'Z'

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
#    xind, yind = np.argwhere(image>thresh*image.max()).T
#    coords = np.vstack((xind,yind))
#    cov = np.cov(coords)
#    evals, evecs = np.linalg.eig(cov)
#    idmax = np.argmax(evals)
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
    
    idxes = np.argwhere(image>thresh*image.max()).T
    coords = np.vstack(idxes)
    cov = np.cov(coords)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    if sortdescending is True:
        idxsort = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxsort]
        eigenvectors = eigenvectors[idxsort]
        
#    idmax = np.argmax(evals)
#    evec1 = evecs[idmax]

    return eigenvalues, eigenvectors


def imageToEigenvectors(image):

    thresh = 0.001 # pixels as dim as this (in percent) of peak pixel brightness will be considered
#    y, x = np.argwhere(image>thresh*image.max()).T
    xind, yind = np.argwhere(image>thresh*image.max()).T
#    print("x.sum(), y.sum() = ", x.sum(), y.sum())
#    y, x = np.nonzero(image)
    x = xind - xind.mean()
    y = yind - yind.mean()
    coords = np.vstack((x,y))
    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)
    print("evals = ", evals)
#    idmax = np.argmin(evals)
    idmax = np.argmax(evals)
    evec1 = evecs[idmax]
    print("evec1 = ", evec1)

#    sig = whichside(evec1,np.array([0.,1.]))
    sig = whichside(np.array([0.,1.]),evec1)
    sig2 = whichside(evec1,np.array([0.,1.]))
#    sig = whichside(evec1,np.array([1.,0.]))
    print("sig = ", sig, sig2)
    
#    if sig == -1.:
##        evec1[0] = -evec1[0]
#        evec1 = -evec1
    
    return evec1, xind, yind, sig2
    
#    sort_indices = np.argsort(evals)[::-1]
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
         array([ 1.,  0.]) # 2-d unit vector along x-axis

       getUnitVector(1,2)
         array([ 0.,  1.]) # 2-d unit vector along y-axis

       getUnitVector(1,5)
         array([ 0.,  1.,  0.,  0.,  0.]) # 5-d unit vector along 2nd axis

    """
    
    uvec = np.eye(ndim)[axis]

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
    aux1 = np.linalg.norm(np.cross(a,b))
    aux2 = np.dot(a,b)
    angle = np.degrees(np.arctan2(aux1,aux2))
    
    sign = whichside(a,b)
    angle = angle * sign

    if pa is True and angle < 0:
        angle += 180.

    return angle 






def get_moment_old(image,radius=1.,angular='a',m=1):

    """Compute m-th moment of image.

    Parameters
    ----------
    image : array (or instance of Image?)
       2d array (or instance of Image?)
    """
    
    npix = image.shape[0]
    x = np.linspace(-radius,radius,npix)
    theta = (x,x)
    ip = ndi.NdimInterpolation(image,theta,mode='lin')

    angulars = {'a':np.cos,'b':np.sin}
    ang = angulars[angular]


    def get_xy(r,phi):
        x_ = r*np.cos(phi)
        y_ = r*np.sin(phi)
        return (x_,y_)


    def getI(phi,r):
        x_, y_ = get_xy(r,phi)
        I = ip((x_,y_))
        res = I * ang(m*phi) * r**m
        return res

    
    res = integrate.dblquad(getI, 0., radius, lambda x:0., lambda x:2*np.pi,epsabs=1e-03, epsrel=1e-03)[0]
#    integrate.dblquad(func, -pi/2, pi/2, lambda x:-pi/2, lambda x:pi/2)[0]

#    return ip, theta
    return res


def get_power(image,m,r=1.):

    a0 = get_moment(image,radius=r,angular='a',m=0)

    if m == 0:
        P = (a0*np.log(r))**2
    elif m > 0:
        am = get_moment(image,radius=r,angular='a',m=m)
        bm = get_moment(image,radius=r,angular='b',m=m)
        P = (am**2+bm**2) / (2.*m**2*r**(2*m))

    return P


def get_wavelet(npix,a):
    print(a)
    x = np.arange(-npix//2,1+npix//2,1)
    print(x)
    X, Y = np.meshgrid(x,x)
    g = (2.-(X**2.+Y**2.)/a**2.)*np.exp(-(X**2.+Y**2.)/(2.*a**2.))
    return g


def get_wavelet_elliptical_mexh(npix,a,b):
    print(a,b)
    xa = np.arange(-npix//2+1,1+npix//2,1)
    xb = np.arange(-npix//2+1,1+npix//2,1)
    X,Y = np.meshgrid(xa,xb)
    g = (1./(2*np.pi*a**3*b**3)) * (a**2+b**2-(X**2/(a/b)**2)-(Y**2/(b/a)**2)) * np.exp(-0.5*((X**2/a**2)+(Y**2/b**2)))
    return g

#def get_wavelet_elliptical_mexh_vuong(npix,s=10.,sigma=1.,a=0,b=0):
#def get_wavelet_elliptical_mexh_vuong(npix,s=5.,sigma=2.):
def get_wavelet_elliptical_mexh_vuong(npix,sx=5.,sy=5.):
#    print(a,b)
    sigma = sx/float(sy)
    xa = np.arange(-npix//2+1,1+npix//2,1)
    xb = np.arange(-npix//2+1,1+npix//2,1)
    X,Y = np.meshgrid(xa,xb)

#    g = (1./(2*np.pi*a**3*b**3)) * (a**2+b**2-(X**2/(a/b)**2)-(Y**2/(b/a)**2)) * np.exp(-0.5*((X**2/a**2)+(Y**2/b**2)))
    K = sigma**2./(16*np.pi**3.)
#    g = (1./(s*np.sqrt(K)) * (2. - (X-a)**2./(s*sigma)**2 - (Y-a)**2./s**2) * np.exp(-0.5*((X-a)**2/(s*sigma)**2 - (Y-b)**2/s**2)))
#    g = (1./(s*np.sqrt(K)) * (2. - X**2./(s*sigma)**2. - Y**2./s**2.) * np.exp(-0.5*(X**2/(s*sigma)**2. + Y**2/s**2.)))
#
#    g = (1./(sx*np.sqrt(K)) * (2. - X**2./sx**2. - Y**2./sy**2.) * np.exp(-0.5*(X**2/sx**2. + Y**2/sy**2.)))
    g = (2. - X**2./sx**2. - Y**2./sy**2.) * np.exp(-0.5*(X**2/sx**2. + Y**2/sy**2.))
    return g

def get_wavelet_elliptical_mexh_vuong_fast(npix,sx=5.,sy=5.):
#    sigma = sx/float(sy)
    x_ = np.arange(-npix//2+1,1+npix//2,1)
    X, Y = np.meshgrid(x_,x_,indexing='ij')
    AUX = (X**2./sx**2. + Y**2./sy**2.)
#    K = sigma**2./(16*np.pi**3.)
#    g = (1./(sx*np.sqrt(K)) * (2. - X**2./sx**2. - Y**2./sy**2.) * np.exp(-0.5*(X**2/sx**2. + Y**2/sy**2.)))
    g = (2. - AUX) * np.exp(-0.5*AUX)
    return g


def get_wavelet_elliptical_mexh_gaillot(npix,sx=5.,sigma=1.):
    x_ = np.arange(-npix//2+1,1+npix//2,1)
    X, Y = np.meshgrid(x_,x_)
    AUX = (X**2. + (sigma*Y)**2.) / sx**2.
    g = (2. - AUX) * np.exp(-0.5*AUX)
    return g

def get_wavelet_elliptical_mexh_gaillot_full(npix,x0=0,y0=0,sx=5.,sigma=1.):
    x_ = np.arange(-npix//2+1,1+npix//2,1)
    X, Y = np.meshgrid(x_,x_)
    AUX = ((X-x0)**2. + sigma**2*(Y-y0)**2.) / sx**2.
    g = (2. - AUX) * np.exp(-0.5*AUX)
    return g


def plot():
    a_ = 17
    cmap=plt.cm.viridis
    IMG = img
    npix=IMG.shape[0]
    wavelet = get_wavelet(IMG.shape[0],a_)
#    conv = convolve_fft(IMG,wavelet/a_)
    conv = convolve_fft(IMG,wavelet//a_)
    ax1.cla()
    ax1.imshow(IMG.T,origin='lower',interpolation='none',cmap=cmap)
    ax1.contour(IMG.T,5,linestyles='-',colors='w')
    ax2.cla()
    ax2.imshow(wavelet.T,origin='lower',interpolation='none',cmap=cmap)
    ax3.cla()
    ax3.imshow(conv.T,origin='lower',interpolation='none',cmap=cmap)
    ax3.contour(conv.T,5,linestyles='-',colors='w')
    ax3.plot((npix//2-133/2,npix//2+133/2),(npix//2+133,npix//2+133),ls='-',lw=3,c='w')
    ax4.cla()
    ax4.semilogx(a,results,'b.-',ms=5)
    ax4.axvline(a_)


def work(a):
#    return convolve_fft(img,get_wavelet(img.shape[0],a)/a).mean()/a
    return convolve_fft(img,get_wavelet(img.shape[0],a)/a).mean()/a

    
#
#
#
#    
#    X, Y = np.meshgrid(x,x)
#    R = np.sqrt(X**2+Y**2)
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
    waves = np.linspace(2.2,18.5,40)
    angles = np.linspace(0,90,40)
    res = np.zeros((waves.size,angles.size))
    for iw,w in enumerate(waves):
        print(w)
        for ia,a in enumerate(angles):
            print(a)
            img = cube.get_image((a,5,0,w))
            ratio = img[:,idx+1:].sum()/img[:,:idx].sum()
            res[iw,ia] = ratio

            
def fluxdensity_i_wave(cube):
    waves = np.linspace(2.2,18.5,40)
    angles = np.linspace(0,90,40)
    res = np.zeros((waves.size,angles.size))
    for iw,w in enumerate(waves):
        print(w)
        for ia,a in enumerate(angles):
            print(a0)
            img = cube.get_image((a,5,0,w))
            aux = img.sum()
            res[iw,ia] = aux

    return waves, angles, res


def get_wavelet(npix,a):
    x = np.arange(-npix//2,1+npix//2,1)
    X,Y = np.meshgrid(x,x)
    g = (2.-(X**2+Y**2)/a**2)*np.exp(-(X**2+Y**2)/(2.*a**2))
    return g

def eq11(I,sig,a):
    return 2*(I/a) * (1.+(sig**2/a**2))**(-2.)


# y = eq11(1.,10,a)

def test1():
    G1 = getGaussian(101,10,-5, 5,1.)
    G2 = getGaussian(101,10, 5,-5,1.)
    G3 = G1+G2

    a = np.linspace(1,50)
    wmaxes = [signal.convolve2d(img,get_wavelet(npix,a_)/a_,mode='same').max()/a_ for a_ in a]

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(G3.T,origin='lower',interpolation='none',cmap=plt.cm.inferno)
    ax2.plot(a,wmaxes,'b.-')
    ax2.set_xlim(min(a),max(a))

# TEST CASES WITH FIGURES FOR PAPER

def symmetric1(npix=51,savefile='symmertric1.npz'):

    x = np.arange(1.,npix+1)
    print(x.size)
#    x = np.arange(npix)

    print("Computing images")

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

    print("Computing convolutions")
    convomaxvals = np.zeros((x.size,len(images)))
    for jimg, img in enumerate(images):
        wl = wavelets[jimg]
        for jx, x_ in enumerate(x):
            if jx % 5 == 0: print(x_)
#        wl = get_wavelet(npix,x_)
            C = signal.convolve2d(img,wl/x_) #/ float(x_)
#            C = signal.convolve2d(img,wl/float(x_))
            convomaxvals[jx,jimg] = C.max() / x_
    
    if savefile is not None:
        print("Storing results in file '{:s}'".format(savefile))
        with open(savefile,'w') as f:
#            np.savez(savefile,G=G,Gh=Gh,Gv=Gv)
            np.savez(savefile,x=x,images=images,wavelets=wavelets,convomaxvals=convomaxvals)
        print("Done.")

    
def plot_symmetric1(loadfile='symmetric1.npz'):

    data = np.load(loadfile)
    x = data['x']
    images = data['images']
    print("len(images) = ", len(images))
    wavelets = data['wavelets']
    convomaxvals = data['convomaxvals']

    print("images max vals = ")
    print([img.max() for img in images])
    print([img.sum() for img in images])
    
    fig = plt.figure(figsize=(6,6))
    for j in range(0,len(images)):
        ax = fig.add_subplot(3,3,j+1)
        ax.imshow(images[j].T,origin='lower')

    for j in range(0,len(images)):
        ax = fig.add_subplot(3,3,j+1+3)
        ax.imshow(wavelets[j].T,origin='lower')
        
    for j in range(0,len(images)):
        ax = fig.add_subplot(3,3,j+1+6)
#        C = convomaxvals[j-3]
        C = convomaxvals[:,j]
#        C /= np.array(x).astype('float')
        ax.semilogx(x,C,'b-')
        xmax = x[np.argmax(C)]
        ax.axvline(xmax)
        ax.set_title("xmax = {:g}".format(xmax))

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
    cpix = npix//2
#    plt.imshow(image.T,origin='lower',interpolation='none')
    evec1,x_,y_,sig2 = imageToEigenvectors(image)
    measured = getAngle(evec1,ey)
#        plt.axvline(cpix,c='w',lw=1)
#        scale = 20
#        m = np.tan(np.radians(measured-90.))
#        b = y0-m*x0
#        xl = 0.
#        yl = m*xl+b
#        xr = npix
#        yr = m*xr+b
#        plt.plot((xl,xr),(yl,yr),ls='-',lw=1,c='b')
#        plt.xlim(0,npix-1)
#        plt.ylim(0,npix-1)
#        plt.title('measured PA = %.2f, with x-axis = %.2f' % (measured,measured-90.))
#        plt.gca().get_xaxis().set_visible(False)
#        plt.gca().get_yaxis().set_visible(False)
#        plt.waitforbuttonpress()
#        plt.draw()


def findOrientation_loop(I):

    ey = getUnitVector(axis=1,ndim=2)
    angles = np.arange(0,181,10)

    for angle in angles:
        print(angle)
        plt.clf()
        image = ndimage.rotate(I,angle,reshape=False)
        #x0,y0 = np.unravel_index(np.argmax(image),image.shape)
        x0,y0 = ndimage.center_of_mass(image)
        npix = image.shape[0]
        cpix = npix//2
        plt.imshow(image.T,origin='lower',interpolation='none')
        #plt.scatter(x0,y0,marker='x',s=25,c='b')
        evec1,x_,y_,sig2 = imageToEigenvectors(image)
        #plt.plot(x_[:1],y_[:1],c='w',marker='o')
        measured = getAngle(evec1,ey)
        plt.axvline(cpix,c='w',lw=1)
        scale = 20
        m = np.tan(np.radians(measured-90.))
        b = y0-m*x0
        xl = 0.
        yl = m*xl+b
        xr = npix
        yr = m*xr+b
        plt.plot((xl,xr),(yl,yr),ls='-',lw=1,c='b')
        plt.xlim(0,npix-1)
        plt.ylim(0,npix-1)
        plt.title('measured PA = {:.2f}, with x-axis = {:.2f}'.format(measured,measured-90.))
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.waitforbuttonpress()
        plt.draw()
        print()
