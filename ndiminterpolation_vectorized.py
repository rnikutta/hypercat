__author__ = "Robert Nikutta <robert.nikutta@gmail.com>"
__version__ = '20160626'

import numpy as N
import warnings
from scipy import interpolate, ndimage
import itertools

# Convert RuntimeWarnings, e.g. division by zero in some array elements, to Exceptions
warnings.simplefilter('error', RuntimeWarning)


class NdimInterpolation:

    """N-dimensional interpolation on data hypercubes.

    Operates on image(index) coordinates. Multi-linear or cubic-spline
    (default).

    """

    def __init__(self,data,theta,order=1,mode='log'):

        """Initialize an interpolator object.

        Parameters
        ----------
        data : n-dim array or 1-d array
            Model database to be interpolated. Sampled on a
            rectilinear grid (it need not be regular!). 'data' is
            either an n-dimensional array (hypercube), or a
            1-dimensional array. If hypercube, each axis corresponds
            to one of the model parameters, and the index location
            along each axis grows with the parameter value. The last
            axis is the 'wavelength' axis. If 'data' is a 1-d array of
            values, it will be converted into the hypercube
            format. This means that the order of entries in the 1-d
            array must be as if constructed via looping over all axes,
            i.e.

            .. code:: python

                counter = 0
                for j0 in theta[0]:
                    for j1 in theta[1]:
                        for j2 in theta[2]:
                            ...
                            hypercube[j0,j1,j2,...] = onedarray[counter]
                            counter += 1

        theta : list
            List of 1-d arrays, each holding in ascending order the
            unique values for one of the model parameters. The last
            1-d array in theta is the wavelength array. Example: for
            the CLUMPY models of AGN tori (Nenkova et al. 2008)

              theta = [{i}, {tv}, {q}, {N0}, {sig}, {Y}, {wave}]

            where the {.} are 1-d arrays of unique model parameter
            values, e.g.

              {i} = array([0,10,20,30,40,50,60,70,80,90]) (degrees).

        order : int
            Order of interpolation spline to be used. order=1
            (default) is multi-linear interpolation, ``order=3`` is
            cubic-spline (quite a bit slower, and not necessarily
            better, especially for complicated n-dim
            functions. ``order=1`` is recommended.

        mode : str
            ``log`` is default, and will take log10(data) first, which
            severely improves the interpolation accuracy if the data
            span many orders of magnitude. This is of course only
            applicable if all entries in `data` are greater than
            0. Any string other that ``log`` will keep `data` as-is.

        Returns
        -------
        NdimInterpolation instance.

        Example
        -------
        General way to use ndiminterpolation

        .. code:: python

            # to be written

        """
        
        self.theta = theta   # list of lists of parameter values, unique, in correct order

        shape_ = [len(t) for t in self.theta]

        # determine if data is hypercube or list of 1d arrays
        if shape_ == data.shape:
            self.input = 'hypercube'
            self.data_hypercube = data
        else:
            self.input = 'linear'
            self.data_hypercube = data.reshape(shape_,order='F')

        # interpolation orders
        assert (order in (1,3)), "Interpolation spline order not supported! Must be 1 (linear) or 3 (cubic)."
        self.order = order

        # interpolate in log10 space?
        self.mode = mode
        if self.mode == 'log':
            try:
                self.data_hypercube = N.log10(self.data_hypercube)
            except RuntimeWarning:
                raise Exception, "For mode='log' all entries in 'data' must be > 0."

        # set up n 1-d linear interpolators for all n parameters in theta
        self.ips = []   # list of 1-d interpolator objects
        for t in self.theta:
            self.ips.append(interpolate.interp1d(t,N.linspace(0.,float(t.size-1.),t.size)))

        if self.order == 3:
            print "Evaluating cubic spline coefficients for subsequent use, please wait..."
            self.coeffs = ndimage.spline_filter(self.data_hypercube,order=3)
            print "Done."


    def get_coords(self,vec):
        
        """Construct a full 2D matrix of coordinates in pixel-space from a
           vector of coordinates in real space.
        
        Parameters
        ----------
        vec : tuple

            Tuple of lenght len(self.theta), with each element either
            a single value of theta_i (of the i-th parameter), or
            itself a tuple (of arbitrary length). If an element of vec
            is itself a tuple (of length m), then m*(n-1) combinations
            of all parameters will be added to the resulting 2D matrix
            of coordinates.

        Returns
        -------
        coorinates_pix : 2D array
            2D array of coordinates in pixel space, on which then the
            multi-dim interpolation can be performed.

            Overall, A = N.prod([len(vec_i) for vec_i in vec])
            coordinate sets, for B = len(self.theta) parameters, will
            be generated, i.e. the returned coordinate matrix has
            coordinates.shape = (A,B).

        shape_ : tuple
           The shape tuple to reshape coords_pix with to obtain a
           properly shaped interpolated array.

        Example
        -------
        .. code:: python

            self.parameters
                array(['a', 'b', 'c'])

            [t.size for t in self.theta]
                (3,5,2)

            self.theta
                [array(1.,2.,3.,), array(10.,15,18,24,26), array(100.,126)]

            # vector of real-space coordinate to interpolate self.data_hypercube on
            vec = (1.5,18.,110.)
        
            # compute pixel-space vector matrix, and shape of resulting array
            coords_pix, shape_ = self.get_coords(vec)


        Old example, rework it:

        .. code:: python

            vec = (0,1,2,3,(0,1,2),(3,4,5),6)
            vectup = [e if isinstance(e,tuple) else (e,) for e in vec]
               [(0,), (1,), (2,), (3,), (0, 1, 2), (3, 4, 5), (6,)]
            coords = N.array([e for e in itertools.product(*vectup)])  # in pixel space
               array([[0, 1, 2, 3, 0, 3, 6],
                      [0, 1, 2, 3, 0, 4, 6],
                      [0, 1, 2, 3, 0, 5, 6],
                      [0, 1, 2, 3, 1, 3, 6],
                      [0, 1, 2, 3, 1, 4, 6],
                      [0, 1, 2, 3, 1, 5, 6],
                      [0, 1, 2, 3, 2, 3, 6],
                      [0, 1, 2, 3, 2, 4, 6],
                      [0, 1, 2, 3, 2, 5, 6]])

        """

#        print "vec = ", vec

        vectup = [e if isinstance(e,tuple) else (e,) for e in vec]
#        print "vectup = ", vectup
        shape_ = [len(e) for e in vectup]

        coords_real = N.array([e for e in itertools.product(*vectup)])  # use ip(.) here
#        print "coords_pix, coords_pix.shape = ", coords_pix, coords_pix.shape
        
        columns = coords_real.T.tolist()
#        print "columns, len(columns) = ", columns, len(columns)
        
        coords_pix = N.array([ self.ips[j](columns[j]) for j in xrange(len(columns)) ])
#        print "coords_pix.shape = ", coords_pix.shape
#        print "self.data_hypercube.shape = ", self.data_hypercube.shape
        
        return coords_pix, shape_


    def __call__(self,vector):
        """Interpolate in N dimensions, using mapping to image coordinates."""

        if self.order == 1:
            coords, shape_ = self.get_coords(vector)
            aux = ndimage.map_coordinates(self.data_hypercube,coords,order=1)
            aux = aux.reshape(shape_)
# temporarily disabled order==3, b/c not yet tested
#        elif self.order == 3:
#            aux = ndimage.map_coordinates(self.coeffs,self.get_coords(vector,pivots=pivots),order=3,prefilter=False)

        if self.mode == 'log':
            aux = 10.**aux

        return aux
