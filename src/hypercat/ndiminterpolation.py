"""N-dimensional interpolation on data hypercubes.
"""

__author__ = "Robert Nikutta <robert.nikutta@gmail.com>"
__version__ = '20200913' #yyyymmdd

#TODO: update doc strings

import numpy as np
from numpy import ma
import warnings
from scipy import interpolate, ndimage
import itertools
from copy import copy

# Convert RuntimeWarnings, e.g. division by zero in some array elements, to Exceptions
warnings.simplefilter('error', RuntimeWarning)


class NdimInterpolation:

    """N-dimensional interpolation on data hypercubes.

    Operates on image(index) coordinates. Multi-linear (default) or
    cubic-spline (currently deactivated; needs more testing).
    """

    def __init__(self,data,theta,order=1,mode='log'):

        """Initialize an interpolator object.

        Parameters
        ----------
        data : n-dim array or 1-d array
            Datacube to be interpolated. Sampled on a rectilinear grid
            (it need not be regular!). 'data' is either an
            n-dimensional array (hypercube), or a 1-dimensional
            array. If hypercube, each axis corresponds to one of the
            model parameters, and the index location along each axis
            grows with the parameter value (the parameter values are
            given in `theta`). If 'data' is a 1-d array of values, it
            will be converted into the hypercube format. This means
            that the order of entries in the 1-d array must be as if
            constructed via looping over all axes, i.e.

            .. code:: python

                counter = 0
                for j0 in theta[0]:
                    for j1 in theta[1]:
                        for j2 in theta[2]:
                            ...
                            hypercube[j0,j1,j2,...] = onedarray[counter]
                            counter += 1

        theta : list
            List of lists, each holding in ascending order the unique
            values for one of the axes in `data` hypercube. Example:
            for the CLUMPY models of AGN tori (Nenkova et al. 2008)

              theta = [{i}, {tv}, {q}, {N0}, {sig}, {Y}, {wave}]

            where the {.} are 1-d arrays of unique model parameter
            values, e.g.

              {i} = array([0,10,20,30,40,50,60,70,80,90]) (degrees).

        order : int
            Order of interpolation spline to be used. ``order=1``
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

        self.theta = copy(theta) # list of lists of parameter values, unique, in correct order
        
        if not isinstance(self.theta,(list,tuple)):
            self.theta = [self.theta]
        
        shape_ = tuple([len(t) for t in self.theta])

        # determine if data is hypercube or list of 1d arrays
        if shape_ == data.shape:
            self.data_hypercube = data
        else:
            raise Exception("'theta' not compatible with the shape of 'data'.")

        # interpolation orders
        if order in (1,3):
            self.order = order
        else:
            raise Exception("Interpolation spline order not supported! Must be 1 (linear) or 3 (cubic).")

        # interpolate in log10 space?
        self.mode = mode

        # take log10 of 'data' ('y' values)
        if self.mode in ('log','loglog'):

            self.data_hypercube = ma.masked_less_equal(self.data_hypercube, 0.) # masking zeros and below
            
            try:
                self.data_hypercube = ma.log10(self.data_hypercube)
            except RuntimeWarning:
                raise Exception("For mode='log' all entries in 'data' must be > 0.")

#        # take log10 of 'theta' ('x' values)
#        if self.mode == 'loglog':
#            for jt,t in enumerate(self.theta):
#                try:
#                    self.theta[jt] = np.log10(t)
#                except:
#                    raise # Exception

        # set up n 1-d linear interpolators for all n parameters in theta
        self.ips = [] # list of 1-d interpolator objects
        for t in self.theta:
            self.ips.append(interpolate.interp1d(t,np.linspace(0.,float(t.size-1.),t.size)))

# Not yet tested; potentially too slow to be practical
#        if self.order == 3:
#            print("Evaluating cubic spline coefficients for subsequent use, please wait...")
#            self.coeffs = ndimage.spline_filter(self.data_hypercube,order=3)
#            print("Done.")


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

            Overall, A = np.prod([len(vec_i) for vec_i in vec])
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
            coords = np.array([e for e in itertools.product(*vectup)])  # in pixel space
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

        vectup = [e if isinstance(e,tuple) else (e,) for e in vec] # make tuple of vectors

        shape_ = [len(e) for e in vectup] # tuple shape

        # create a fleshed-out mesh of (multi-dim) locations to interpolate `data` at
        coords_real = np.array([e for e in itertools.product(*vectup)])
        
        columns = coords_real.T.tolist() # transpose

        # convert physical coordinate values to (fractional) pixel-space
        coords_pix = np.array([ self.ips[j](columns[j]) for j in range(len(columns)) ])
        
        return coords_pix, shape_


    def serialize_vector(self,vector):

        vec = list(vector)

        # sub-vectors can be arrays or lists; convert to tuples
        for j,v in enumerate(vec):
            if isinstance(v,(list,tuple)):
                vec[j] = tuple(v)
            elif isinstance(v,np.ndarray):
                vec[j] = tuple(v.squeeze()) # to allow for 1-d arrays embedded in higher-dims
            else:
                vec[j] = v
                
        vec = tuple(vec)
        
        return vec

    
    def __call__(self,vector,mask_op=ma.masked_equal,mask_thresh=0.):
        """Interpolate in N dimensions, using mapping to image coordinates."""

        if not isinstance(vector,(list,tuple)):
            vector = [vector]
            
#        if self.mode == 'loglog':
#            vector = [np.log10(e) for e in vector]
            
        vec = self.serialize_vector(vector)
        
        coords, shape_ = self.get_coords(vec)
        if self.order == 1:
            aux = ndimage.map_coordinates(self.data_hypercube,coords,order=1)
            aux = aux.reshape(shape_)
# Not yet tested; potentially too slow to be practical
#        elif self.order == 3:
##            aux = ndimage.map_coordinates(self.coeffs,self.get_coords(vector,pivots=pivots),order=3,prefilter=False)
#            aux = ndimage.map_coordinates(self.data_hypercube,coords,order=3)
#            aux = aux.reshape(shape_)

        aux = aux.squeeze() # remove superflous length-one dimensions from result array

        
        
        if self.mode in  ('log','loglog'):
            #print("In mask_op branch")
            aux = mask_op(aux,mask_thresh)
            mask = aux.mask
            aux = 10.**aux
            aux[mask] = mask_thresh

        return aux
