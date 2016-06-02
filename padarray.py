"""Pad / unpad a 2D ragged array (variable-length array).

This is to work around Numpy's bug 2190, which for instance prevents
storing of ragged arrays with dtype='float64' in HDF5 files (using
h5py). Maybe it is useful in other scenarios.

Please see docstring of PadArray for usage.

"""

__author__ = "Robert Nikutta"
__version__ = "20150507"

import numpy as N


class PadArray:

    """Class for padding a ragged array.

    This is limited to the simple case of a list of 1-d arrays of
    variable lengths; convert them through padding into a 2-d array of
    shape (len(list),max([e.size for e in list])). The values of all
    1-d arrays are written left-bounded into the rows of the array,
    with their right-hand sides padded with padval (default: nan) up
    until the max length of all of the 1-d arrays.

    The other direction also works, i.e. providing a 2-d array and
    returning a list of 1-d arrays, with the specified padval removed
    from them.

    Parameters:
    -----------
    inp : seq
        List or tuple of 1-d arrays of numbers (or things that are
        float()-able.

    padval : {float, nan}
        Pad value to be used. Default is nan. The value can be
        anything that can be converted by float(), e.g. '3', or 1e7,
        etc.

    Example:
    --------
    # pad a list of 1-d arrays
    inp = [array([1,2]),array([1,2,3]),array([1,2,3,4,5])]
    pa = PadArray(inp)
    pa.unpad
      Out:   [array([1,2]),array([1,2,3]),array([1,2,3,4,5])]
    pa.pad
      Out:  array([[  1.,   2.,  nan,  nan,  nan],
                   [  1.,   2.,   3.,  nan,  nan],
                   [  1.,   2.,   3.,   4.,   5.]])

    # unpad a 2-d array
    inp = array( [1.,2.,-1.,-1.,-1.],
                 [1.,2.,3., -1.,-1.],
                 [1.,2.,3.,  4., 5.] )
    pa = PadArray(inp,padval=-1)
    pa.pad
      Out:  array( [1.,2.,-1.,-1.,-1.],
                   [1.,2.,3., -1.,-1.],
                   [1.,2.,3.,  4., 5.] )
    pa.unpad
      Out:  [array([ 1.,  2.]), array([ 1.,  2.,  3.]), array([ 1.,  2.,  3.,  4.,  5.])]

    """

    def __init__(self,inp,padval=N.nan):

        self.inp = inp
        self.padval = padval
        self._setup()
        self._convert()


    def _setup(self):

        try:
            self.padval = float(self.padval)
        except ValueError:
            raise Exception, "padval is not convertible to a floating-point number."

        if isinstance(self.inp,(list,tuple)):
            # TODO: check if all members of inp can be safely converted to 1-d numerical arrays
            self.inpmode = 'unpadded'
            self.unpad = self.inp
            self.nrow = len(self.unpad)
            self.ncol = max([e.size for e in self.unpad])

        elif isinstance(self.inp,(N.ndarray)):
            if self.inp.ndim == 2:
                self.inpmode = 'padded'
                self.pad = self.inp
                self.nrow, self.ncol = self.inp.shape
            else:
                raise Exception, "input appears to be an array, but is not 2-d."

        else:
            raise Exception, "input is neither a sequence of 1-d arrays, nor a 2-d array."


    def _convert(self):

        if self.inpmode == 'unpadded':
            self._pad()
        elif self.inpmode == 'padded':
            self._unpad()


    def _pad(self):

        self.pad = N.ones((self.nrow,self.ncol))*self.padval

        for j,e in enumerate(self.unpad):
            self.pad[j,:e.size] = e


    def _unpad(self):

        self.unpad = []

        for j in xrange(self.nrow):
            aux = self.pad[j,:]

            if N.isnan(self.padval):
                aux = aux[~N.isnan(aux)]
            else:
                aux = aux[aux!=self.padval]
                
            self.unpad.append(aux)
