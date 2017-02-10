__version__ = '20170209'   #yyymmdd
__author__ = 'Robert Nikutta <robert.nikutta@gmail.com>'

"""General helper func for hypercat.

.. automodule:: utils
"""

import numpy as N


def arrayify(arg,n=None,shape=None,direction='x'):

    """Repeat 'arg' 'n' times, put that sequence in array with shape 'shape'.

    If arg is already a list-like sequence, ignore n.
    If 'shape' is a 2-tuple, reshape sequence to an array with shape 'shape'.
    If len(sequence)>array.size, only put the first array.size elements of sequence into the array,
    If len(sequence)<=array.size, pad the remaining elements in array with None.

    Parameters
    ----------
    arg : singlet, or list-like sequence
        If not list or tuple, arg is a single-element object (can be
        anything) and will be repeated either n times, or
        N.prod(shape) times.

    n : int|None

    shape : 2-tuple|None

    direction : str
        If 'x' (default), elements in resulting 2d array are counted
        along rows first, then columns. If 'y', the other way round.

    Examples
    --------
    .. code-block:: python
    
        # generate 1d sequence
        arrayify('foo',n=10,shape=None)
        --> ['foo', 'foo', 'foo', 'foo', 'foo', 'foo']

        # and reshape to 2d
        arrayify('foo',n=10,shape=(2,3))
        --> array([['foo', 'foo', 'foo'],
                   ['foo', 'foo', 'foo']], dtype=object)
        
        # embed sequence in larger 2d array (pad tail with None)
        arrayify('foo',n=10,shape=(2,4))
        --> array([['foo', 'foo', 'foo', 'foo'],
                   ['foo', 'foo', None,   None]], dtype=object)

        # place sequence in smaller 2d array (only use first elements)
        arrayify('foo',n=6,shape=(2,2))
        --> array([['foo', 'foo'],
                   ['foo', 'foo']], dtype=object)

        # provide a list-like sequence (no effect)
        arrayify(['foo','bar','baz'],n=None,shape=None)
        --> ['foo', 'bar', 'baz']

        # provide a list-like sequence, and n (no effect, n ignored)
        arrayify(['foo','bar','baz'],n=6,shape=None)
        --> ['foo', 'bar', 'baz']

        # provide a list-like sequence, and embed in larger array
        arrayify(['foo','bar','baz'],n=6,shape=(2,4))
        --> array([['foo', 'bar', 'baz', None],
                   [None, None, None, None]], dtype=object)        

        # provide a list-like sequence, and embed in smaller array
        arrayify(['foo','bar','baz'],n=6,shape=(1,2))
        --> array([['foo', 'bar']], dtype=object)

    """

    lili = (list,tuple)  # list-like

    if not isinstance(arg,lili):
        if isinstance(n,int):
            seq = [arg]*n
        elif shape is not None:
            seq = [arg]*N.prod(shape)
        else:
            seq = N.empty((1,1),dtype=object)
            seq[0,0] = arg
    else:
        n = len(arg)
        seq = arg

    if isinstance(shape,lili) and len(shape)==2:
        seq2d = N.empty(shape,dtype=object)
        idx = min(n,seq2d.size)
        seq2d.ravel()[:idx] = seq[:idx]
        seq = seq2d

        if direction == 'y':
            seq = seq.reshape(shape[::-1]).T

    return seq


def mirror_axis(cube,axis=-2):

    """Mirror one axis of a cube.

    Parameters
    ----------
    cube : array
        Expected shape of cube: (...,nx,...).

    axis : int
        axis index to be mirrored. Default: -2. Mirroring assumes a
        central columns of elements. See 'Returns' section.

    Returns
    -------
    newcube : array
        Array with 'axis' dimension mirrored, such that
          newcube.shape = (...,2*nx-1,...)
        i.e. newcube.shape[axis] always odd.

    Examples
    --------
    .. code-block:: python

        c = N.arange(9).reshape((3,3))
        c
        --> array([[0, 1, 2],
                   [3, 4, 5],
                   [6, 7, 8]])

        mirror_axis(c,axis=0)
        --> array([[6, 7, 8],
                   [3, 4, 5],
                   [0, 1, 2],
                   [3, 4, 5],
                   [6, 7, 8]])

        mirror_axis(c,axis=1)
        --> array([[2, 1, 0, 1, 2],
                   [5, 4, 3, 4, 5],
                   [8, 7, 6, 7, 8]])

    """
        
    ndim = cube.ndim
    npix = cube.shape[axis]
    allpads = [(0,0)]*ndim
    allpads[axis] = (npix-1,0)
    
    newcube = N.pad(cube,tuple(allpads),'reflect')

    return newcube
    

def seq2str(seq,jstr=','):
    
    """Join a sequence ``seq`` of string-ifiable elements with a string
    ``jstr``.

    Example
    -------
    .. code:: python

        seq2str(('a','b','c'),' - ')
          'a - b - c'

        seq2str([1,2,3],':')
          '1:2:3'

    """
    
    return jstr.join([str(_) for _ in seq])