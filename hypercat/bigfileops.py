from __future__ import print_function

__version__ = '20180316'   #yyymmdd
__author__ = 'Robert Nikutta <robert.nikutta@gmail.com>'

"""Utilities for handling large hypercubes in hdf5 files.

   Some functionality:
     - mem-mapping
     - hyper-slicing
     - interactive hyper-slab selection
     - saving selections as index lists of lists to json files
     - saving sub-cubes to hdf5 files

.. automodule:: bigfileops

"""

import logging
import json
import urwid, urwid.curses_display
import numpy as np
import h5py


def storeCubeToHdf5(cube,hdffile,groupname=None):

    """Store :class:`hypercat.ModelCube` instance to separate hdf5 file.

    Maps these hdf5 file elemens to the properties of a
    :class:`hypercat.ModelCube` instance: (where
    ``h = h5py.File(hdffile)`` and ``g = h[groupname]``)

    .. code-block:: text

       / level (root)
          h['Nhypercubes'] --> 1 if hdffile is new; +1 for each newly added group
          h['hypercubenames'] --> dataset/list of added groups, extended each time via .resize()

       /group level
          g['hypercube'] --> cube.data
          g['hypercubeshape'] --> cube.data.shape
          g['paramnames'] --> cube.paramnames
          g['theta'] --> cube.theta (but PadArray'd)

    Opens ``hdffile`` in append mode if it already exists. Creates it
    otherwise.

    Fails if groupname already exists in hdffile. Creates group otherwise.

    Parameters
    ----------
    cube : instance
        Instance of :class:`hypercat.ModelCube` class.
    
    hdffile : str
        Path to hdf5 file where the cube and all relevant properties
        will be stored. If hdffile does not yet exists on disk, it
        will be created

    groupname : str | None
        Name of group in hdffile under which this cube will be
        stored. If a group of this name already exists in ``hdffile``,
        will raise an Exception. If ``None``, will use
        ``cube.groupname``.

    """

    # open or create hdf5 file
    h = h5py.File(hdffile,'a')  # open in read/write mode if file exists, create otherwise

    # add group, or fail if already exists
    if groupname is None:
        groupname = cube.groupname

    g = h.create_group(groupname)
    
    # add members to group, or fail if one already exists
    mapping = {'hypercube':cube.data,
               'hypercubeshape':cube.data.shape,
               'paramnames':cube.paramnames,
               'theta': cube.theta}

    for name,data in mapping.items():
        print("name, type(data): ", name, type(data))

        if isinstance(data[0],np.ndarray) and isragged(data):
            dflt = h5py.special_dtype(vlen=np.dtype('float64'))
            ds = g.create_dataset(name, (len(data),), dtype=dflt)
            for j,v in enumerate(data):
#                print("j, v, type(v) = ", j, v, type(v))
                ds[j] = v

        else:
            if isinstance(data,list) and isinstance(data[0],str):  # CAUTION: test it more
                data = [e.encode() for e in data]

            
            print("type(data): ", type(data))
            ds = g.create_dataset(name,data=data)

    # if all went well, increment h['Nhypercubes'] by one...
    try:
        ds = h['Nhypercubes']
    except KeyError:
        ds = h.create_dataset('Nhypercubes',data=1)
    else:
        ds[()] += 1


    # ... and append the group name to the ``hypercubes`` dataset (a 1-d array of strings)
    try:
        ds = h['hypercubenames']
    except KeyError:
        ds = h.create_dataset('hypercubenames',data=[groupname],maxshape=(None,))
    else:
        n = ds.shape[0]
        ds.resize((n+1,))
        ds[n] = groupname

    h.close()


def isragged(arr):

    """Test if an array is ragged (i.e. unequal-length records).

    Parameters
    ----------
    arr : array

    Returns
    -------
    ragged : bool
        Returns True of not all entries in arr are of same length
        (i.e. arr is indeed ragged). False otherwise.

    """
    
    ragged = not (np.unique([len(r) for r in arr]).size == 1)
    
    return ragged


def memmap_hdf5_dataset(hdf5file,dsetpath):

    """Memory-map a dataset in a hdf5 file.

    The memory mapping provides pointers from RAM to the location of
    bits on-disk. Before accessing the data, nothing is being copied
    to RAM.

    Parameters
    ----------
    hdffile : str
        Path to the hdf5 file containing the dataset.

    dsetpath : str
        The hdf5 qualifying path to the desired dataset. It can
        contain groups and subgroups. The end point of ``dsetpath`` is
        the dataset.

    Returns
    -------
    dsmemap : memmap
        Memory-mapped object (as ``numpy.core.memmap.memmap``),
        representing a dataset on disk.

    Examples
    --------

    .. code-block:: python

       memmap = memmap_hdf5_dataset('hypercat_20170109.hdf5','imgdata/hypercube')
       memmap.shape
         (5, 11, 4, 12, 5, 7, 19, 221, 441)  # sig,i,Y,N,q,tv,wave,x,y

    """
    
    # assert dataset properties, determine dtype, shape, and offset
    with h5py.File(hdf5file,'r') as f:
        ds = f[dsetpath]

        # We get the dataset address in the HDF5 fiel.
        offset = ds.id.get_offset()
        
        # We ensure we have a non-compressed contiguous array.
        if ds.chunks is not None:
            raise ValueError("mem-mapping only works for non-chunked hdf5 dastasets. Datset '{:s}' appears to have chunks.".format(dsetpath))
        
        if ds.compression is not None:
            raise ValueError("mem-mapping only works for non-compressed hdf5 dastasets. Datset '{:s}' appears to have compression ON.".format(dsetpath))
        
        if offset <= 0:
            raise ValueError("Invalid offset found for dataset '{:s}'. Offset must be an integer > 0".format(dsetpath))
        
        dtype = ds.dtype
        shape = ds.shape

    # provide memory-mapped array object (this does not occupy RAM until the array (or part of it) is explicitly read
    dsmemmap = np.memmap(hdf5file, mode='r', shape=shape, offset=offset, dtype=dtype)
    
    return dsmemmap


def get_hyperslab_via_mesh(dset,idxlist):

    """Select from n-dim array via open mesh of indices.

    Parameters
    ----------
    dset : n-dim array
        N-dimensional array or mem-mapped object, from which certain
        indexed vertices will be selected via ``idxlist``.

    idxlist : seq
       List of lists, or list of tuples, or tuple of lists or tuples,
       etc. For every axis in ``dset``, each sub-sequence lists the
       index positions to be selected along the current axis from
       ``dset``. An open mesh of n-dim index positions will then be
       contstructed from ``idxlist``, and the selection from ``dset``
       performed.

    Returns
    -------
    sel : n-dim array
        Sub-array selected from ``dst`` according to ``idxlist``.

    Examples
    --------

    .. code-block:: python

       import numpy as N
       A = np.arange(3*4*5).reshape((3,4,5))
         array([[[ 0,  1,  2,  3,  4],
                 [ 5,  6,  7,  8,  9],
                 [10, 11, 12, 13, 14],
                 [15, 16, 17, 18, 19]],

                 [20, 21, 22, 23, 24],
                 [25, 26, 27, 28, 29],
                 [30, 31, 32, 33, 34],
                 [35, 36, 37, 38, 39]],

                 [40, 41, 42, 43, 44],
                 [45, 46, 47, 48, 49],
                 [50, 51, 52, 53, 54],
                 [55, 56, 57, 58, 59]]])

       idxlist = [[1],[2,3],[0,3]]
       selarr = get_hyperslab_via_mesh(A,idxlist)
       B = bigfileops.get_hyperslab_via_mesh(A,idxlist)
       B
         array([[[30, 33],
                 [35, 38]]])

    """
    
    mesh = np.ix_(*idxlist)
    arr = dset[mesh]

    return arr


def getIndexLists(theta,paramnames,initsize=None,wordsize=4,omit=()):

    """Interactively select values from a number of parameter value lists.

    For each parameter in ``paramnames``, select interactively values
    from the corresponding list of parameter values in ``theta``,
    using class CheckListSelector (see also docstring there).

    Parameters
    ----------
    theta : list
        List of lists, each sub-list being a sequence of values to
        select. Every element will be printed next to a check box that
        can be selected or unselected.

    paramnames : list
        List of strings, each being the name given to the corresponding
        sub-list in ``theta``.

    initsize : float | None
        If not Note, it is the size (e.g. in GB, or simply in counts)
        of the pre-selection full cube. This size will be numerically
        reduced for every element removed during the selection. If
        None, the size will be computed as np.prod(shape) * wordsize,
        where ``shape`` are all lengths of all sub-lists in
        `'theta``. For ``wordsize`` see below.

    wordsize : int
        The size in bytes of a single number. Will be used to compute
        the size of the full cube in case no ``initsize`` is
        provided. Default: wordsize=4 (i.e. float32).

    omit : tuple
        Tuple of strings, each a parameter name in ``paramnames``,
        which should be omitted from being presented to the user for
        interactive selection. I.e., the parameter values for all
        parameter is ``omit`` will be returned unchanged.

    Returns
    -------
    t : list
        List of 1-d arrays, each containing the seleced values made
        interactively from every sub-sequence in ``theta``.

    i : list
        List of list of indices, each being the indices that select
        t[j] for j a sub-sequence in ``theta``.

    currentsize : float
        The (possibly) reduced size, computed from ``initsize`` (in
        whatever units) by making the interactive selections.

    Examples
    --------

    .. code-block:: python

       theta = [[0,1,2,3],[11.,12,13,14,15],[0.1,0.4,0.6]]
       paramnames = ['a','b','c']
       t,i,cs = getIndexLists(theta,paramnames)
       # ...make interactive selections on the screen...
       print(theta)
         [[0, 1, 2, 3], [11.0, 12, 13, 14, 15], [0.1, 0.4, 0.6]]
       print(i)
         [[1, 2, 3], [0, 1, 4], [0, 1, 2]]
       print(t )
         [array([1, 2, 3]), array([ 11.,  12.,  15.]), array([ 0.1,  0.4,  0.6])]
       print(cs)
         108.0  # currentsize; before selections it was 60 elements x 4 bytes 240 bytes

    """
    
    if initsize is not None:
        currentsize = initsize
    else:
        shape_ = np.array([len(_) for _ in theta])
        currentsize = np.prod(shape_) * wordsize
    
    # select a sub-hypercube
    t = []
    i = []

    for j,parname in enumerate(paramnames):
        idxes = []

        if parname in omit:
            t.append(theta[j])
#            i.append(range(len(theta[j])))
            i.append(list(range(len(theta[j]))))
        else:
            while len(idxes) == 0:
                CLS = CheckListSelector(parname,theta[j],initsize=currentsize)
                idxes = CLS.idxes
                    
            thnew, currentsize = CLS.theta[idxes], CLS.currentsize
            t.append(thnew)
            i.append(idxes)

    return t, i, currentsize  # currentsize in bytes


class CheckListSelector:

    """Interactively select values from a list."""

    def __init__(self,parname,theta,initsize=None):

        """Parameters
        ----------
        parname : str
            Name of the list in ``theta``. It we be displayed to the
            user in the selection dialog.

        theta : 1-d array
            Sequence of values. If not an array, will be turned into
            an array. The values in this sequence will be presented to
            the user for interactive selection. Every value will have
            a checkbox.

        initsize : float | None
            If not None, it is the size (e.g. in GB, or simply in
            counts) associated with the full size of 'something',
            pre-selection. This size will be numerically reduced for
            every element removed during the selection. If None, no
            size computation will be performed.

        Returns
        -------
        Nothing. But after running the interactive selection, see
        member self.idxes, it contains the selected indices.

        """

        self.theta = theta
        if not isinstance(self.theta,np.ndarray):
            self.theta = np.array(self.theta)
        
        self.parname = parname
        self.initsize = initsize

        # set colors for bg, fg, and other elements
#        self.palette = [ ('header', 'white,bold', 'dark red'),
#                         ('body', 'white,bold', 'light gray'),
#                         ('buttn', 'black', 'light gray'),
#                         ('buttnf', 'light green,bold', 'light gray'),
#                         ('selected', 'white', 'dark blue') ]
        self.palette = [ ('header', 'white,bold', 'dark blue'),
                         ('body', 'white,bold', 'dark cyan'),
                         ('buttn', 'black', 'dark cyan'),
                         ('buttnf', 'white,bold', 'dark cyan'),
                         ('selected', 'white', 'dark blue') ]

        self.ui = urwid.curses_display.Screen()

        self.HEADER = urwid.AttrWrap(
                      urwid.Text("Use the keyboard arrows to move between checkboxes.\n" +\
                                 "Hit SPACE to select/deselect. Hit ENTER when done.\n\n" +\
                                 "Select values for parameter {:s}".format(parname)),
                      'header')

        strings = [str(e) for e in self.theta]
        len_ = int(np.max([len(e) for e in strings]))
        self.cells = [urwid.AttrWrap(urwid.CheckBox(e,state=True),'buttn','buttnf') for e in strings]

        self.pad = urwid.Padding( urwid.GridFlow(self.cells,4+int(len_),2,0,'left'), ('fixed left',4), ('fixed right',3) )
        self.listbox = urwid.ListBox([self.pad])
        self.view = urwid.Frame( urwid.AttrWrap(self.listbox, 'body'), header=self.HEADER)
        self.update_footer()
        
        self.ui.register_palette(self.palette)
        self.ui.run_wrapper(self.run)

    # main event loop
    def run(self):
        while 1:
            self.draw_canvas()
            keys = None
            
            while not keys: 
                keys = self.ui.get_input()
                
            for k in keys:
                if k == 'enter':
                    return
                else:
                    self.view.keypress( self.size, k )
                    self.draw_canvas()
                    self.update_footer()

    def draw_canvas(self):
        self.width = self.ui.get_cols_rows()[0]
        self.height = self.pad.rows((self.width,)) + 5
        self.size = (self.width,self.height)
        self.canvas = self.view.render( self.size, focus=1 )
        self.ui.clear()
        self.ui.draw_screen( self.size, self.canvas )

    def update_footer(self):
        self.bidxes = np.array([e.get_state() for e in self.cells])
        self.idxes = np.arange(self.theta.size)[self.bidxes].tolist()
        nselected = np.argwhere(self.bidxes).size

        text = "Selected {:d}/{:d}.".format(nselected,self.bidxes.size)
        
        if self.initsize is not None:
            self.currentsize = self.initsize * nselected / float(self.bidxes.size)
            prefix, suffix = get_bytes_human(self.currentsize)
            text += " (Total size: {:.2f} {:s})".format(prefix,suffix)
            
        if nselected == 0:
            text += " Select at least one value."
            
        self.view.footer = urwid.AttrWrap( urwid.Text(text), 'header')
        

def storejson(jsonfile,d):

    """Store the objects from a dictionary to a human-readable json file.

    Parameters
    ----------
    jsonfile : str
        Path to json file to be written.

    d : dict
        Dictionary to be stored in jsonfile.

    """

    print("In storejson: d = ", d)
    
    with open(jsonfile,'w') as f:
        json.dump(d,f)

    logging.info("Saved objects in file {:s}.".format(jsonfile))
    

def loadjson(jsonfile):

    """Load content from json file as dictionary.

    Parameters
    ----------
    jsonfile : str
        Path to json file to be read.

    Returns
    -------
    d : dict
        Dictionary of the content stored in the json file.
    """
    
    logging.info("Loading objects from file {:s}.".format(jsonfile))

    with open(jsonfile,'r') as f:
        d = json.load(f)

    return d


def get_bytesize(lol,wordsize=4):

    """Compute total number of elements in a list of lists, times wordsize in bytes

    Parameters
    ----------
    lol : list 
        List of of lists. Total number of elements in ``lol`` will be
        computed.

    wordsize : int
        The size in bytes of a single element. Will be used to compute
        the total size of ``lol``. Default: ``wordsize=4`` (i.e. float32).

    Returns
    -------
    bytesize : int
        Total number of bytes in ``lol``.

    """
    
    shape_ = [len(_) for _ in lol]
    bytesize = np.prod(shape_) * wordsize

    return bytesize


def get_bytes_human(nbytes):

    """Converts nbytes (assuming it is a bytes count) to human-readable format.

    Also works with negative nbytes, and handles larger-than-largest
    numbers gracefully.

    Parameters
    ----------
    nbytes : int

    Returns
    -------
    prefix : float
        Mantissa (or significand) of the human-readable bytes count.

    suffix : str
        The human-readable string representing the exponent power of
        the the human-readable bytes count. In log steps of 3.

    Examples
    --------

    .. code:: python

        prefix, suffix = get_bytes_human(1023)
        print(prefix,suffix)
        1023.0 bytes

        prefix, suffix = get_bytes_human(1024)
        print(prefix,suffix)
        1.0 KB

    """

    from math import log
    from numpy import sign
    
    suffixes = ('bytes','KB','MB','GB','TB','PB','EB','ZB','YB')
    maxorder = len(suffixes)-1

    sig = sign(nbytes)
    
    if nbytes != 0:
        order = int(log(abs(nbytes),2)/10.)
    else:
        order = 0

    order = min((order,maxorder))

    prefix = abs(nbytes)/(1024.**order)
    suffix = suffixes[order]
    
    return sig*prefix, suffix
