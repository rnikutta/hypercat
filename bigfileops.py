__version__ = '20170120'   #yyymmdd
__author__ = 'Robert Nikutta <robert.nikutta@gmail.com>'

"""Utilities for handling large hypercubes in hdf5 files.

.. automodule:: bigfileops
"""

import logging
import json
import urwid
import urwid.curses_display
import numpy as N
import h5py

def memmap_hdf5_dataset(hdf5file,dsetpath):
    
    # assert dataset properties, determine dtype, shape, and offset
    with h5py.File(hdf5file,'r') as f:
        ds = f[dsetpath]

        # We get the dataset address in the HDF5 fiel.
        offset = ds.id.get_offset()
        
        # We ensure we have a non-compressed contiguous array.
        if ds.chunks is not None:
            raise ValueError("mem-mapping only works for non-chunked hdf5 dastasets. Datset '%s' appears to have chunks." % dsetpath)
        
        if ds.compression is not None:
            raise ValueError("mem-mapping only works for non-compressed hdf5 dastasets. Datset '%s' appears to have compression ON." % dsetpath)
        
        if offset <= 0:
            raise ValueError("Invalid offset found for dataset '%s'. Offset must be an integer > 0" % dsetpath)
        
        dtype = ds.dtype
        shape = ds.shape

    # provide memory-mapped array object (this does not occupy RAM until the array (or part of it) is explicitly read
    dsmemmap = N.memmap(hdf5file, mode='r', shape=shape, offset=offset, dtype=dtype)
    
    return dsmemmap


def get_hyperslab_via_mesh(dset,idxlist):

    mesh = N.ix_(*idxlist)
    arr = dset[mesh]

    return arr


def getIndexLists(theta,paramnames,initsize=None,wordsize=4.,omit=()):

    if initsize is not None:
        currentsize = initsize
    else:
        shape_ = N.array([len(_) for _ in theta])
        currentsize = N.prod(shape_) * wordsize
    
    # select a sub-hypercube
    t = []
    i = []

    for j,parname in enumerate(paramnames):
        idxes = []

        if parname in omit:
            t.append(theta[j])
            i.append(range(len(theta[j])))
        else:
            while len(idxes) == 0:
                CLS = CheckListSelector(parname,theta[j],initsize=currentsize)
                idxes = CLS.idxes
                    
            thnew, currentsize = CLS.theta[idxes], CLS.currentsize
            t.append(thnew)
            i.append(idxes)

    return t, i, currentsize  # currentsize in bytes


class CheckListSelector:

    def __init__(self,parname,theta,initsize=None):

        self.theta = theta
        self.parname = parname
        self.initsize = initsize

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
                                 "Select values for parameter %s" % parname),
                      'header')

        strings = [str(e) for e in self.theta]
        len_ = int(N.max([len(e) for e in strings]))
        self.cells = [urwid.AttrWrap(urwid.CheckBox(e,state=True),'buttn','buttnf') for e in strings]

        self.pad = urwid.Padding( urwid.GridFlow(self.cells,4+int(len_),2,0,'left'), ('fixed left',4), ('fixed right',3) )
        self.listbox = urwid.ListBox([self.pad])
        self.view = urwid.Frame( urwid.AttrWrap(self.listbox, 'body'), header=self.HEADER)
        self.update_footer()
        
        self.ui.register_palette(self.palette)
        self.ui.run_wrapper(self.run)

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
        self.bidxes = N.array([e.get_state() for e in self.cells])
        self.idxes = N.arange(self.theta.size)[self.bidxes].tolist()
        nselected = N.argwhere(self.bidxes).size

        text = "Selected %d/%d." % (nselected,self.bidxes.size)
        
        if self.initsize is not None:
            self.currentsize = self.initsize * nselected / float(self.bidxes.size)
            prefix, suffix = get_bytes_human(self.currentsize)
            text += " (Total size: %.2f %s)" % (prefix,suffix)
            
        if nselected == 0:
            text += " Select at least one value."
            
        self.view.footer = urwid.AttrWrap( urwid.Text(text), 'header')
#        self.draw_canvas()
        

def storejson(jsonfile,d):

    """Store the objects from a dictionary to a human-readable json file.

    Parameters
    ----------
    jsonfile : str
        Path to json file to be written.

    d : dict
        Dictionary to be stored in jsonfile.

    """
    
    with open(jsonfile,'w') as f:
        json.dump(d,f)

    logging.info("Saved objects in file %s." % jsonfile)
    

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
    
    logging.info("Loading objects from file %s." % jsonfile)

    with open(jsonfile,'r') as f:
        d = json.load(f)

    return d


def get_bytesize(lol,wordsize=4.):

    """lol : list of lists"""
    
    shape_ = N.array([len(_) for _ in lol])
    bytesize = N.prod(shape_) * wordsize

    return bytesize


def get_bytes_human(nbytes):

    """Converts int (assuming it's a bytes count) to human-readable format.

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
        print prefix, suffix
        1023.0 bytes

        prefix, suffix = get_bytes_human(1024)
        print prefix, suffix
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
