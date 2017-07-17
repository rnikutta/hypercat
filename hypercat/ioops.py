from __future__ import print_function

__version__ = '20170717'   #yyymmdd
__author__ = 'Robert Nikutta <robert.nikutta@gmail.com>'

"""Utilities for handling I/O.

.. automodule:: ioops

"""

import astropy.io.fits as fits

class FitsFile:

    """Convenience class to read (OI)FITS files."""
    
    def __init__(self,f):

        """Init.

        Parameters
        ----------
        f : str
            Path to (oi)fits file.
        """
        
        self.oifitsfile = f
        self.hdulist = fits.open(self.oifitsfile,mode='readonly')
        

    def info(self):

        """Print hdulist.info()"""
        
        print(self.hdulist.info())
        

    def getheader(self,hdu=0,verbose=False):

        """Get header of hdu.

        Parameters
        ----------
        hdu : int
            hdu number to read. Default = 0

        verbose : bool
            If True, also print the header to STDOUT. Otherwise only
            return it.

        Returns
        -------
        header : instance
            Instance of :class:`astropy.io.fits.header.Header`

        """
        
        hdu = self.hdulist[hdu]
        header = hdu.header
        if verbose is True:
            print(header)
            
        return header


    def get(self,key,hdu=0,what='value'):

        """Get either a card, value, or comment, from the header of hdu.

        Parameters
        ----------
        key : str Name of key, e.g. ``'naxis'``. Capitalization is not
            important.

        hdu : int
            hdu number to read. Default = 0

        what : str
            What to return from the header of hdu. Possible values are
            ``'card'``, ``'value'``, and ``'comment'``.

            If ``'value'``, returns ``hdu.header[key]``.

            If ``'comment'``, returns ``hdu.comments[idx]``, where
            ``idx = header.index(key)``, i.e. the first index that
            matches ``keyword``.

            If ``'card'``, returns ``hdu.cards[idx]``, where ``idx =
            header.index(key)``, i.e. the first index that matches
            ``keyword``.

        Examples
        --------

        .. code-block:: python

           ff = FitsFile('oiftisfile.fits')

           ff.get('naxis',4,'value')   # return value of key 'naxis' from hdu=4
             2

           ff.get('naxis',4,)   # 'value' is the default
             2

           ff.get('naxis',4,'comment')  # return only the corresponding comment
             'Binary table'

           ff.get('naxis',4,'card')    # return the entire card = (key,value,comment) as a tuple
             ('NAXIS', 2, 'Binary table')

        """

        header = self.getheader(hdu=hdu)
        idx = header.index(key)

        if what is 'card':
            res = header.cards[idx]

        elif what is 'comment':
            res = header.comments[idx]
        
        elif what is 'value':
            res = header[key]

        return res
    

    def getdata(self,hdu=0,field=None):

        """Get the data from a hdu, possibly a field from a record array,

        Parameters
        ----------

        hdu : int
            hdu number to read. Default = 0


        field : str | None

            If ``None``, return the entire ``hdu.data``. If a string,
            assume that hdu.data is a record array, and fetch the
            record named by ``field``.

        Examples
        --------

        .. code-block:: python

           ff = FitsFile('oiftisfile.fits')

           ff.getdata(hdu=4)  # return all hdu.data

           ff.getdata(4,'vcoord')  # Get v-coordinates (from u,v plane)
              array([-41.04285049,   1.51398432,  13.22854233,  35.11543655,
                      30.22608757,   1.65713799, -31.00023651, -40.52364731,
                      42.68415833,  33.51140594,  15.40285206,   3.22378659,
                      -3.13334179,  -8.67062855,  15.3754034 ,  14.42059231,
                       9.64568615,   6.66133642,   3.32804942,  -2.93655539,
                      -8.03857231,  -8.59758091,  15.26423931,  13.05595779,
                      10.33599758,   6.71977472,   0.81495309,  -2.18943191])

           ff.getdata(2,'raep0'), ff.getdata(2,'decep0')  # get RA & DEC
              (array([ 213.292174]), array([-65.33936]))
        """

        hdu = self.hdulist[hdu]

        if field is None:
            res = hdu.data
        else:
            res = hdu.data[field]

        return res
    
