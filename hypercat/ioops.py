from __future__ import print_function

__version__ = '20180207' #yyymmdd
__author__ = 'Robert Nikutta <robert.nikutta@gmail.com>'

"""Utilities for handling I/O.

.. automodule:: ioops

"""

import numpy as np
import astropy.io.fits as fits
from astropy import wcs
from astropy.coordinates import name_resolve
import logging
import datetime

def save2fits(image,fitsfile,usewcs=True):

    """Save instance of class with ``.data`` member to a new ImageHDU in FITS file.

    Some other properties will also be saved, e.g. a WCS (if present),
    telescope and instrument names, etc.

    Subsequent calls of this function with new image instances will
    all cause new ImageHDUs to appended to ``fitsfile``.

    Parameters
    ----------
    image : instance | seq of instances
        Instance of a class, which has a ``.data`` member (and
        optionally ``.wcs``, see below). For example, hypercat's
        classes :class:`imageops.Image` and :class:`psf.PSF` qualify
        (sky/obs and psf objects).

       ``image`` can also be a list-like sequence of instances of
       :class:`imageops.Image` and :class:`psf.PSF`. In that case, all
       instances will be saved to the ``fitsfile`` in sequential HDUs.

    fitsfile : str
        Path to FITS file. If the file does not yet exist, it will be
        created, and a new empty PrimaryHDU will be created. If
        ``fitsfile`` does already exist, it will be opened in append
        mode. In both cases, ``image.data`` will then be written to a
        new ImageHDU, which will be appended to the FITS file.

    usewcs : bool
        If ``True`` (default) and if image has a member ``.wcs``
        (which is an instance of :class:`astropy.wcs.wcs.WCS`), the
        header of the ImageHDU to be created will be constructed with
        a proper world coordinate system. Otherwise, a minimal header
        will be constructed (without a WCS).

    Returns
    -------
    Nothing.

    Examples
    --------
    Assuming you have several instances with ``.data`` members (and
    optionally ``.wcs``), e.g. ``sky``, ``obs``, ``psf``:

    .. code-block:: python

       import ioops
       ioops.save2fits(sky,'myfitsfile.fits') # new file created, new ImageHDU appended
       ioops.save2fits(obs,'myfitsfile.fits') # new ImageHDU appended
       ioops.save2fits(psf,'myfitsfile.fits') # new ImageHDU appended

    Or do it in one go:

    .. code-block:: python

       ioops.save2fits((sky,obs,psf),'myfitsfile.fits') # all 3 instances saved to separate HDUs

    """

    # open existing or create new FITS file
    fout = fits.open(fitsfile,mode='append')

    try:
        phdu = fout[0].header
        logging.info("Existing FITS file '{:s}' opened.".format(fitsfile))
    except IndexError:
        logging.info("New FITS file '{:s}' created.".format(fitsfile))
        phdu = fits.PrimaryHDU()
        fout.append(phdu)
        logging.info("New (empty) primary HDU written.")

    if not isinstance(image,(tuple,list)):
        image = (image,)

    for image_ in image:
        
        # use WCS system if requested (and if present)
        if usewcs is True and hasattr(image_,'wcs') and isinstance(image_.wcs,wcs.wcs.WCS):
            header = image_.wcs.to_header()
        else:
            header = fits.Header()

        # add type of image to header
        try:
            imgtype = image_.__class__.__name__
            header['IMGTYPE'] = (imgtype, 'Content of image')
        except:
            raise
            
        # add cards for scaling and units
        header['BSCALE'] = (1.00000, 'Scaling of pixel values')
        header['BZERO'] = (0.00000, 'Zero offset')
        header['BUNIT'] = (image_.data.unit.to_string(), 'Unit of image data')

        # helper func for simple attributes
        def add2header(obj,attr,comment='',keyword=None,suffix=''):
            if hasattr(obj,attr):
                if keyword is None:
                    keyword = attr.upper()

                if keyword.endswith('_'):
                    keyword = keyword[:-1]
                    
                keyword = "{:<8s}".format(keyword.upper()[:(8-len(suffix))]+suffix)
                header[keyword] = (getattr(obj,attr),comment)

        # use helper func to put attrs in header
        add2header(image_,'objectname','target name','OBJECT')
        add2header(image_,'telescope','telescope/facility')
        add2header(image_,'instrument','instrument')

        # store all model parameter values
        # enforce last valid non-whitespace char to be '_' to avoid name-space collisions
        pairs = [(k,v) for k,v in image_.__dict__.items() if k.endswith('_')]
        for k,v in pairs:
            add2header(image_,k,'model parameter value',suffix='_')

        # slightly more complex attrs
        if hasattr(image_,'pa'):
            header['PA'] = (image_.pa.value, 'position angle ({:s} from N)'.format(image_.pa.unit.to_string()))
            
        if hasattr(image_,'wave'):
            header['wave'] = (image_.wave.value, 'wavelength ({:s})'.format(image_.wave.unit.to_string()))

        # Add HDU creation timestamp.
        # Format according to https://fits.gsfc.nasa.gov/standard30/fits_standard30aa.pdf
        #   section 4.4.2.1 "General descriptive keywords, DATE keyword"
        header['DATE'] = (datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"), 'HDU creation time (UTC)')

        # encapsulate image data in new ImageHDU and append
        dhdu = fits.ImageHDU(image_.data.value.T,header=header)
        fout.append(dhdu)
        logging.info("Data saved as new ImageHDU.")

    fout.close()
    

class FitsFile:

    """Convenience class to read (OI)FITS files."""
    
    def __init__(self,f,mode='r'):

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
        key : str
            Name of key, e.g. ``'naxis'``. Capitalization is not
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

           ff.get('naxis',4,'value') # return value of key 'naxis' from hdu=4
             2

           ff.get('naxis',4,) # 'value' is the default
             2

           ff.get('naxis',4,'comment') # return only the corresponding comment
             'Binary table'

           ff.get('naxis',4,'card') # return the entire card = (key,value,comment) as a tuple
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

           ff = FitsFile('..examples/Circinus_Burtscher_2013.oifits')

           ff.getdata(hdu=4) # return all hdu.data

           ff.getdata(4,'vcoord') # Get v-coordinates (from u,v plane)
              array([-41.04285049,   1.51398432,  13.22854233,  35.11543655,
                      30.22608757,   1.65713799, -31.00023651, -40.52364731,
                      42.68415833,  33.51140594,  15.40285206,   3.22378659,
                      -3.13334179,  -8.67062855,  15.3754034 ,  14.42059231,
                       9.64568615,   6.66133642,   3.32804942,  -2.93655539,
                      -8.03857231,  -8.59758091,  15.26423931,  13.05595779,
                      10.33599758,   6.71977472,   0.81495309,  -2.18943191])

           ff.getdata(2,'raep0'), ff.getdata(2,'decep0') # get RA & DEC
              (array([ 213.292174]), array([-65.33936]))
        """

        hdu = self.hdulist[hdu]

        if field is None:
            res = hdu.data
        else:
            res = hdu.data[field]

        return res
    
