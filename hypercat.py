import os
import logging
logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')
import numpy as N
from scipy import ndimage
import pyfits
import h5py
import padarray
import ndiminterpolation_vectorized


__version__ = '20160602'   #yyymmdd
__author__ = 'Robert Nikutta <robert.nikutta@gmail.com>'

"""Utilities for handling CLUMPY image files."""


class ModelCube:

#    def __init__(self,hdffile='/home/robert/dev/hypercubizer/clumpy_img_cube_2200models_19waves_halfsized.hdf5',\
    def __init__(self,hdffile='clumpy_img_cube_2200models_19waves_halfsized.hdf5',\
                 hypercube='imgdata',\
                 ndinterpolator=True):

        logging.info("Opening HDF5 file: %s " % hdffile)
        self.h = h5py.File(hdffile,'r')
        self.group = self.h[hypercube]
        
        logging.info("Loading sampling parameters.")
        shape = self.group['hypercubeshape'].value
        select = N.argwhere(shape>1).flatten()  # selector mask: only theta_i with more than one element
        
        self.theta = (padarray.PadArray(self.group['theta'].value).unpad)
        self.theta = [self.theta[j] for j in select]

        self.paramnames = (self.group['paramnames'].value)[select]

        if self.paramnames[-1] == 'wave':
            self.x = self.theta[-3]
            self.y = self.theta[-2]
            self.wave = self.theta[-1]
        else:
            self.x = self.theta[-2]
            self.y = self.theta[-1]

        ramgigs = self.data = self.group['hypercube'].size*4/1024**3.
        logging.info("Loading hypercube '%s' to RAM (~%.2f GB required) ..." % (hypercube,ramgigs))
        self.data = self.group['hypercube'][...]
        self.data = self.data.squeeze()  # drop from ndim-index all dimensions with length-one

        if ndinterpolator is True:
            logging.info("Instantiating n-dim interpolation object ...")
            self.ip = ndiminterpolation_vectorized.NdimInterpolation(self.data,self.theta,mode='lin')

        logging.info("Done. Closing HDF5 file.")
        self.h.close()


    def pprinter(self):

        """Very basic pretty-print of the parameter names and values."""
        
        import pprint

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(zip(self.paramnames,self.theta))

        
    def print_sampling(self,n=11,fmt="%7.3f"):

        """Print a summary table of the sampled parameters and values.

        Parameters:
        -----------
        n : int
            n first elements of every parameter value list will be
            printed. If the list is shorter than 'n', all of it will
            be printed. Default: 11.

        fmt : str
            Format string to use for a single value. Default: '%7.3f'

        """

        maxstr = "%%% ds  " % max([len(p) for p in self.paramnames])  # longest parameter name
        maxn = max([int(N.ceil(N.log10(t.size))) for t in self.theta])  # largest parameter cardinality

        header = "Parameter Range                Nvalues  Values\n" + "-"*72
        print header
        
        for p,v in zip(self.paramnames,self.theta):

            srange = "[%s" % fmt % v[0] + " - %s]" % fmt % v[-1]  # range string
            m = min(n,v.size)  # print n elements or all if number of elements smaller than n

            vals = ["%s" % fmt % val for val in v[:m]]
            svals = ",".join(vals)  # all values to be printed, as a single string
            if v.size > n:
                svals += ', ...'  # continuation indicator, if any

            # bring everything together
            print maxstr % p + "    %s" % srange + "  (%%%dd)   " % maxn % v.size +  svals


    def get_image(self,vector,full=True):

        # this is for the current array layout: ['sig', 'i', 'N', 'q', 'tv', 'x', 'y', 'wave']

        vec = list(vector)
        vec.insert(-1,tuple(self.x.tolist()))
        vec.insert(-1,tuple(self.y.tolist()))
        vec = tuple(vec)
        
        image = self.ip(vec)
        image = image.squeeze()
#        image = image.reshape((self.x.size,self.y.size))
#        
        if full is True:
            image = mirror_halfimage(image)
                
        return image.squeeze()
    
            
#    def get_image(self,theta,wave,full=False):
#
#        """
#
#        theta = [sig,i,N,q,tv]
#        wave = [2.2,2.8,3.5], or [2.2,], etc.
#
#        """
#
#        try:
#            theta = list(theta)
#        except:
#            raise
#
#        try:
#            nwave = len(wave)
#        except TypeError:
#            wave = (wave,)
#            nwave = 1
#            assert (len(wave) == nwave), "Problem converting 'wave' to tuple."
#
#        image = N.zeros((self.x.size,self.y.size,nwave))
#        for ix in xrange(self.x.size):
#            for iy in xrange(self.y.size):
#                image[ix,iy,:] = self.ip(N.array(theta+[ix,iy]),N.array(wave))
#
#        if full is True:
#            image = mirror_halfimage(image)
#                
#        return image.squeeze()



#class Image:
#
#    def __init__(self,imgarray):
#
#        self.image = imgarray
#        assert (self.image.ndim == 2), "'imgarray' must be 2-dimensional (but is %d)" % self.image.ndim
#        assert (self.image.shape[0] == self.image.shape[1])



def mirror_halfimage(halfimage):

    """Take half-sized image (image cube) and return the full version.

    Parameters:
    -----------
    halfimage : array
        Expected shape of halfimage: (nx,ny,nz), or (nx,ny),
        and nx == ny/2 +1 must hold (or an exception will be raised).

    Returns:
    --------
    fullimage : array
        Array with first dimension mirrored, i.e. fullimage.shape =
        (ny,ny,nz).

    """
        
    shape = list(halfimage.shape)
    nx, ny = shape[:2]

    assert (nx == (ny/2 + 1)), "Image/cube doesn't seem to contain a half-sized array (dims = (%s). Not mirroring." % (','.join([str(s) for s in shape]))

    shape[0] = 2*nx-1
    
    logging.info("Mirroring half-cube / half-image.")
    
    fullimage = N.zeros(shape,dtype=N.float32)
    fullimage[:nx,...] = halfimage[::-1,...]
    fullimage[nx-1:,...] = halfimage[:,...]
    
    return fullimage


def rotate_image(image,angle,direction='NE'):

    """Rotate an image around its central pixel by 'angle' degrees.

    The parts of 'image' which after rotation are outside the frame,
    will be dropped. Areas in the rotated image which are empty will
    be filled will zeros.

    The rotation is performed using high-quality cubic spline
    interpolation.

    Parameters:
    -----------

    image : 2D array
        The square image array to be rotated.

    angle : float or int
        Image will be rotated by that many degrees.

    direction : str
        Rotation direction. Default is 'NE', i.e. from North to East,
        i.e. anti-clockwise, which is astronomical standard. To rotate
        clockwise instead, give direction='NW'.

    Returns:
    --------

    rotimage : 2D array
        The 'image', rotated by 'angle' degrees towards 'direction'.

    """
    
    assert (image.ndim == 2), "'image' must be 2-dimensional (but is %d)" % image.ndim
    assert (image.shape[0] == image.shape[1]), "'image' must be square (but has shape (%s))" % ','.join(list(image.shape))

    if direction == 'NW':
        angle = angle
    
    rotimage = ndimage.rotate(image,angle,reshape=False)

    return rotimage
    

def get_clean_file_list(d,suffix='.fits'):

    """Get a sorted list of file-paths ending in suffix, from directory d."""

    sep = os.path.sep
    files = os.listdir(d)
    files = sorted([os.path.normpath(d+sep+f) for f in files if f.endswith(suffix)])

    return files


def get_sed_from_fitsfile(fitsfile):

    """Compute SED from all image slices stored in fitsfile.

    Computes for every image slice the area integral

      \int I(x,y) dx dy = \Delta x \times \sum_j I_j

    where j runs over all pixels of the image, and
    \Delta x (= \Delta y) is the physical pixel size.

    Parameters:
    -----------
    fitsfile : str
        Name/path of a CLUMPY (full-sized) FITS file.

    Returns:
    --------
    wave : array
        Wavelength in micron.

    sed : array
        The SED lambda*F_lambda, area-integrated from the images.

    """

    header = pyfits.getheader(fitsfile)
    wave = N.array([v for k,v in header.items() if k.startswith('LAMB')])  # in micron
    data = pyfits.getdata(fitsfile)  # uses first HDU by default
    sed = N.sum(data,axis=(1,2)) * header['CDELT1']**2   # \int I(x,y) dx dy

    return wave, sed


def mirror_fitsfile(fitsfile,hdus=('IMGDATA','CLDDATA'),save_backup=True):

    """Mirror/flip a half-sized CLUMPY image cube (in fits file) along its x-axis.

    CLUMPY image fits files come in two variants: full-sized (square
    in x,y), and half-sized (reducing storage, b/c of the left-right
    symmetry of the images).

    This function expands all data cubes in a half-sized fits file to
    full (square) size, and updates the fits file in-place. The
    original file can be back-up automatically.

    Parameters:
    -----------
    fitsfile : str
        Name/path of a CLUMPY half-sized FITS file.

    hdus : tuple
        HDU names to expand. Default: ('IMGDATA','CLDDATA'). If one of
        the requested HDUs does not seem to conform to the half-sized
        type, a warning will be printed, and this HDU will be skipped.

    save_backup : bool
        If True (default), the original half-size fits file will be
        backed up before the expanded (full-sized) fits file will be
        written under the original name. The backed up file will have
        the suffix '.bak'.

    """

    logging.info("Opening FITS file: %s " % fitsfile)

    hdulist = pyfits.open(fitsfile,mode='update',save_backup=save_backup)

    hdunames = [h.name for h in hdulist]
    
    for hduname in hdus:
        
        logging.info("Accessing HDU: %s" % hduname)
        try:
            hdu = hdulist[hduname]
        except KeyError:
            logging.warning("HDU '%s' in file %s not found. Skipping this HDU." % (hduname,fitsfile))
            continue
        
        shape = list(hdu.shape)
        ny, nx = shape[-2:]
        if nx != (ny/2 + 1):   # caution: integer division
            logging.warning("HDU '%s' in file %s doesn't seem to contain a half-sized array (dims = (%s). Skipping this HDU." % (hduname, fitsfile, ','.join([str(s) for s in shape])))
            continue
        else:
            shape[-1] = 2*nx-1
        
        logging.info("Mirroring half-cube.")
        aux = hdu.data[...,::-1]  # x-flipped half-cube
        newdata = N.zeros(shape,dtype=N.float32)
        newdata[...,:nx] = aux[...]
        newdata[...,nx-1:] = hdu.data[...]
        hdu.data = newdata

    logging.info('Flushing and closing fits file.')
    hdulist.close()
    print
            

def mirror_all_fitsfiles(d,suffix='.fits',hdus=('IMGDATA','CLDDATA')):

    """Mirror/flip all half-sized CLUMPY image cubes (in fits files) along their x-axis.

    Please see docstring of mirror_fitsfile() for details.

    Parameters:
    -----------
    d : str
        Directory holding the files to be mirrored.

    suffix : str
        File ending of the fits files. Default: '.fits'

    hdus : tuple
        Sequence of HDU names to mirror/flip in every FITS file.

    """
    
    files = get_clean_file_list(d,suffix='.fits')
    nfiles = len(files)
    
    logging.info("Found %d '%s'-files in directory %s" % (nfiles,suffix,d))
    
    for j,f in enumerate(files):
        logging.info("Working on file %s [%d/%d]" % (f,j+1,nfiles))
        mirror_fitsfile(f,hdus=hdus)
    
    logging.info("All files mirrored.")
