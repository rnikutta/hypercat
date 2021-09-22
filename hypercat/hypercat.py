__version__ = '0.1.10' # version tag last updated on 2021-09-20
__author__ = 'Robert Nikutta <robert.nikutta@gmail.com>'

"""Utilities for handling the CLUMPY image hypercube.

.. automodule:: hypercat
"""

# IMPORTS
# std lib
import os
from collections import OrderedDict
from operator import itemgetter
from copy import copy
import time
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# 3rd party
import numpy as np
from astropy import units as u
from astropy import wcs
from astropy.io import fits
import h5py

# own
from .loggers import *
from . import ndiminterpolation
from .obsmodes import *
from .imageops import *
from .utils import *
from .ioops import *
from .units import *

# CLASSES

class ModelCube:

    def __init__(self,hdffile='hypercat_20200830_all.hdf5',\
                 hypercube='imgdata',\
                 subcube_selection='onthefly',\
                 subcube_selection_save=None,
                 omit=('x','y'),bold=True):

        """Model hypercube of CLUMPY images. Can be generalized to any hypercube.

        Parameters
        ----------
        hdffile : str
            Path to the model hdf5 file. Default: ``'hypercat_20200830_all.hdf5'``

        hypercube : str
            Name of the hypercube within ``hdffile`` to use (currently
            either ``'imgdata'`` or ``'clddata'``). Default: ``'imgdata'``.

        subcube_selection : str | None

            ``'onthefly'``, ``'interactive'`` or path to a `.json`
            file containing the indices to select for every axis in
            the ``hypercube`` in the ``hdffile``.

            If ``'onthefly'``, the hypercube will be memory-mapped,
            but not yet loaded into RAM (it's too large anyway). When
            called the instance of :class:`ModelCube` with a parameter
            vector, a minimal cube spanning all parameter sub-vectors
            will be computed and then loaded into memory. `onthefly`
            is the default mode, as it is the most convenient. It is
            also the slowest (~400ms per image if selecting from the
            full hypercube).

            If ``'interactive'``, a simple selection dialog will be
            launched in the terminal/session, allowing to
            select/unselect entries from every axis (one at a
            time). Once done, the corresponding list of index lists is
            created, and the hyper-slab (sub-cube) loaded from disk to
            RAM. The nested list of selected indices can be
            conveniently stored into a json file for later re-use (see
            arg ``subcube_selection_save``).

            If ``subcube_selection`` is the path to a json file, it is
            a file that can be created with
            ``subcube_selection='interactive'``. Then also provide as
            ``subcube_selection_save`` a file path to the to-be-stored
            json file.

        subcube_selection_save : str | None
            If not ``None``, it a the path to a json file with the list of
            index lists that select a subcube from the full
            ``hypercube``. I.e. a json file with selection indices can
            be created once, and then the same selection can be
            repeated any time by simply loading the json file.

        omit : tuple
            Tuple of parameter names (as strings) to omit from subcube
            selection. These axes will be automatically handled.
        
            .. warning:: 

               This functionality is not fully implemented yet, and is
               currently meant for the 'x' and 'y' axes only. Best not
               to touch for now.

        Example
        -------
        .. code-block:: python

            # instantiate
            cube = ModelCube()  # all defaults (i.e. default hdf5 file, 'imgdata' hypercube, 'onthefly' mode

        """

        self.bold = bold
        self.hdffile = hdffile
        self.omit = omit
        self.subcube_selection = subcube_selection
#P        print('self.hdffile,self.omit,self.subcube_selection', self.hdffile,self.omit,self.subcube_selection)

        logging.info("Opening HDF5 file: {:s} ".format(hdffile))
        self.open_hdffile(hdffile)
        self.groupname = hypercube
        self.get_group()
        logging.info("Loading sampling parameters.")
        self.get_cube_layout()
        self.get_eta()
        logging.info("Closing HDF5 file.")
        self.close_hdffile()
        self.compute_cubesize() # compute total byte size


        # SELECT A SUB-HYPERCUBE
        if self.subcube_selection is not None:
            if self.subcube_selection == 'interactive':
                self.theta, self.idxes, self.subcubesize =\
                    getIndexLists(self.theta,self.paramnames,initsize=self.fullcubesize,omit=self.omit)

                if subcube_selection_save is not None:
                    storejson(subcube_selection_save,{self.groupname : self.idxes})

            elif os.path.isfile(self.subcube_selection):
                d = loadjson(self.subcube_selection)
                self.idxes = d[self.groupname]
                self.theta = [self.theta[j][self.idxes[j]] for j in range(len(self.theta))]
                self.subcubesize = get_bytesize(self.idxes)
                
            elif self.subcube_selection == 'onthefly': # single-values per parameter; load minimal hyperslab around that
                pass

            else:
                raise Exception("Unknown mode for 'subcube_selection'. Must be either of 'onthefly', 'interactive', or a '.json' file containing selecting indices.")

        if not isinstance(self.theta,np.ndarray):
            self.theta = np.array(self.theta)
            
        # for each parameter save its sampling as a member (attach '_' to the name, to minimize the change of name space collisions)
        for j,pn in enumerate(self.paramnames):
            setattr(self,pn+'_',self.theta[j])

        self.theta_full = copy(self.theta)
        
                
        prefix, suffix = get_bytes_human(self.subcubesize)
            
        if self.subcubesize != self.fullcubesize:
            hypercubestr = 'hyperslab [shape: ({:s})] from'.format(seq2str([len(_) for _ in self.theta]))
        else:
            hypercubestr = ''
            
        logging.info("Loading {:s} hypercube '{:s}' [shape: ({:s})] to RAM ({:.2f} {:s} required) ...".format(hypercubestr,hypercube,seq2str(self.fullcubeshape),prefix,suffix))
        self.dsmm = memmap_hdf5_dataset(hdffile,hypercube+'/hypercube')
        
        if subcube_selection != 'onthefly':

            # materialize data cube
            self.data = get_hyperslab_via_mesh(self.dsmm,self.idxes)
            logging.info("Done.")

            self.ip = self.make_interpolator()
        
        print("Inspect the loaded hypercube with .print_sampling()\n")
        self.print_sampling(8) # print first 8 sampled values


    def __call__(self,vector,full=True):

        """Just a convenience wrapper for :func:`get_image()`."""

        img = self.get_image(vector,full)
        
        return img
        

    def open_hdffile(self,f,mode='r'):
        self.h = h5py.File(self.hdffile,mode)

    def close_hdffile(self):
        self.h.close()

    def get_group(self):
        self.group = self.h[self.groupname]

    def get_cube_layout(self):
        self.paramnames = [e.decode() for e in self.group['paramnames']]
        self.theta = self.group['theta'].value
        self.idxes = [list(range(len(t))) for t in self.theta]

    def get_eta(self):
        iY = self.paramnames.index('Y')
        self.Ymax = self.theta[iY].max()  # largest Y, i.e. 'FOV' of the images in units of Rd
        iy = self.paramnames.index('y')
        npix = self.theta[iy].size
        self.npix_per_Rd = (npix-1)/(2.*float(self.Ymax))
        self.eta = self.npix_per_Rd  # alias

    def compute_cubesize(self):
        # compute total byte size
        self.fullcubeshape = [t.size for t in self.theta]
        self.nvoxels = np.prod(self.fullcubeshape)
        self.wordsize = 4.
        self.fullcubesize = self.nvoxels * self.wordsize #/ 1024.**3
        self.subcubesize = self.fullcubesize
                     
    
    def print_sampling(self,n=11,fmt="%7.3f",bold=False):

        startbold, endbold = "", ""
        if bold is True:
            startbold, endbold = "\033[1m", "\033[0m" 
        
        maxlen = max([len(p) for p in self.paramnames])  # length of longest parameter name
        maxn = max([int(np.ceil(np.log10(t.size))) for t in self.theta])  # largest parameter cardinality

        header = "Parameter  Range                Nvalues  Sampled values"
        _len = len(header)
        rule = "-"*_len
        print(rule)
        print(header)
        print(rule)

        for p,v in zip(self.paramnames,self.theta):

            srange = "[%s" % fmt % v[0] + " - %s]" % fmt % v[-1]  # range string
            m = min(n,v.size)  # print n elements or all if number of elements smaller than n

            vals = ["%s" % fmt % val for val in v[:m]]
            svals = ",".join(vals)  # all values to be printed, as a single string
            if v.size > n:
                svals += ', ...'  # continuation indicator, if any

            # bring everything together
            parstr = '%%%ds' % maxlen % p
            asterisk = " "
            if (p not in self.omit) and (len(vals) != 1):
                parstr = startbold + parstr
                asterisk = "*"
                svals = svals  + endbold
                
            print(parstr + asterisk + "    %s" % srange + "  (%%%dd)   " % maxn % v.size +  svals)
            
        print(rule)
        print("Parameters printed in %sbold%s and/or marked with an asterisk (*) are interpolable." % (startbold,endbold))

        prefix, suffix = get_bytes_human(self.subcubesize)
        print("Hypercube size: %g (%s)" % (prefix, suffix))

    parameter_values = property(print_sampling) #: Property alias for :func:`print_sampling`
                     

#    def get_minimal_cube_single(self,vector):
#
#        idxes = []
#        for j in range(len(vector)):
#            t_ = self.theta[j]
#            v_ = vector[j]
#            left = np.digitize(v_,t_).item() - 1
#            if t_[left] == v_:
#                right = left
#                left = left - 1
#            else:
#                right = left + 1
#                
#            idxes.append([left,right])
#
#        idxes.append(list(range(self.theta[-2].size)))
#        idxes.append(list(range(self.theta[-1].size)))
#
#        theta = [self.theta[j][idxes[j]] for j in range(len(self.theta))]
#        subcubesize = get_bytesize(idxes)
#
#        # materialize data cube
#        data = get_hyperslab_via_mesh(self.dsmm,idxes)
#
#        return idxes,theta,data
        

    def get_minimal_cube(self,vector):

        idxes = []
        for j in range(len(vector)):
            t_ = self.theta[j]
            v_ = vector[j]
            if not isinstance(v_,tuple):
                v_ = tuple([v_])
                
            digits = np.digitize(v_,t_,right=True)
            left = max((min(digits)-1,0))
            right = max((max(digits)+1,2))           

            idxes_ = list(range(left,right))
            idxes.append(idxes_)
            
        idxes.append(list(range(self.theta[-2].size)))
        idxes.append(list(range(self.theta[-1].size)))

        theta = [self.theta[j][idxes[j]] for j in range(len(self.theta))]
        logging.info("Loading a subcube of %g %s into RAM." % (get_bytes_human(get_bytesize(idxes))))

        # materialize data cube
        data = get_hyperslab_via_mesh(self.dsmm,idxes)

        return idxes,theta,data

    
    def make_interpolator(self,idxes=None,theta=None,data=None):

        if idxes is None:
            idxes = self.idxes
        if theta is None:
            theta = self.theta
        if data is None:
            data = self.data
        
        # find axes with dim=1, squeeze subcube, remove the corresponding paramnames
        logging.info("Squeezing all dim-1 axes...")
        sel = np.argwhere([len(t)>1 for t in idxes]).flatten().tolist()
        theta_sel = itemgetter(*sel)(theta)
        
        # instantiate an n-dim interpolator object
        logging.info("Instantiating n-dim interpolation object ...")
#        ip = ndiminterpolation.NdimInterpolation(data.squeeze(),theta_sel,mode='lin')
        ip = ndiminterpolation.NdimInterpolation(data.squeeze(),theta_sel,mode='log')

        logging.info("Done.")
        
        return ip
        
        
    def get_image(self,vector,full=True):

        """Extract hyperslice from the hypercube via N-dim interpolation.

        Parameters
        ----------
        vector : seq
            A vector of model parameter values at which the image
            should be interpolated. This is very flexible and can
            return variously-shaped arrays, from standard 2D images
            (x,y) to multi-dimensional hyper-slices of the CLUMPY
            hypercube. See `Examples` below.

        full : bool
            If ``True`` (default), the extracted image (which by
            default is a half-image due to CLUMPY's inherent axial
            symmetry) is left-right mirrored, and a full-sized
            (square) image is returned.

        Examples
        --------
        .. code:: python

            # vector of parameter values; pixel axes are implicit (i.e. don't specify them)
            theta = (30,0,3,0,20,9.7) # here: (sig,i,N0,q,tauv,lambda)
            image = cube.get_image(theta)
            print(image.shape)
              (441,441)   # (x,y)

            # multi-wavelength cube
            theta = (30,0,3,0,20,(2.2,9.7,12.)) # 3 lambda values
            image = cube.get_image(theta)
            print(image.shape)
              (3,441,441)   # (x,y,lambda)
            
            # multi-wavelength and multi-viewing angle
            theta = (30,(0,30,60,90),3,0,20,(2.2,9.7,12.)) # 4 viewing angles, 3 lambdas
            image = cube.get_image(theta)
            print(image.shape)
              (4,3,441,441)
        """

        if self.subcube_selection == 'onthefly':
            idxes, theta, data = self.get_minimal_cube(vector)
            ip = self.make_interpolator(idxes,theta,data)
        else:
            ip = self.ip
        
        vec = list(vector)

        # sub-vectors can be arrays or lists; convert to tuples
        for j,v in enumerate(vec):
            if isinstance(v,np.ndarray):
                vec[j] = tuple(v.tolist())
            elif isinstance(v,list):
                vec[j] = tuple(v)
        
        vec.append(tuple(self.x_.tolist()))
        vec.append(tuple(self.y_.tolist()))
        
        vec = tuple(vec)

        image = ip(vec)
        image = image.squeeze()

        # mirror image (or cube) about the x-axis (axis=-2)
        if full is True:
            image = mirror_axis(image,axis=-2)
            
        return image

    
    def get_sed(self,vec,wave=None):

        """Get raw SED for all sample wavelengths in the cube.

        Parameters
        ----------
        vec : tuple
            Tuple of all model parameters in the loaded cube, _except_
            wavelength (will be taken from cube as-is). Only single
            values per parameter please.

        wave : seq
            List/tuple/array of wavelengths. If None (default), use
            the built-in wavelengths.

        """

        if wave is None:
            wave = self.wave_ # list of built-in wavelengths; otherwise use supplied wavelengths

        vec = tuple((*vec,tuple(wave)))  # construct parameter vector with wvelengths
        images = self(vec)  # interpolate images at all wavelengths
        sed = np.array([_.sum() for _ in images])  # sum up all images to get SED
            
        return wave, sed


class Source:
    
    def __init__(self,cube,luminosity='1e45 erg/s',distance='1 Mpc',tsub='1500 K',pa='0. deg',objectname=''):

        """Abstraction of an astrophysical source (AGN).

        Takes a hypercube of model images. Instantiates an n-dim
        interpolation object. Computes dust sublimation radius and
        image pixelscale. Resolves source name with Vizier and
        constructs a WCS (optional).

        Parameters
        ----------
        cube : instance
            Instance of ModelCube. Holds the n-dim hypercube of model
            images.

        luminosity : str
            AGN bolometric luminosity, e.g. '1e45 erg/s' or '1e12 Lsun'.

        distance : str
            Distance to the source, e.g. '14.4 Mpc'.

        tsub : str
            Dust sublimation temperature. Default is '1500 K',
            appropriate e.g. for astrophysical silicates.

        pa : str
            Position angle with respect to North (=0 deg). Default is
            '0. deg'. If not 0, the image will be rotated by pa
            (positive values rotate North-to-East,
            i.e. anti-clockwise, negative values rotate North-to-West,
            i.e. clockwise).

        """

        self.objectname = objectname
        self.cube = cube  # instance of ModelCube
        self.luminosity = getQuantity(luminosity,recognized_units=UNITS['LUMINOSITY'])
        self.distance = getQuantity(distance,recognized_units=UNITS['CUNITS'])
        self.pa = getQuantity(pa,recognized_units=UNITS['ANGULAR'])  # PA of source, in angular unitsfrom N to E
        
        self.Rd = get_Rd(luminosity,tsub=tsub,outunit='pc')
        self.pixelscale = get_pixelscale(self.Rd,self.distance,outunit='mas',npix=cube.eta)[2]*u.pix  # mas (per pixel)

        
    def __call__(self,theta,total_flux_density='1 Jy',snr=None,brightness_units='Jy/arcsec^2'):

        self.theta = theta

#        if self.cube.data.ndim != len(theta) + len(self.cube.omit):
#            raise Exception, "The provided model hypercube and vector of parameters are not compatible!"
#        if self.cube.ip.data_hypercube.ndim != len(theta) + len(self.cube.omit):
#            raise Exception, "The provided model hypercube and vector of parameters are not compatible!"

        # find out what wave is
        cube_waveidx = self.cube.paramnames.index('wave')
        cube_wave = self.cube.theta[cube_waveidx]
        if cube_wave.size == 1:  # single-valued wave in cube
            wave = cube_wave[0] * u.micron
        else: # multi-valued wave in cube, i.e. last entry in user-provided theta vector is wave
            wave = self.theta[-1] * u.micron

        # get raw image
        rawimage = self.cube.get_image(theta)
#        print("rawimage.min(): ",rawimage.min())
#ROT        co = (rawimage < 0.)
#ROT        rawimage[co] = 0. #*rawimage #.unit

        # instantiatie Image class, with physical units
#        print("SOURCE: self.pa before instaniating Image = ")
#        print(self.pa, type(self.pa))
        sky = Image(rawimage,pixelscale=self.pixelscale,pa=self.pa,total_flux_density=total_flux_density,snr=snr,brightness_units=brightness_units)

        sky.theta = self.theta
        sky.wave = wave
        sky.pa = self.pa
        sky.objectname = self.objectname  # attach source.objectname to the image of the sky

        # find all values of parameters used for creating this image (including the squeezed axes)
        vals = copy(self.cube.theta)
        for j,e in enumerate(vals):
            if e.size == 1:
                vals[j] = e[0]

        idx = [j for j,e in enumerate(self.cube.theta) if e.size>1 and self.cube.paramnames[j] not in self.cube.omit]
        vals[idx] = sky.theta
        pairs = [(e,vals[j]) for j,e in enumerate(self.cube.paramnames) if e not in self.cube.omit]
        for attr,val in pairs:
            setattr(sky,attr+'_',val)
        
        # construct WCS if possible
        wcs = get_wcs(sky)
        if wcs is not None:
            sky.wcs = wcs
            
        return sky



# HELPER FUNCTIONS

def lum_dist_to_pixelscale(lum,dist,cube):

    Rd = get_Rd(lum,tsub=1500.,outunit='pc')
    
    angular, angular_per_linsize, angular_per_pixel = get_pixelscale(Rd,dist,outunit='mas',npix=cube.eta)

    return angular, angular_per_linsize, angular_per_pixel
    

def get_Rd(luminosity='1e45 erg/s',tsub='1500 K',outunit='pc'):

    """Compute the dust sublimation radius :math:`R_d` from luminosity of
    source and dust sublimation temperature.

    Uses Eq. (1) from Nenkova+2008b: http://adsabs.harvard.edu/abs/2008ApJ...685..160N

    .. math::

       R_d = 0.4 \\left( \\frac{L}{10^{45}\,{\\rm erg\,s^{-1}}}\\right)^{\\!1/2} \\left( \\frac{1500\,\\rm K}{T_{\\rm sub}}\\right)^{\\!2.6} {\\rm pc}

    Parameters
    ----------
    luminosity : str
        AGN bolometric luminosity, e.g. '1e45 erg/s'.

    tsub : str
        Dust sublimation temperature. Default is '1500 K',
        corresponding e.g. to astrophysical silicates.

    outunit : str
        Desired output units of length. The result will be
        automatically converted to these units. Defaults to 'pc'.

    Returns
    -------
    Rd : float
        Dust sublimation radius Rd in units `outunit`.

    """

    lum = getQuantity(luminosity,recognized_units=UNITS['LUMINOSITY']).to('erg/s').value
    tsub = getQuantity(tsub,recognized_units=UNITS['TEMPERATURE']).to('K').value
    
    Rd = 0.4*np.sqrt(lum/1e45) * (1500./tsub)**2.6 * u.pc

    return Rd.to(outunit)


def get_pixelscale(linsize,distance,outunit='arcsec',npix=1):

    """From linear size in the sky and distance to source, compute angular
    size, and angular sizes per linear size and per pixel.

    Parameters
    ----------

    linsize : str
        Linear size of a source or feature in the sky, e.g. '1 pc'.

    distance : str
        Distance to the source from observer, e.g. '4 Mpc'.

    outunit : str
        Desired output units for the returned angular size. The result
        will be automatically converted to these units. Default: 'arcsec'

    npix : int
        The number of pixels that the angular size will be imaged
        with.
    """
    
    linsize = getQuantity(linsize,UNITS['LINEAR'])
    distance = getQuantity(distance,UNITS['LINEAR'])
    
    angular = np.arctan2(linsize,distance).to(outunit)
    angular_per_linsize = angular / linsize  # e.g. arcsec/pc
    angular_per_pixel = angular / (float(npix)*u.pix)  # e.g. arcsec/pixel
    
    return angular, angular_per_linsize, angular_per_pixel


# PROBABLY HISTORICAL FUNCS, MAYBE DROP THEM

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

    header = fits.getheader(fitsfile)
    wave = np.array([v for k,v in header.items() if k.startswith('LAMB')])  # in micron
    data = fits.getdata(fitsfile)  # uses first HDU by default
    sed = np.sum(data,axis=(1,2)) * header['CDELT1']**2   # \int I(x,y) dx dy

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

    logging.info("Opening FITS file: {:s} ".format(fitsfile))

    hdulist = fits.open(fitsfile,mode='update',save_backup=save_backup)

    hdunames = [h.name for h in hdulist]
    
    for hduname in hdus:
        
        logging.info("Accessing HDU: {:s}".format(hduname))
        try:
            hdu = hdulist[hduname]
        except KeyError:
            logging.warning("HDU '{:s}' in file {:s} not found. Skipping this HDU.".format(hduname,fitsfile))
            continue
        
        shape = list(hdu.shape)
        ny, nx = shape[-2:]
#        if nx != (ny/2 + 1):   # caution: integer division
        if nx != (ny//2 + 1):   # caution: integer division
            logging.warning("HDU '{:s}' in file {:s} doesn't seem to contain a half-sized array (dims = ({:s}). Skipping this HDU.".format(hduname, fitsfile, ','.join([str(s) for s in shape])))
            continue
        else:
            shape[-1] = 2*nx-1
        
        logging.info("Mirroring half-cube.")
        aux = hdu.data[...,::-1]  # x-flipped half-cube
        newdata = np.zeros(shape,dtype=np.float32)
        newdata[...,:nx] = aux[...]
        newdata[...,nx-1:] = hdu.data[...]
        hdu.data = newdata

    logging.info('Flushing and closing fits file.')
    hdulist.close()
    print()
            

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
    
    logging.info("Found {:d} '{:s}'-files in directory {:s}".format(nfiles,suffix,d))
    
    for j,f in enumerate(files):
        logging.info("Working on file {:s} [{:d}/{:d}]".format(f,j+1,nfiles))
        mirror_fitsfile(f,hdus=hdus)
    
    logging.info("All files mirrored.")

