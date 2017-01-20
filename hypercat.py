__version__ = '20170120'   #yyymmdd
__author__ = 'Robert Nikutta <robert.nikutta@gmail.com>'

"""Utilities for handling the CLUMPY image hypercube.

.. automodule:: hypercat
"""

# Class constants
UNITS_ANGULAR = ('arcsec','mas','milliarcsecond','deg','rad')  #: Recognized angular units, e.g. for pixel scale.
UNITS_LINEAR = ('m','cm','pc','kpc','Mpc','lyr','AU')  #: Recognized linear units (either for pixel scale, or for source distance, etc.)
CUNITS = UNITS_ANGULAR + UNITS_LINEAR  #: Their union.
# TODO: implement also per-beam, and per-pixel brightness specifications (and maybe also per-pc^2 etc.)
UNITS_BRIGHTNESS = ('Jy/pix','mJy/pix')  #: Recognized units for brightness-per-pixel.
#        self.UNITS_BRIGHTNESS_SOLIDANGLE = ('Jy/arcsec^2','Jy/mas^2','Jy/milliarcsec^2','mJy/arcsec^2','mJy/mas^2','mJy/milliarcsec^2')

import os
import sys
import logging
from collections import OrderedDict
from operator import itemgetter
import numpy as N
from scipy import ndimage
from astropy import nddata  # todo: re-implement the functionality provided by nddata.extract_array() & remove this dependency
from astropy import units as u
from astropy.io import fits as pyfits
import h5py
import padarray  # todo: test and check if current versions of numpy fix numpy bu 2190; if so, remove the dependency on padarray
import ndiminterpolation_vectorized  # re-integrate ndiminterpolation_vectorized back into ndiminterpolation
import bigfileops as bfo
import PSF_modeling

# CLASSES

# Custom formatter for logger
class LogFormatter(logging.Formatter):

    def __init__(self, fmt="%(levelno)s: %(msg)s"):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):

        # Replace the original format with one customized by logging level
        if record.levelno == logging.INFO:
            self._fmt = '%(message)s'
        else:
            self._fmt = '%(levelname)s: %(message)s'

        # return formatted msg
        return logging.Formatter.format(self, record)

# set up logger
hdlr = logging.StreamHandler(sys.stdout)
hdlr.setFormatter(LogFormatter())
logging.root.addHandler(hdlr)
logging.root.setLevel(logging.INFO)


class ModelCube:

    def __init__(self,hdffile='hypercat_20170109.hdf5',\
                 hypercube='imgdata',\
                 ndinterpolator=True,\
                 subcube_selection='interactive',
                 subcube_selection_save=None,
                 omit=('x','y')):

        """Model hypercube of CLUMPY images.

        Parameters
        ----------
        hdffile : str
            Path to the model hdf5 file. Default:
            `hypercat_20170109.hdf5`

        hypercube : str
            Name of the hypercube within `hdffile` to use (currently
            either ``imgdata`` or ``clddata``). Default" ``imgdata``.

        ndinterpolator : bool
            If ``True`` (default), an interpolation object for
            N-dimensional interpolation of the hypercube will be
            instantiated, and accessible via :func:`get_image`.

        subcube_selection : str

            'interactive' or the name of a json file containing the
            indices to select for every axis in the `hypercube` in the
            `hdffile`.

            If 'interactive', a simple selection dialog will be
            launched int the terminal/session, allowing to
            select/unselect entries from every axis (one at a
            time). Once done, the corresponding list of index lists is
            created, and the hyper-slab (sub-cube) loaded from disk to
            RAM.

            If a json file, it is a file that can be created with
            subcube_selection=True, and providing as
            subcube_selection_save a file path to the to-be-stored
            json file.

        subcube_selection_save : str | None
            If not None, it a the path to a json file withe the list
            of index lists that select a subcube from the full
            'hypercube'.

        omit : tuple

            Tuple of parameter names (as strings) to omit from subcube selection, and from

        Example
        -------
        .. code-block:: python

            # instantiate
            M = ModelCube()

        """

        self.omit = omit

        logging.info("Opening HDF5 file: %s " % hdffile)
        self.h = h5py.File(hdffile,'r')
        self.group = self.h[hypercube]
        
        logging.info("Loading sampling parameters.")
        self.paramnames = self.group['paramnames'].value.tolist()
        self.theta = padarray.PadArray(self.group['theta'].value).unpad
        self.idxes = [xrange(len(t)) for t in self.theta]
        logging.info("Closing HDF5 file.")
        self.h.close()
       
        # compute total byte size
        self.fullcubeshape = [t.size for t in self.theta]
        self.nvoxels = N.prod(self.fullcubeshape)
        self.wordsize = 4.
        self.fullcubesize = self.nvoxels * self.wordsize #/ 1024.**3
        self.subcubesize = self.fullcubesize

#        self.paramnames = (self.group['paramnames'].value)[select]

        # SELECT A SUB-HYPERCUBE
        if subcube_selection is not None:

            if subcube_selection == 'interactive':
                self.theta, self.idxes, self.subcubesize =\
                    bfo.getIndexLists(self.theta,self.paramnames,initsize=self.fullcubesize,omit=self.omit)

                if subcube_selection_save is not None:
                    bfo.storejson(subcube_selection_save,{'idxes':self.idxes})
                        
            else:
                d = bfo.loadjson(subcube_selection)
                self.idxes = d['idxes']
                self.theta = [self.theta[j][self.idxes[j]] for j in xrange(len(self.theta))]
                self.subcubesize = bfo.get_bytesize(self.idxes)

        prefix, suffix = bfo.get_bytes_human(self.subcubesize)
            
        if self.subcubesize != self.fullcubesize:
            hypercubestr = 'hyperslab [shape: (%s)] from' % seq2str([len(_) for _ in self.theta])
        else:
            hypercubestr = ''
            
        logging.info("Loading %s hypercube '%s' [shape: (%s)] to RAM (%.2f %s required) ..." % (hypercubestr,hypercube,seq2str(self.fullcubeshape),prefix,suffix))
        dsmm = bfo.memmap_hdf5_dataset(hdffile,hypercube+'/hypercube')
        self.data = bfo.get_hyperslab_via_mesh(dsmm,self.idxes)
        logging.info("Done.")

        self.x = self.theta[-2]
        self.y = self.theta[-1]
        if self.paramnames[-1] == 'wave':
            self.x = self.theta[-3]
            self.y = self.theta[-2]
            self.wave = self.theta[-1]

        # find axes with dim=1, squeeze subcube, remove the corresponding paramnames
        logging.info("Squeezing all dim-1 axes...")
        sel = N.argwhere([len(t)>1 for t in self.idxes]).flatten().tolist()
        self.theta = itemgetter(*sel)(self.theta)
        self.paramnames = itemgetter(*sel)(self.paramnames)
        self.data = self.data.squeeze()  # drop from ndim-index all dimensions with length-one
        logging.info("Done. New shape: (%s)" % seq2str(self.data.shape))
        
        # instantiate an n-dim interpolator object
        if ndinterpolator is True:
            logging.info("Instantiating n-dim interpolation object ...")
            self.ip = ndiminterpolation_vectorized.NdimInterpolation(self.data,self.theta,mode='lin')

        logging.info("Done.")
        
        print "Inspect loaded hypercube with .print_sampling()\n"
        self.print_sampling()
        

    def print_sampling(self,n=11,fmt="%7.3f"):

        """Print a summary table of the parameters and sampled values in the
        hypercube.

        Parameters
        ----------
        n : int
            The first `n` sampled elements of every parameter will be
            printed. If the list is shorter than `n`, all of it will
            be printed. Default: 11.

        fmt : str
            Format string to use for a single value. Default: '%7.3f'

        Example
        -------
        .. code-block:: python

            M.print_sampling(n=6)

        """

        maxstr = " %%% ds " % max([len(p) for p in self.paramnames])  # longest parameter name
        maxn = max([int(N.ceil(N.log10(t.size))) for t in self.theta])  # largest parameter cardinality

        header = "Parameter  Range                Nvalues  Sampled values"
        len_ = len(header)
        rule = "-"*len_
        print header
        print rule
        
        for p,v in zip(self.paramnames,self.theta):

            srange = "[%s" % fmt % v[0] + " - %s]" % fmt % v[-1]  # range string
            m = min(n,v.size)  # print n elements or all if number of elements smaller than n

            vals = ["%s" % fmt % val for val in v[:m]]
            svals = ",".join(vals)  # all values to be printed, as a single string
            if v.size > n:
                svals += ', ...'  # continuation indicator, if any

            # bring everything together
            parstr = maxstr % p
            asterisk = " "
            if p not in self.omit:
                parstr = "\033[1m" + parstr
                asterisk = "*"
                svals = svals  + "\033[0m"
                
            print parstr + asterisk + "    %s" % srange + "  (%%%dd)   " % maxn % v.size +  svals
            
        print rule
        print "Parameters printed in \033[1mbold\033[0m and/or marked with an asterisk (*) are interpolatable."


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
            image = M.get_image(theta)
            print image.shape
              (441,441)   # (x,y)

            # multi-wavelength cube
            theta = (30,0,3,0,20,(2.2,9.7,12.)) # 3 lambda values
            image = M.get_image(theta)
            print image.shape
              (3,441,441)   # (x,y,lambda)
            
            # multi-wavelength and multi-viewing angle
            theta = (30,(0,30,60,90),3,0,20,(2.2,9.7,12.)) # 4 viewing angles, 3 lambdas
            image = M.get_image(theta)
            print image.shape
              (4,3,441,441)
        """
        
        vec = list(vector)

        # sub-vectors can be arrays or lists; convert to tuples
        for j,v in enumerate(vec):
            if isinstance(v,N.ndarray):
                vec[j] = tuple(v.tolist())
            elif isinstance(v,list):
                vec[j] = tuple(v)
        
        vec.append(tuple(self.x.tolist()))
        vec.append(tuple(self.y.tolist()))
        
        vec = tuple(vec)
        
        image = self.ip(vec)
        image = image.squeeze()

        if full is True:
            if (2*image.shape[-2] - 1 == image.shape[-1]):
                image = mirror_axis(image)
            else:
                logging.warn("x dimension seems not suitable for mirroring. Try with full=False")
            
        return image


class Image:

    pix = u.pix  #: ``u.pix`` alias, defined for convenience.

    def __init__(self,image,\
                 pixelscale='1 arcsec',distance=None,\
                 peak_pixel_brightness='1 Jy/pix',\
                 pa=0.,\
                 psf=None,psfdict={},\
                 pixelscale_detector=None):

        """From a 2D array instantiate an Image object.

        The instance has members for pixelscale, brightness, unit
        conversions, etc.  The way to specify physical quantities with
        units follows roughly the CASA (Common Astronomy Software
        Applications package) philosophy, i.e. they are strings like
        'value unit', e.g. '1 Jy/pix'.

        Parameters
        ----------
        image : float array (2D)
            Monochromatic relative brightness per pixel. Will be
            scaled to physical values via arguments 'pixelscale' and
            'peak_pixel_brightness' (see below).

        pixelscale : str
            Pixel scale of a single pixel in the model image. See
            docstring of setPixelscale() function.

        distance : str|None
            Physical distance to the source. Only required if
            'pixelscale' was specified in linear (not angular)
            units. See docstring of setPixelscale() function.

        peak_pixel_brightness : str
            Physical brightness value of the highest-valued pixel in
            `image`. The entire image will be scaled such that the
            peak pixel has value peak_pixel_brightness. See docstring
            of setBrightness() function.

        pa : float
            Position angle (in degrees) with respect to North (=0
            deg). If not 0., the image will be rotated by pa (positive
            values rotate North-to-East, i.e. anti-clockwise, negative
            values rotate North-to-West, i.e. clockwise).

        psf : None | 'model' | FITS filename

            If None, the model image will not be convolved with any
            PSF.

            If 'model', then psf_dict must also be provided (see
            there), and the PSF will be modeled with a Gaussian+Airy
            pattern. Reference:

              John W. Hardy "Adaptive Optics for Astronomical
              Telescopes", Oxford Series in Optical and Imaging
              Sciences 1998

            If psf is a string ending it '.fits', it is the path to a
            FITS file containing the image of a PSF (for instance
            generated with the WebbPSF tool:

              http://www.stsci.edu/~mperrin/software/webbpsf.html).

            psf_dict must then also be provided, but with different
            content (see there).

        psf_dict : dictionary
            Only used when psf is not None.

            If psf='model', then psf_dict must contain the keys and
            values for the wavelength in micron, telescope diameter in
            meters, and Strehl ratio, e.g. like this:

              psf_dict = {'wavelength':2.2, 'diameter': 30., 'strehl': 0.8}

            If psf is the path to a FITS file that contains a PSF
            image, then the content of psf_dict must be the keys and
            values for the HDU extension (name or numeric) that
            contains the PSF image (EXT name HDU number), and the name
            of the keyword containing the PSF's image pixel scale
            (in arcsec/pixel). E.g.:

              psf='PSF_MIRI_F1000W.fits'  # from WebbPSF tool
              psf_dict = {'hdukw':1, 'pixelscalekw':'pixelscl'}  # name of HDU and pixelscale keyword

        pixelscale_detector : float | None
            If not None, the (PSF-convolved, if psf not None) model
            image will be resampled such that the pixelscale of the
            final image corresponds to pixelscale_detector. This will
            increase or decrease the number of pixels in the model
            image, but the FOV is preserved. The total flux density
            will also be preserved (you can check this by calling
            self.getTotalFluxDensity()) Typical detector pixelscales
            are e.g.:
              TMT_IRIS = 4 mas/pix
              TMT_MICHI = 11.9 mas/pix.

        Examples
        --------
        .. code-block:: python

            # get raw image (see :class:`ModelCube`:func:`get_image`)
            theta = (30,0,3,0,20,9.7)
            image = M.get_image(theta)

            # init `Image` instance using angular pixel scale
            I = Image(image,pixelscale='1 arcsec',peak_pixel_brightness='3.0 mJy/pix')
            print I.pixelscale; print I.data.max()
              1.0 arcsec
              3.0 mJy / pix

            # init using linear pixel scale and source distance
            I = Image(image,pixelscale='1 AU',distance='1 pc',peak_pixel_brightness='3.0 mJy/pix')
            print I.pixelscale; print I.distance; print I.data.max()
              1.0 arcsec
              1.0 pc
              3.0 mJy / pix

        """
        
        # sanity checks
        self.npix = checkImage(image,returnsize=True)
        
        # PIXELSCALE
        self.setPixelscale(pixelscale=pixelscale,distance=distance)

        # IMAGE DATA
        self.data_raw = image  # keep original array
        self.data = image   # will rescale this array in setBrightness()

        self.setBrightness(peak_pixel_brightness)
        print "AFTER setBrightness, type(self.data.value)", type(self.data.value)
        
        if pa != 0.:
            self.rotate(pa)
        
        if psf is not None:
            print "BEFORE applyPSF, type(self.data.value)", type(self.data.value)
            self.applyPSF(psf,psfdict)
            print "AFTER applyPSF, type(self.data.value)", type(self.data.value)

        if pixelscale_detector is not None:
            print "BEFORE resample, type(self.data.value)", type(self.data.value)
            pixelscale_detector = getQuantity(pixelscale_detector,CUNITS)
            self.resample(self.pixelscale / pixelscale_detector)
            

    def applyPSF(self,psf,psfdict):
    
        if psf == 'model':
            wavelength, diameter, strehl = psfdict['wavelength'], psfdict['diameter'], psfdict['strehl']
            self.psf_image = PSF_modeling.PSF_model(self.data,wavelength=wavelength,diameter=diameter,pxscale=self.pixelscale.to('arcsec').value,strehl=strehl)
                
        elif psf.endswith('.fits'): # PSF model from fits file; must have keyword PIXELSCL

            try:
                hdukw = psfdict['hdukw']
            except KeyError:
                hdukw = 0

            try:
                pixelscalekw = psfdict['pixelscalekw']
            except KeyError:
                pixelscalekw = 'pixelscl'  # WebbPSF names pixelscale 'pixelscl' in their fits files
            
            header = pyfits.getheader(psf,hdukw)
            self.psf_image = pyfits.getdata(psf,hdukw)

            pixelscale_psf = header[pixelscalekw]
            if pixelscale_psf != self.pixelscale:
                self.psf_image_raw = self.psf_image[...]
                self.psf_image, newfactor_ = resampleImage(self.psf_image,pixelscale_psf/self.pixelscale.to('arcsec').value)

        print "BEFORE PSF_conv, type(self.data.value)", type(self.data.value)
        dunit = self.data.unit
        self.data = PSF_modeling.PSF_conv(self.data.value,self.psf_image) * dunit
        print "AFTER PSF_conv, type(self.data.value)", type(self.data.value)
        

    def setPixelscale(self,pixelscale='1 arcsec',distance=None):

        """Set size and area of a pixel from user input.

        This does not touch the brightness per pixel.

        Parameters
        ----------
        pixelscale : str or 'Quantity' instance
            Angular or linear scale of one pixel. See docstring of
            :func:`getValueUnit` for the requirements.

            Recognized angular units: (see ``UNITS_ANGULAR`` in :class:`Image`)

            Recognized linear units: (see ``UNITS_LINEAR`` in :class:`Image`)

            If linear unit, then it is the linear size of a pixel at
            the distance of the source. Then the `distance` argument
            must be also specified (see below).

        distance : str | 'Quantity' instance | None
            If `pixelscale` was specified in linear units, then the
            distance to the source must be not ``None``, but rather an
            argument akin to 'pixelscale' above, i.e. str or 'Quantity' instance.

            Recognized linear units: (see ``UNITS_LINEAR`` in :class:`Image`)

        Examples
        --------
        .. code-block:: python

            I = Image(pixelscale='0.2 arsec')  # definition of an 'Image' instance
            print I.pixelscale; print I.pixelarea
              0.2 arcsec
              0.04 arcsec2 / pix

            I.setPixelscale(pixelscale='1 AU',distance='1 pc')
            print I.pixelscale, I.pixelarea
              1.0 arcsec
              1.0 arcsec2 / pix

        """

        self.pixelscale = getQuantity(pixelscale,CUNITS)

        if self.pixelscale.unit in UNITS_LINEAR:
            try:
                self.distance = getQuantity(distance,UNITS_LINEAR)
            except AttributeError:
                logging.error("Must provide a value for 'distance' argument. Current value is: "+str(distance))
                raise

            self.pixelscale = N.arctan2(self.pixelscale,self.distance).to('arcsec')
            

        self.__computePixelarea()
        self.__computeFOV()


    def setBrightness(self,peak_pixel_brightness):
        
        """Scale the brightness of the image to ``peak_pixel_brightness``.

        Scale the ``self.data`` array such that the max pixel has
        value `peak_pixel_brightness`. This does not touch the
        ``self.pixelsize`` and ``self.pixelarea`` values of the image.

        Parameters
        ----------
        peak_pixel_brightness : str | 'Quantity' instance
            Target brightness-per-pixe of the max pixel in
            ``self.data``. See docstring of :func:`getValueUnit` for
            the requirements.

            Recognized brightness units: (see ``UNITS_BRIGHTNESS`` in :class:`Image`)

        Examples
        --------
        .. code-block:: python

            print I.data_raw.max(); I.data.max()
              1.0
              1.5 mJy / pix

            I.setBrightness(peak_pixel_brightness='3.3 Jy/pix')
            print I.data_raw.max(); I.data.max()
              1.0
              3.3 Jy / pix

        """

        peak_pixel_target, brightness_unit = getValueUnit(peak_pixel_brightness,UNITS_BRIGHTNESS)
        numfactor = (peak_pixel_target / self.data.max())
        self.data = self.data * numfactor * brightness_unit

        
    def getBrightnessInUnits(self,units='Jy/arcsec^2'):

        """Compute the image brightness in the requested 'units'
        (brightness-per-solid-angle).

        Parameters
        ----------
        units : str
            Units of brightness-per-solid-angle, to which the
            self.data array will be converted using self.pixelscale

            The units are quite flexible. An incomplete list of
            possibilities: ``Jy/mas^2, mJy/arcsec^2, uJy/sr,
            Jy/deg^2``, etc.

        Example
        -------
        .. code-block:: python

             print I.pixelscale; print I.data.max(); print I.getBrightnessInUnits('Jy/arcsec^2')
               1.0 arcsec
               3.0 mJy / pix
               0.003 Jy / arcsec2

        """
        
        try:
            print "type(self.data), type(self.pixelarea)", type(self.data), type(self.pixelarea)
            brightness =  (self.data / self.pixelarea).to(units)
        except ValueError:
            logging.error("Use valid brightness-per-solid-angle units, e.g. 'Jy/arcsec^2' or 'mJy/sr', etc.")
            raise

        return brightness
    

    def setFOV(self,fov):

        """From field-of-view `fov`, compute new ``pixelscale`` & new ``pixelarea``.

        ``self.npix`` remains the same, i.e. recomputes
        ``self.pixelscale`` and ``self.pixelarea`` and sets
        ``self.FOV``.

        Parameters
        ----------
        fov : str or 'Quantity' instance
            Target angular size of the field-of-view. See docstring of
            :func:`getValueUnit` for the requirements.

            Recognized angular units: (see ``UNITS_ANGULAR`` in :class:`Image`)

        Examples
        --------
        .. code-block:: python

            I = hypercat.Image(img,pixelscale='1 AU',distance='1 pc',peak_pixel_brightness='4.0 mJy/pix')
            print ('%s\\n'*4) % (I.pixelscale,I.FOV,I.data.max(),I.getBrightnessInUnits('mJy/arcsec^2').max())
              1.0 arcsec
              7.0 arcsec
              4.0 mJy / pix
              4.0 mJy / arcsec2

            I.setFOV(fov='14.0 arcsec')
            print ('%s\\n'*4) % (I.pixelscale,I.FOV,I.data.max(),I.getBrightnessInUnits('mJy/arcsec^2').max())
              2.0 arcsec
              14.0 arcsec
              4.0 mJy / pix
              1.0 mJy / arcsec2

        """

        FOV = getQuantity(fov,UNITS_ANGULAR)
        self.pixelscale = FOV / N.float(self.npix)
        self.__computePixelarea()
        self.__computeFOV()

        
    def embedInFOV(self,fov):

        """Embed ``self.data`` in a new field of view `fov`, or crop to it.

        If ``fov > self.FOV``, embed self.data in a larger array. If
        ``fov < self.FOV``, crop the central pixels of ``self.data``
        to `fov`. Cropping removes the cropped parts of the image
        permanently.

        Parameters
        ----------
        fov : str or 'Quantity' instance
            Angular size of the target field of view (FOV). See
            docstring of :func:`getValueUnit` for the requirements.
            `fov` can be both larger or smaller than the current image
            FOV ``self.FOV``.

        Notes
        -----
        Since the image is pixelated (discretized), embed/crop to the
        nearest odd integer ``npix``, and use the corrected FOV, which is
        not necessarily exactly the specified `fov`.

        Note that ``self.pixelscale`` and ``self.pixelarea`` remain
        unchanged!

        """
        
        FOV = getQuantity(fov,UNITS_ANGULAR)
        factor = (FOV/self.FOV).decompose().value
        newsize_int, newfactor = computeIntCorrections(self.npix,factor)
        cpix = self.npix/2

        newimage = nddata.extract_array(self.data.value,(newsize_int,newsize_int),(cpix,cpix),fill_value=0.)
        self.data = newimage * self.data.unit

        # updates
        self.npix = self.data.shape[0]
        self.__computeFOV()


    def resample(self,resamplingfactor):

        """Increase or decrease the image sampling in x and y by `resamplingfactor`.

        The resampling is performed by third-order spline
        interpolation. The FOV remains constant, but
        ``self.pixelscale`` and ``self.pixelarea`` are adjusted.  The
        brightness-per-pixel is adjusted according to the change in
        pixel area, but total flux density is preserved.  Note that
        consecutive application of :func:`resample` to the same image
        can yield inexact results due to accumulated interpolation
        errors.
        
        Because resampling is performed by spline interpolation, the
        total flux in the image may change slightly. This function
        thus renormalizes the resampled image to the previous total
        flux.

        Parameters
        ----------
        resamplingfactor : float
            The factor by which the the image sampling (number of
            pixels) is increased (if >1) or decreased (if <1). Note
            that `resamplingfactor` may be adjusted because of the
            requirement that ``npix_new`` be integer and odd-valued.

        Examples
        --------
        .. code:: python

            theta = (30,0,3,0,20,9.7) # (sig,i,N0,q,tauv,lambda)
            img = M.get_image(theta)  # raw model image

            # `Image` instance
            I = hypercat.Image(img,pixelscale='1 AU',distance='1 pc',peak_pixel_brightness='1.0 mJy/pix')

            print ", ".join([str(e) for e in (I.npix,I.pixelscale,I.pixelarea,I.FOV,I.data.max(),I.getBrightnessInUnits('mJy/arcsec^2').max())])
              221, 1.0 arcsec, 1.0 arcsec2 / pix, 221.0 arcsec, 1.0 mJy / pix, 1.0 mJy / arcsec2
            print I.getTotalFluxDensity()
              5.35844 Jy  # total flux density

            # new instance, upsampled 2x
            I = hypercat.Image(img,pixelscale='1 AU',distance='1 pc',peak_pixel_brightness='1.0 mJy/pix')
            I.resample(2.0)
            print ", ".join([str(e) for e in (I.npix,I.pixelscale,I.pixelarea,I.FOV,I.data.max(),I.getBrightnessInUnits('mJy/arcsec^2').max())])
              443, 0.49887 arcsec, 0.24887 arcsec2 / pix, 221.0 arcsec, 0.26328 mJy / pix, 1.05787 mJy / arcsec2
            print I.getTotaFluxDensity()
              5.38289 Jy  # total flux density preserved; difference due to interpolation errors

            # new instance, downsampled 0.5x
            I = hypercat.Image(img,pixelscale='1 AU',distance='1 pc',peak_pixel_brightness='1.0 mJy/pix')
            I.resample(0.5)
            print ", ".join([str(e) for e in (I.npix,I.pixelscale,I.pixelarea,I.FOV,I.data.max(),I.getBrightnessInUnits('mJy/arcsec^2').max())])
              111, 1.99099099 arcsec, 3.96405 arcsec2 / pix, 221.0 arcsec, 3.92935 mJy / pix, 0.99125 mJy / arcsec2
            print I.getTotalFluxDensity()
              5.30354 Jy  # still preserved

        """

        newimage, newfactor, self.npix = resampleImage(self.data.value,resamplingfactor,conserve=True)
        self.data = newimage * self.data.unit
        self.setPixelscale(pixelscale=self.pixelscale/newfactor)

        
    def rotate(self,angle,direction='NE',returnimage=False):

        """Rotate ``self.data`` by `angle` degrees from North towards
        `direction` (either East or West).

        See docstring of :func:`rotateImage()`

        """
        
        self.data = rotateImage(self.data.value,angle=angle,direction=direction) * self.data.unit
        logging.info("Rotated image (see self.data) by %g degrees in direction '%s'." % (angle,direction))

        if returnimage:
            return self.data
        
        
    def getTotalFluxDensity(self):

        """Compute the total flux density within the FOV.

        The total flux density is the brightness integral over the solid
        angle of the FOV.

        Returns
        -------
        totalFluxDensity : float
            Total flux density in Jy.

        Example
        -------
        .. code:: python

            total = I.getTotalFluxDensity()
            print total; print total.to('mJy')  # can be easily converted to other units
              0.0050459482673 Jy
              5.0459482673 mJy
        """
        
        aux = self.pixelarea*self.pix
        units = 'Jy/'+aux.unit.to_string()

        return self.getBrightnessInUnits(units).sum() * self.pixelarea * self.pix

    
    def __computeFOV(self):

        self.FOV = self.npix * self.pixelscale
        
        
    def __computePixelarea(self):

        self.pixelarea = (self.pixelscale)**2 / self.pix
        
    
# HIGH-LEVEL HELPER FUNCTIONS

def mirror_axis(cube,axis=-2):

    """Mirror one axis of a cube.

    Parameters
    ----------
    cube : array
        Expected shape of cube: (...,nx,...).

    axis : int
        axis index to be mirrored. Default: -2. Mirroring assummes a
        central columns of elements, i.e. nx is odd.

    Returns
    -------
    newcube : array
        Array with 'axis' dimension mirrored, i.e.  newcube.shape =
        (...,2*nx-1,...).

    """
        
    ndim = cube.ndim
    npix = cube.shape[axis]
    allpads = [(0,0)]*ndim
    allpads[axis] = (npix-1,0)
    
    newcube = N.pad(cube,tuple(allpads),'reflect')

    return newcube
    

def rotateImage(image,angle,direction='NE'):

    """Rotate an image around its central pixel by `angle` degrees.

    The parts of `image` which after rotation are outside the frame,
    will be truncated. Areas in the rotated image which are empty will
    be filled will zeros.

    The rotation is performed using high-quality cubic spline
    interpolation.

    Parameters
    ----------
    image : 2D array
        The square image array to be rotated.

    angle : float or int
        Image will be rotated by that many degrees.

    direction : str
        Rotation direction. Default is 'NE', i.e. from North to East,
        i.e. anti-clockwise, which is astronomical standard. To rotate
        clockwise instead, give direction='NW' (or negate the `angle`
        argument).

    Returns
    -------
    rotimage : 2D array
        The `image`, rotated by `angle` degrees towards `direction`.

    """
    
    checkImage(image,returnsize=False)
    
    if direction == 'NW':
        angle = -angle
    
    rotimage = ndimage.rotate(image,angle,reshape=False)

    return rotimage


def resampleImage(image,resamplingfactor,conserve=True):

    """Resample an image by `resamplingfactor`.

    Parameters
    ----------
    image : 2D array
        The 2D image array to be resampled. npix must be odd-valued.

    resamplingfactor : float
        The factor by which the the image sampling (number of pixels)
        is increased (if >1) or decreased (if <1). Note that
        `resamplingfactor` may be adjusted because of the requirement
        that ``npix_new`` be integer and odd-valued.

    conserve : bool
        If True (default), the resampled image will be renormalized
        such that the total flux between the original and the
        resampled images is preserved.
    
    Returns
    -------
    newimage : 2d array
        The resampled image. Final size ``npix`` is adjusted to the
        nearest odd-valued integer.

    newfactor : float
        The actually used resampling factor. Adjusted, if necessary, by
        the requirement that ``npix_new`` be integer and odd-valued.

    npix : int
        Number of pixels along one axis of ``newimage``.

    """
    
    npix = checkImage(image,returnsize=True)
    total = image.sum()
    newsize_int, newfactor = computeIntCorrections(npix,resamplingfactor)
    newimage = ndimage.zoom(image,newfactor)
    checkImage(newimage,returnsize=False)

    if conserve is True:
        newtotal = newimage.sum()
        newimage *= (total / newtotal)
    
    return newimage, newfactor, npix
    

# LOW-LEVEL HELPER FUNCTIONS
        
def getQuantity(quantity,recognized_units):

    """Split a string 'Value Units' into numerical value and string-units,
    and create an instance of astropy.units.Quantity.

    Parameters
    ----------
    
    quantity : str
       E.g. quantity='1 Jy' would return <Quantity 1.0 Jy>

    recognized_units : tuple
        Tuple of valid units (as strings) for ``quantity``. E.g., for
        a length quantity, recognized units could be
        ('micron','m','pc','AU'), etc.

    Returns
    -------
    quantity : instance
        Instance of astropy.units.Quantity, i.e. a value with units.

    Examples
    --------
    getQuantity('1 m',('micron','m','pc','AU'))
      <Quantity 1.0 m>

    getQuantity('1 W',('micron','m','pc','AU'))
      ValueError: Specified unit 'W' is not recognized. Recognized are: m,AU,pc

    """
    
    cdelt, cunit = getValueUnit(quantity,recognized_units)
    return cdelt * cunit


def getValueUnit(quantity,recognized_units):

    """Split `string` into value and units string.

    Evaluate both and return the numerical value, and the units object.

    Parameters
    ----------
    quantity : str or `Quantity` instance
        If string, its format is `num units`, where `num` is the
        string representation of a numerical value, and `units` is one
        of the recognized units in the sequence of strings provided in
        ``recognized_units``. `num` and `units` are separated by a
        single whitespace. E.g. '1.5 Jy/arcsec^2').

        If instance of `Quantity`, ``quantity.value`` and
        ``quantity.unit`` are the equivalents of `num` and `unit`.

    recognized_units :  seq of strings
        Sequence of strings representing units that the `units` part
        of `string` will be checked against.

    Examples
    --------
    .. code:: python

        recognized_units = ('Jy/arcsec^2','mJy/mas^2','uJy/sr')
        getValueUnit('5.2 Jy/arcsec^2',recognized_units)
          (5.2, Unit("Jy / arcsec2"))

    """

    if isinstance(quantity,u.Quantity):
        value, unit = quantity.value, quantity.unit
        unit = unit.to_string().replace(" ","")
    elif isinstance(quantity,str):
        value, unit = quantity.split()
        unit = unit.replace(" ","")
    else:
        raise AttributeError("Argument 'quantity' is neither string nor instance of 'Quantity'.")

    value = N.float(value)

    if unit not in recognized_units:
        raise ValueError("Specified unit '%s' is not recognized. Recognized are: %s" % (unit,','.join(recognized_units)))

    unit = u.Unit(unit)

    return value, unit


def checkInt(x):
    
    """Check whether x is integer."""
    
    if not isinstance(x,int):
        raise TypeError('x is not integer.')

def checkOdd(x):

    """Check whether x is integer and odd."""
    
    checkInt(x)
        
    if x % 2 == 0:
        raise ValueError('x is not odd.')

def checkImage(image,returnsize=True):

    """Make a few common assertions for ``image``.

    Verifies that ``image`` is 2D, square with shape (npix,npix), and
    that npix is odd-valued. Raises various Exceptions if test fails.

    Parameters
    ----------
    image : 2D array
        The image (2D array) to be tested.

    returnsize : bool
        If ``True``, returns ``image.shape[0]``, i.e. the number of
        pixels along `image`, if all test pass.

    Examples
    --------
    .. code:: python

        # all tests pass, returns nothing
        image = N.ones(11,11)
        check(image,returnsize=False)

        # fails with Exception if image not square
        image = N.ones(11,10)
        check(image,returnsize=False)
          ...
          ValueError: 'image' must be square (nx=ny)

        # fails with Exception if size not odd
        image = N.ones(10,10)
        check(image,returnsize=False)
          'image' size in pixels (nx) must be odd and integer, but is nx =  10
          ...
          ValueError: x is not odd.

        # fails with Exception if image not 2D
        image = N.ones(11,11,3)
        check(image,returnsize=False)
          ...
          ValueError: 'image' must be 2D.

    """

    # image props assertions
    if image.ndim != 2:
        raise ValueError("'image' must be 2D.")

    nx, ny = image.shape
    
    if (nx != ny):
        raise ValueError("'image' must be square (nx=ny)")

    try:
        checkOdd(nx)
    except (TypeError, ValueError):
        print "'image' size in pixels (nx) must be odd and integer, but is nx = ", nx
        raise

    if returnsize:
        return nx


def computeIntCorrections(npix,factor):

    """Compute `newnpix` and `newfactor` such that `npix_new` is the
    nearest odd-valued integer to ``npix*factor``.

    Parameters
    ----------

    npix : int
        Odd-valued integer.

    factor : float
        Multiplicative factor.

    Returns
    -------
    newnpix : int
        Nearest odd integer to ``npix*factor``.

    newfactor : float
        `factor` but corrected such that ``npix*newfactor = newnpix``.

    Examples
    --------
    .. code:: python
    
        computeIntCorrections(3,2.)
          (7, 2.33333)  # factor was corrected

        computeIntCorrections(3,5)
          (15, 5.0)  # no corrections necessary

        computeIntCorrections(15,0.6)
          (9, 0.6)  # can also scale down

        computeIntCorrections(15,0.5)
          (7, 0.46667)  # factor was corrected

    """
    
    checkOdd(npix)
    newnpix = npix*factor
    newnpix = N.int((2*N.floor(newnpix/2)+1))  # rounded up or down to the nearest odd integer
    newfactor = newnpix/float(npix)

    return newnpix, newfactor



def get_Rd(lum,tsub=1500.,outunit='pc'):

    """Get dust sublimation radius Rd from luminosity of source and dust
    sublimation temperature.

    Uses Eq. (1) from Nenkova+2008b.

    Parameters
    ----------
    lum : float
        AGN bolometric luminosity in erg/s, e.g. 1e45.

    tsub : float
        Dust sublimation temperature. Default is 1500K, corresponding
        e.g. to astrophysical silicates.

    outunit : str
        Desired output units of length. The result will be
        automatically converted to these units. Defaults to 'pc'.

    Returns
    -------

    Rd : float
        Dust sublimation radius Rd in units `outunit`.
    """

    Rd = 0.4*N.sqrt(lum/1e45) * (1500./tsub)**2.6 * u.pc

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
    
    linsize = getQuantity(linsize,UNITS_LINEAR)
    distance = getQuantity(distance,UNITS_LINEAR)
    
    angular = N.arctan2(linsize,distance).to(outunit)
    angular_per_linsize = angular / linsize  # e.g. arcsec/pc
    angular_per_pixel = angular / (float(npix)*u.pix)  # e.g. arcsec/pixel
    
    return angular, angular_per_linsize, angular_per_pixel


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


