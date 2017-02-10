__version__ = '20170202'   #yyymmdd
__author__ = 'Robert Nikutta <robert.nikutta@gmail.com>'

"""Utilities for handling the CLUMPY image hypercube.

.. automodule:: instruments
"""

from copy import copy
import psf
from units import *


class Instrument:

    """Base class for instruments of all kind (e.g. Telescope, Interferometer, etc."""
    
    def __init__(self,name=''):
        self.name = name
        self.type = self.__class__.__name__

    def observe(self,sky):

        """'Observe' the sky with an instrument, i.e. return an image of the
           sky processed by the instrument (e.g. PSF applied etc.).

        Parameters
        ----------
        sky : instance
            Instance of ``Image`` class, which is the model image of
            the sky, with proper units etc., but not yet processed by
            an instrument (e.g. before PSF convolution etc.)

        Returns
        -------
        observation : instance
            A copy of the ``sky`` instance, but processed by the given
            instrument (e.g. PSF-convolved, resampled to detector
            pixelscale, etc.)

        """
        
        observation = self.__call__(sky)
        
        return observation


class Telescope(Instrument):

    
    def __init__(self,psfdict={},pixelscale_detector=None,name=''):

        """Instrument to simulate the effects of a single-dish telescope with
        a pixel camera attached to it.

        Model a Gaussian+Airy PSF, or load PSF from a FITS file. Apply
        PSF to sky image.
        
        Resample resulting image to ``pixelscale_detector``.

        Parameters
        ----------

        psfdict : dict

            A dictionary with several keys. Key 'psf' is common to all
            configurations, and must be either 'model' oder the path
            to a FITS file that contains an image of the PSF to be
            used.

            If 'model'


        psf : None | 'model' | FITS filename

            If None, the model image will not be convolved with any
            PSF.

            If 'model', then psfdict must also be provided (see
            there), and the PSF will be modeled with a Gaussian+Airy
            pattern. Reference:

              John W. Hardy "Adaptive Optics for Astronomical
              Telescopes", Oxford Series in Optical and Imaging
              Sciences 1998

            If psf is a string ending it '.fits', it is the path to a
            FITS file containing the image of a PSF (for instance
            generated with the WebbPSF tool:

              http://www.stsci.edu/~mperrin/software/webbpsf.html).

            psfdict must then also be provided, but with different
            content (see there).

        psfdict : dictionary
            Only used when psf is not None.

            If psf='model', then psfdict must contain the keys and
            values for the wavelength in micron, telescope diameter in
            meters, and Strehl ratio, e.g. like this:

              psfdict = {'wavelength':2.2, 'diameter': 30., 'strehl': 0.8}

            If psf is the path to a FITS file that contains a PSF
            image, then the content of psfdict must be the keys and
            values for the HDU extension (name or numeric) that
            contains the PSF image (EXT name HDU number), and the name
            of the keyword containing the PSF's image pixel scale
            (in arcsec/pixel). E.g.:

              psf='PSF_MIRI_F1000W.fits'  # from WebbPSF tool
              psfdict = {'hdukw':1, 'pixelscalekw':'pixelscl'}  # name of HDU and pixelscale keyword

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

        """
        
        Instrument.__init__(self,name)
        
        self.psfdict = psfdict
        self.pixelscale_detector = pixelscale_detector
        if pixelscale_detector is not None:
            self.pixelscale_detector = getQuantity(self.pixelscale_detector,CUNITS)
        self.name = name
        

    def __call__(self,sky):  # __call__ is invoked by Instrument.observe()
        
        # do all the deeds a single-dish observation does
        image = copy(sky)
        
        if 'psf' in self.psfdict:
            PSF = psf.getPSF(self.psfdict,image)  # instance of 'PSF' class, which itself is instance of 'Image'
            if self.psfdict['psf'] != 'model':
                PSF.rotate(str(image.pa))
            
            _unit = image.data.unit
            image.data = PSF.convolve(image.data.value) * _unit  # psf image pixels have no units attached

        if self.pixelscale_detector is not None:
            image.resample(self.pixelscale_detector)
            # TODO: also resample self.psf image?

        if PSF.FOV != image.FOV:
            PSF.changeFOV(str(image.FOV))
        
        # return a convenient instance
        if 'psf' in self.psfdict:
            return image, PSF
        else:
            return image

        
class Interferometer(Instrument):

    """Instrument to simulate an optical/IR interferometer.

    Given brightness distribution ``sky``, perform FFT, and enable
    extraction of correlated fluxes, and of visibilities, given an
    input baseline (BL length and PA).
    """
    
    def __init__(self):
        Instrument.__init__(self)
        
    def __call__(self):
        pass # do all the deeds an interferometer does
