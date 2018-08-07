from __future__ import print_function

from copy import copy
import psf
from units import *
from astropy.coordinates import name_resolve
from utils import *
from imageops import add_noise, measure_snr

import astropy.io.ascii as ascii

__version__ = '20180202' #yyyymmdd
__author__ = 'Robert Nikutta <robert.nikutta@gmail.com>'

__version__ = '20180806' #yyyymmdd
__author__ = 'Enrique Lopez-Rodriguez<enloro@gmail.com>'

"""Utilities for handling the CLUMPY image hypercube.

.. automodule:: instrumsents
"""

class Instrument:

    """Base class for optical observing apparatus of all kinds (e.g. Telescope, Interferometer, etc.)"""

    def __init__(self,telescope='',instrument=''):

        """Instantiate observing instrument.

        Parameters
        ----------
        telescope : str
            Name of telescope, optional. E.g. 'TMT'. Default ''.

        instrument : str
            Name of instrument mounted to telescope, optional. E.g. 'VISIR'. Default ''.

        """

        self.telescope = telescope
        self.instrument = instrument
        self.type = self.__class__.__name__

    def observe(self,sky,**kwargs):

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

        observation = self.__call__(sky,**kwargs)

        return observation

    def pupilpsf(self,wavelength,**kwargs):

        """'Obtain the Pupil image of a telescope and the PSF image
            at a given wavelength.

        Parameters
        ----------
        wavelength : float
             wavelength with units usign atropy

        Returns
        -------
        observation : instance
            Pupil and PSf image with theur pixel scales.

        """

        instrument_pupilpsf = self.__call__(wavelength,**kwargs)

        return instrument_pupilpsf

class Telescope(Instrument):

#TODO:go this route    def __init__(self,psf='model', diameter=None,strehl=None, hdukw=None,    pixelscale_detector=None,telescope='',instrument=''):
    def __init__(self,psfdict={},pixelscale_detector='',telescope='',instrument=''):

        """Simulate the effects of a single-dish telescope with a pixel camera
        attached to it.

        Model a Gaussian+Airy PSF, or load PSF from a FITS file. Apply
        PSF to sky image.

        Resample resulting image to ``pixelscale_detector``, if not ``None``.

        Parameters
        ----------
        psfdict : dict
            A dictionary with several keys. Key ``psf`` is common to
            all configurations, and must be either: ``None``, or
            ``'model'``, or the path to a FITS file that contains an
            image of the PSF to be used.

            - If ``psf=None``, the model image will not be convolved
              with any PSF.

            - If ``psf='model'``, the PSF will be modeled with a
              Gaussian+Airy pattern.

              .. note:: Reference: John W. Hardy "Adaptive Optics for
                Astronomical Telescopes" Oxford Series in Optical and
                Imaging Sciences, 1998

              In this case, ``psfdict`` must also contain the keys
              ``diameter`` and ``strehl``, and their values.  They are
              the diameter of the telescope aperture and the strehl
              ratio. E.g.

              .. code-block:: text

                 psfdict = {'psf':'model','diameter':'30 m','strehl':0.8}

            - If ``psf`` is a string ending with ``'.fits'``, it is
              the path to a FITS file that contains the image of a PSF
              (for instance generated with the WebbPSF tool:
              http://www.stsci.edu/~mperrin/software/webbpsf.html).

              In this case, ``psfdict`` must also have the keys and
              values for the HDU extension (name or numeric) that
              contains the PSF image (EXT name or HDU number), and the
              name of the keyword containing the PSF's image pixel
              scale. E.g.:

              .. code-block:: text

                 # FITS file containing PSF image made with WebbPSF tool, name/number of HDU, pixelscale keyword
                 psfdict = {'psf':'PSF_MIRI_F1000W.fits','hdukw':1,'pixelscalekw':'pixelscl'}

        pixelscale_detector : str | None
            If not ``None``, then the model image will be resampled
            such that the pixelscale of the final image corresponds to
            pixelscale_detector. This will increase or decrease the
            number of pixels in the model image, but the FOV is
            preserved. If ``psf`` is not ``None``, the resampling will
            be performed on the PSF-convolved image.

            The total flux density will also be preserved. You can
            check this by calling :func:`self.getTotalFluxDensity()`.

            Typical detector pixelscales are e.g.:

            TMT_IRIS:  ``pixelscale_detector='4 mas/pix'``

            TMT_MICHI: ``pixelscale_detector='11.9 mas/pix'``

        """

        Instrument.__init__(self,telescope=telescope,instrument=instrument)

        self.psfdict = psfdict
        self.pixelscale_detector = pixelscale_detector

        #if pixelscale_detector is 'Nyquist':
        #    self.pixelscale_detector = 'Nyquist'

        #if pixelscale_detector is not 'Nyquist':
        #    self.pixelscale_detector = getQuantity(self.pixelscale_detector,UNITS['CUNITS'])


    def __call__(self,sky,snr=None):  # __call__ is invoked by Instrument.observe()

        # do all the deeds a single-dish observation does
        image = copy(sky)
        image.telescope = self.telescope
        image.instrument = self.instrument

        if 'psf' in self.psfdict:
            PSF = psf.getPSF(self.psfdict,image)  # instance of 'PSF' class, which itself is instance of 'Image'
            print('PSF: Computed pixelscale from pupil = ',PSF.pixelscale,' [mas/px]' )
            PSF_resample = copy(PSF)
            #PSF with the same FOV of the image
            if PSF.FOV != image.FOV:
                PSF.changeFOV(str(image.FOV))
                PSF_resample.changeFOV(str(image.FOV))
            #PSF with the same pixelscale as the image
            PSF.resample(image.pixelscale)

            #Model-PSF
            if self.psfdict['psf'] == 'model':
                _unit = image.data.unit
                image.data = PSF.convolve(image.data.value) * _unit  # psf image pixels have no units attached
                if self.pixelscale_detector == 'Nyquist':
                    pupil_diameter = self.psfdict['diameter'].value
                    self.pixelscale_detector =  ((206265*image.wave.to(u.m).value/pupil_diameter/2.)*1000) * u.mas
                _unit = image.data.unit
                image.data = PSF.convolve(image.data.value) * _unit  # psf image pixels have no units attached
                image.resample(self.pixelscale_detector)
                PSF_resample.resample(self.pixelscale_detector)

            #Pupil-PSF
            if self.psfdict['psf'] == 'pupil':
                DIR = '/Users/elopezro/Documents/GitHub/hypercat/'
                pupilfile = DIR+'data/pupils.csv'
                pupil_info = ascii.read(pupilfile)
                #Nyqueit Sampling
                if self.pixelscale_detector == 'Nyquist':
                    pupil_diameter = pupil_info['Diameter'][pupil_info['Telescope'] == image.telescope][0]
                    self.pixelscale_detector =  ((206265*image.wave.to(u.m).value/pupil_diameter/2.)*1000) * u.mas
                #user-defined detector pixelscale
                #if self.pixelscale_detector != 'Nyquist':
                _unit = image.data.unit
                image.data = PSF.convolve(image.data.value) * _unit  # psf image pixels have no units attached
                image.resample(self.pixelscale_detector)
                PSF_resample.resample(self.pixelscale_detector)


        # TEST
        if snr is not None:
            "In Telescope.__call__(): adding noise"
            noisy_image, noise_pattern = add_noise(image.data.value,snr)
            print("Measured SNR = ", measure_snr(noisy_image, noise_pattern))
            image.data = noisy_image * image.data.unit
        # END TEST


        #if PSF.FOV != image.FOV:
            #PSF.changeFOV(str(image.FOV))

        wcs = get_wcs(image)
        if wcs is not None:
            image.wcs = wcs

        # return a convenient instance
        if 'psf' in self.psfdict:
            return image, PSF, PSF_resample
        else:
            return image


class PupilPSF(Instrument):

    def __init__(self,psfdict={},pixelscale_detector='',telescope='',instrument=''):


        Instrument.__init__(self,telescope=telescope,instrument=instrument)

        self.psfdict = psfdict
        self.pixelscale_detector = pixelscale_detector

        if pixelscale_detector is 'Nyquist':
            self.pixelscale_detector = 'Nyquist'

        if pixelscale_detector is not 'Nyquist':
            self.pixelscale_detector = getQuantity(self.pixelscale_detector,UNITS['CUNITS'])


    def __call__(self,wavelength):  # __call__ is invoked by Instrument.pupilpsf()
        pupil_fits, image_psf, pixelscale_pupil, pixelscale_psf = psf.getPupil(self.psfdict,wavelength)
        return pupil_fits[0].data, image_psf, pixelscale_pupil, pixelscale_psf




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
