#Add pixel scale in header of the GMT pupil image
import astropy.io.fits as fits
ELT_file ='GMT_MASK.fits'
ELT = fits.open(ELT_file)
hdr = ELT[0].header
hdr.append(('PIXSCALE', 0.052174), end=True)
hdr.append(('UNITS', 'meters'), end=True)
ELT.writeto('GMT_MASK.fits',overwrite=True)
