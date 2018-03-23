#Add pixel scale in header of the ELT pupil image
import astropy.io.fits as fits
ELT_file ='M1Pupil_elt.fits'
ELT = fits.open(ELT_file)
hdr = ELT[0].header
hdr.append(('PIXSCALE', 0.03997), end=True)
hdr.append(('UNITS', 'meters'), end=True)
ELT.writeto('M1Pupil_elt.fits',overwrite=True)
