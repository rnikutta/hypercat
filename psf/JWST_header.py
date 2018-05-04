#Add pixel scale in header of the JWST pupil image
#Odd pixel numbers in 2D array
import astropy.io.fits as fits
import os as os

file ='jwst_pupil_revW_npix1024.fits'
ori_fits = fits.open(file)

os.system('rm JWST_PupilImage.fits')
new_hdul = fits.HDUList()

#Header
hdr = ori_fits[0].header
pxscale = hdr['PUPLSCAL']
hdr.append(('PIXSCALE', pxscale), end=True)
hdr.append(('UNITS', 'meters'), end=True)

#FITS
ima = ori_fits[0].data

ima = ima[:-1,:-1]

fits.append('JWST_PupilImage.fits', ima, hdr)

