#Add pixel scale in header of the TMT pupil image
import astropy.io.fits as fits
import os as os

file ='TMT_Pupil_Amplitude_Gray_Pixel_Approximated_With_Obscuration.fits'
ori_fits = fits.open(file)

os.system('rm TMT_Pupil_Amplitude_Gray_Pixel_Approximated_With_Obscuration.fits')
new_hdul = fits.HDUList()

#modify header
hdr = ori_fits[0].header

#Odd array
ima = ori_fits[0].data

ima = ima[:-1,:-1]

#save file
fits.append('TMT_Pupil_Amplitude_Gray_Pixel_Approximated_With_Obscuration.fits', ima, hdr)
