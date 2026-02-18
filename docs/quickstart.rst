Quick start
===========

This page shows the most common workflows in Hypercat. All code examples
assume you have already downloaded the CLUMPY HDF5 data file (see
:doc:`installation`).

Loading the model cube
----------------------

.. code-block:: python

   import hypercat as hc

   # Memory-map the full hypercube; individual slabs are loaded on demand
   cube = hc.ModelCube('hypercat_20200830_all.hdf5')

   # Inspect the parameter sampling
   cube.print_sampling()

The ``print_sampling()`` call displays the parameter names, their sampled
values, and the shape of the full hypercube.

Getting an image by interpolation
----------------------------------

Pass a parameter vector ``(i, σ, Y, N₀, q, τᵥ, λ)`` to the cube:

.. code-block:: python

   # (inclination=30°, sigma=30°, Y=10, N0=5, q=1, tau_v=40, lambda=10 micron)
   image_array = cube((30., 30., 10., 5., 1., 40., 10.))

   print(image_array.shape)  # e.g. (241, 481) for the half-image

The returned array is the right half of a symmetric image; pass ``full=True``
(the default) to mirror it into the full square:

.. code-block:: python

   image_array = cube((30., 30., 10., 5., 1., 40., 10.), full=True)
   print(image_array.shape)  # e.g. (241, 241)

Attaching physical scales with Source
--------------------------------------

:class:`~hypercat.hypercat.Source` converts the dimensionless model image into
physical brightness units using the AGN luminosity and distance:

.. code-block:: python

   src = hc.Source(
       cube,
       luminosity='1e45 erg/s',
       distance='14.4 Mpc',       # distance to e.g. NGC 1068
       tsub='1500 K',             # dust sublimation temperature
       pa='90 deg',               # position angle East of North
       objectname='NGC 1068',     # optional: resolves coordinates via Vizier
   )

   # Call src with the same parameter vector; total_flux_density sets the scale
   img = src(
       (30., 30., 10., 5., 1., 40., 10.),
       total_flux_density='0.5 Jy',
       brightness_units='Jy/arcsec^2',
   )

   print(img.pixelscale)   # angular size of one pixel
   print(img.FOV)          # total field of view

Single-dish observation (PSF convolution)
------------------------------------------

.. code-block:: python

   telescope = hc.Imaging(psfdict={
       'psf': 'model',
       'diameter': '8.2 m',        # e.g. a VLT unit telescope
       'wavelength': '10 micron',
       'strehl': 0.9,
       'pixelscale_detector': '53 mas',
   })

   observed, psf, psf_resampled = telescope(img)

Interferometric visibilities
-----------------------------

.. code-block:: python

   interferometer = hc.Interferometry()

   # Set UV baselines: list of (u, v) pairs in metres
   interferometer.set_uv_points([(0., 60.), (30., 90.), (-45., 30.)])

   corrflux, BL, fftscale = interferometer(img)
   print(corrflux)   # complex correlated flux at each UV point

Getting an SED
--------------

.. code-block:: python

   # Fix all parameters except wavelength
   wave, sed = cube.get_sed(vec=(30., 30., 10., 5., 1., 40.))

   import matplotlib.pyplot as plt
   plt.loglog(wave, sed)
   plt.xlabel('Wavelength (micron)')
   plt.ylabel('Relative flux')
   plt.show()

Plotting
--------

:func:`~hypercat.plotting.multiplot` produces multi-panel figures from a
sequence of :class:`~hypercat.imageops.Image` instances:

.. code-block:: python

   images = [src((i, 30., 10., 5., 1., 40., 10.), total_flux_density='1 Jy')
             for i in (0., 30., 60., 90.)]

   fig, axes = hc.multiplot(
       images,
       titles=[f'i = {i}°' for i in (0, 30, 60, 90)],
       units='Jy/arcsec^2',
       colorbars=True,
   )
   plt.show()
