HYPERCAT
========

Hypercubes of (clumpy) AGN tori

Synopsis
--------

Handle a hypercube of CLUMPY brightness maps and 2D projected dust
maps. Easy-to-use classes and functions are provided to interpolate
images in many dimensions (spanned by the model parameters), extract
monochromatic or multi-wavelength images, as well as rotate images,
zoom in and out, apply PSFs, extract interferometric signals, quantify
morphologies, etc.

Authors
-------

Robert Nikutta [\<robert.nikutta@gmail.com\>](mailto:robert.nikutta@gmail.com), Enrique Lopez-Rodriguez, Kohei Ichikawa

Version
-------

Version fo this document: 2021-06-27

Current version of the hypercat sofware: 0.1.5

License and Attribution
-----------------------

HYPERCAT is open-source software and freely available at
https://github.com/rnikutta/hypercat/ and
https://pypi.org/project/hypercat/ under a permissive [BSD 3-clause
license](./LICENSE)

In short, if you are using in your research any of the HYPERCAT
software or its components, and/or the HYPERCAT model data hypercubes,
and/or telescope pupil images, please cite these two papers:

- *Nikutta, Lopez-Rodriguez, Ichikawa, Levenson, Packham, Hönig,
  Alonso-Herrero, "Hypercubes of AGN Tori (Hypercat) -- I. Models and
  Image Morphology", ApJ (2021, accepted)*

- *Nikutta, Lopez-Rodriguez, Ichikawa, Levenson, Packham, Hönig,
  Alonso-Herrero, "Hypercubes of AGN Tori (Hypercat) -- II. Resolving
  the torus with Extremely Large Telescopes", ApJ (2021, under
  review)*


Minimal install instructions
----------------------------

If you don't mind installing HYPERCAT and its dependencies into your
current environment (real or virtual), simply run:

```
pip install hypercat
```


If you prefer to install HYPERCAT into a fresh new environment without affecting your existing Python installation, you can create a new environment in various ways.

- If you are a user of conda / anaconda / miniconda / astroconda:

```
conda create -n hypercat-env python=3.7.2
conda activate hypercat-env

pip install hypercat
```

- If you are a user of pyenv:

```
pyenv install 3.7.2
. .venv/bin/activate

pip install hypercat
```

HYPERCAT / CLUMPY model images and 2D dust cloud maps
-----------------------------------------------------

Hypercat needs to access the hypercubes of Clumpy images and dust
maps. They can be downloaded as hdf5 files from the link given at
https://www.clumpy.org/images/ (which currently is
ftp://ftp.noao.edu/pub/nikutta/hypercat/).

The software, and the example Jupyter notebooks (see below) will need
to be instructed about the location of the model file(s). The is very
easy to do upon loading the model file; the notebooks have several
examples on how to accomplish this, e.g.

```
import hypercat as hc
fname = 'hypercat_20181031_all.hdf5' # use your local location to the HDF5 model file
cube = hc.ModelCube(fname,hypercube='imgdata')  # use 'imgdata' for brightness maps, and 'clddata' for 2D cloud maps
```

Example Jupyter notebooks
-------------------------

Several Jupyter example notebooks demonstrate some of HYPERCAT's functionality:

- [01-hypercat-basics.ipynb](https://github.com/rnikutta/hypercat/tree/master/examples/01-hypercat-basics.ipynb):
  Loading a model hypercube, generating model images, images at
  multiple wavelengths, images at multiple values of other model
  parameters, accessing cloud maps

- [02-hypercat-astro.ipynb](https://github.com/rnikutta/hypercat/tree/master/examples/02-hypercat-astro.ipynb):
  Adding physical units to images, world coordinate system, field of
  view and pixel scale operations, image rotation / position angle,
  saving to FITS files

- [03-hypercat-singledish.ipynb](https://github.com/rnikutta/hypercat/tree/master/examples/03-hypercat-singledish.ipynb):
  Telescope pupil images (JWST, Keck, GMT, TMT, ELT), simulating
  observations with single-dish telescopes, noisy observations,
  Richardson-Lucy deconvolotuion, detector pixel scale, flux
  preservation, observations at multiple wavelengths

- [04-hypercat-morphology-intro.ipynb](https://github.com/rnikutta/hypercat/tree/master/examples/05-hypercat-morphology-intro.ipynb):
  Introduction to morphological measurements (on 2D Gaussians), image
  centroid, rotation, measuring size of emission features, elongation,
  half-light radius, Gini coefficient

- [05-hypercat-morphology-clumpy.ipynb](https://github.com/rnikutta/hypercat/tree/master/examples/05-hypercat-morphology-clumpy.ipynb):
  Morphology of the HYPERCAT model images; morphological sizes,
  elongation, centroid location; compare morphologies of of emission
  and their underlying dust distributions; from 2D cloud maps to real
  cloud numbers per LOS; photon escape probability along a LOS


User Manual
-----------

For more detailed installation instructions and other usage examples,
please see the HYPERCAT User Manual [User Manual](./docs/manual/) (in
addition to the [example Jupyter notebooks](./examples/) )
