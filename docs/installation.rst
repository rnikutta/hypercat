Installation
============

Requirements
------------

Hypercat requires **Python 3.10** or later. The main runtime dependencies are:

* `NumPy <https://numpy.org>`_ ≥ 1.24
* `SciPy <https://scipy.org>`_
* `Astropy <https://www.astropy.org>`_
* `h5py <https://www.h5py.org>`_
* `matplotlib <https://matplotlib.org>`_
* `astroquery <https://astroquery.readthedocs.io>`_ (for WCS / source-name resolution)

The optional graphical interface additionally requires `tkinter` (usually
bundled with CPython) and `urwid` (for the terminal-based sub-cube selection
dialog).

Installing from PyPI
--------------------

.. code-block:: console

   pip install hypercat

Installing for development
--------------------------

Clone the repository and install in editable mode with the developer extras:

.. code-block:: console

   git clone https://github.com/rnikutta/hypercat.git
   cd hypercat
   pip install -e ".[dev]"
   pre-commit install

The ``[dev]`` extra pulls in testing, linting, and documentation tools.

Building the documentation locally
-----------------------------------

After installing Hypercat, install the documentation dependencies and run
Sphinx from inside the ``docs/`` directory:

.. code-block:: console

   pip install sphinx sphinx-rtd-theme sphinx-copybutton
   cd docs
   make html

The output will be placed in ``docs/_build/html/``.

Data files
----------

Hypercat itself is a small library. The model data (the CLUMPY image
hypercubes, in HDF5 format) are distributed separately because of their size
(tens to hundreds of GB). Follow the instructions on the
`Hypercat GitHub page <https://github.com/rnikutta/hypercat>`_ to obtain the
data files and place them where :class:`~hypercat.hypercat.ModelCube` can
find them.
