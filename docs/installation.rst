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

Installing via conda-forge
--------------------------

Hypercat is available from the `conda-forge <https://conda-forge.org>`_ channel.
This is the recommended route if you manage your Python environment with
`conda <https://docs.conda.io>`_ or `mamba <https://mamba.readthedocs.io>`_,
because conda-forge resolves binary dependencies (HDF5, NumPy, SciPy) from a
single consistent channel.

.. code-block:: console

   conda install -c conda-forge hypercat

With `mamba <https://mamba.readthedocs.io>`_ (faster solver):

.. code-block:: console

   mamba install -c conda-forge hypercat

To create a fresh environment and install Hypercat in one step:

.. code-block:: console

   conda create -n hypercat -c conda-forge python=3.11 hypercat
   conda activate hypercat

Installing with Docker
----------------------

A ``Dockerfile`` is provided in the repository root. It builds a self-contained
image with Hypercat and JupyterLab, suitable for exploratory analysis without
touching the host Python environment.

**Build the image**

.. code-block:: console

   git clone https://github.com/rnikutta/hypercat.git
   cd hypercat
   docker build -t hypercat .

**Start a JupyterLab session**

Mount the directory that holds your CLUMPY HDF5 data files to ``/data`` inside
the container, and optionally mount a local working directory to ``/work``:

.. code-block:: console

   docker run --rm -p 8888:8888 \
       -v /path/to/clumpy/data:/data \
       -v $(pwd):/work \
       hypercat

Open ``http://localhost:8888`` in your browser. Inside notebooks the data files
are accessible under ``/data/``.

**Run a one-off script**

.. code-block:: console

   docker run --rm \
       -v /path/to/clumpy/data:/data \
       -v $(pwd):/work \
       hypercat python /work/myscript.py

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
