# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import version

# Make the package importable so autodoc can introspect it
sys.path.insert(0, os.path.abspath("../src/"))

# -- Project information -----------------------------------------------------
project = "hypercat"
copyright = "2025, Robert Nikutta, Enrique Lopez-Rodriguez"
author = "Robert Nikutta, Enrique Lopez-Rodriguez"
release = version("hypercat")
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

# autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
    "special-members": "__call__, __init__",
    "exclude-members": "__weakref__, __dict__, __module__",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
# Mock C-extension / GUI modules unavailable in headless build environments
# (ReadTheDocs, CI without a display, etc.)
autodoc_mock_imports = [
    "tkinter",
    "urwid",
    "rpy2",
    "urwid.curses_display",
    "matplotlib.backends.backend_tkagg",
]

# Napoleon settings (Google + NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_rtype = True

# intersphinx: link to upstream docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "astropy": ("https://docs.astropy.org/en/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "h5py": ("https://docs.h5py.org/en/stable", None),
}

# sphinx-copybutton
copybutton_exclude = ".linenos, .gp"
copybutton_prompt_text = ">>> "
copybutton_selector = "div:not(.no-copybutton) > div.highlight > pre"

templates_path = []
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

master_doc = "index"
html_show_sourcelink = False
add_module_names = False

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "titles_only": False,
}
