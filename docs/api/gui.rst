Graphical user interface
========================

.. automodule:: hypercatgui.hypercatgui
   :no-members:

The optional Tkinter-based GUI provides an interactive point-and-click
interface to the most common Hypercat workflows without requiring any Python
scripting.

.. contents:: On this page
   :local:
   :depth: 2


Launching the GUI
-----------------

After installation, the GUI can be started from the command line:

.. code-block:: console

   hypercatgui

Or programmatically:

.. code-block:: python

   from hypercatgui.hypercatgui import main
   main()

On first launch a configuration file ``hypercatgui.conf`` is created in the
current directory. It stores the last-used HDF5 file path and model parameters
so the GUI reopens in the same state.


App
---

.. autoclass:: hypercatgui.hypercatgui.App
   :members:
   :special-members: __init__
   :member-order: bysource


BetterSpinbox
-------------

.. autoclass:: hypercatgui.hypercatgui.BetterSpinbox
   :members:
   :special-members: __init__
   :member-order: bysource


Helper functions
----------------

.. autofunction:: hypercatgui.hypercatgui.read_or_create_config
.. autofunction:: hypercatgui.hypercatgui.logmsgbox
.. autofunction:: hypercatgui.hypercatgui.main
