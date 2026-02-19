Logging
=======

.. automodule:: hypercat.loggers
   :no-members:

Hypercat uses the standard :mod:`python:logging` module. On import, a
``StreamHandler`` writing to ``stdout`` is installed with a custom formatter.

The log level defaults to ``INFO``. To suppress informational messages:

.. code-block:: python

   import logging
   logging.getLogger('hypercat').setLevel(logging.WARNING)

To enable debug output:

.. code-block:: python

   import logging
   logging.getLogger('hypercat').setLevel(logging.DEBUG)


LogFormatter
------------

.. autoclass:: hypercat.loggers.LogFormatter
   :members:
   :special-members: __init__
   :member-order: bysource

``INFO``-level records are emitted without a prefix; all other levels
(``WARNING``, ``ERROR``, ``DEBUG``) are prefixed with
``[LEVELNAME]`` followed by a space.
