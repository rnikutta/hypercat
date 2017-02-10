__version__ = '20170123'   #yyymmdd
__author__ = 'Robert Nikutta <robert.nikutta@gmail.com>'

"""Utilities for handling global logging.

.. automodule:: logging
"""

import logging
import sys

# Custom formatter for logger
class LogFormatter(logging.Formatter):

    def __init__(self, fmt="%(levelno)s: %(msg)s"):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):

        # Replace the original format with one customized by logging level
        if record.levelno == logging.INFO:
            self._fmt = '%(message)s'
        else:
            self._fmt = '%(levelname)s: %(message)s'

        # return formatted msg
        return logging.Formatter.format(self, record)

# set up logger
hdlr = logging.StreamHandler(sys.stdout)
hdlr.setFormatter(LogFormatter())
logging.root.addHandler(hdlr)
logging.root.setLevel(logging.INFO)
