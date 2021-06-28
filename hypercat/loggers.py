__version__ = '20210617'   #yyymmdd
__author__ = 'Robert Nikutta <robert.nikutta@gmail.com>'

"""Utilities for handling global logging.

.. automodule:: logging
"""

import logging
import sys

# Custom formatter for logger
class LogFormatter(logging.Formatter):

    def __init__(self):
        logging.Formatter.__init__(self)

    def format(self, record):
        # Replace the original format with one customized by logging level
        if record.levelno == logging.INFO:
            s = '{:s}'.format(record.msg)
        else:
            s = '[{:s}] {:s}'.format(record.levelname,record.msg)

        # return formatted msg
        return s

# set up logger
hdlr = logging.StreamHandler(sys.stdout)
hdlr.setFormatter(LogFormatter())
logging.root.addHandler(hdlr)
logging.root.setLevel(logging.INFO)
#logging.root.setLevel(logging.WARNING)
