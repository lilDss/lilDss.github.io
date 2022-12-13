# from . import patch
# from .data.constant import *

import sys
import logging

logger = logging.getLogger("")
# logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)-10s %(filenames)s[line:%(lineno)d] %(message)s", datefmt="%H:%M:%S")

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

__version__ = "0.1.0"

#TODO: setup main