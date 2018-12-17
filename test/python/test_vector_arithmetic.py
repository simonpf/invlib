import sys
sys.path.append("@INVLIB_PYTHONPATH@")

import invlib
from invlib.vector import Vector

import numpy as np


a  = np.arange(10, dtype = np.float)
a_ = a.view(Vector)
