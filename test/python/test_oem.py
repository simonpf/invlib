import sys
sys.path.append("@INVLIB_PYTHONPATH@")

from invlib.oem    import OEM
from invlib.mkl    import MklSparseCsr
from invlib.matrix import Matrix
from invlib.vector import Vector

import numpy as np
import scipy as sp
import scipy.sparse

m = 10
n = 10

K  = MklSparseCsr(sp.sparse.diags(np.ones(m), format = "csr"))
xa = np.zeros((n, 1)).view(Vector)
y  = np.ones((m, 1)).view(Vector)

oem = OEM(K, K, K, xa, y)
