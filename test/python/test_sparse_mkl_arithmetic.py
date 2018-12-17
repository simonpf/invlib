import sys
sys.path.append("@INVLIB_PYTHONPATH@")

import invlib
from invlib.mkl    import MklSparseCsr
from invlib.matrix import Matrix
from invlib.vector import Vector

import numpy as np
import scipy as sp
import scipy.sparse

A  = sp.sparse.diags(np.ones(10))
A_ = MklSparseCsr(A)
v  = np.random.normal(size = (10, 1))
v_ = v.view(Vector)
A_.multiply(v_)

def test_matrix_multiplication():
    m = np.random.randint(10, 100)
    n = np.random.randint(10, 100)
    k = np.random.randint(10, 100)

    A = np.random.normal(size = (m, k)).view(Matrix)
    B = np.random.normal(size = (k, n)).view(Matrix)
    C = A.multiply(B)

    assert(np.all(np.isclose(C, np.dot(A, B))))

    u = np.random.normal(size = (k, 1)).view(Vector)
    v = A.multiply(u)
    assert(np.all(np.isclose(v, np.dot(A, u))))
