import sys
sys.path.append("@INVLIB_PYTHONPATH@")

import invlib
from invlib.vector import Vector
from invlib.matrix import Matrix

import numpy as np

A  = 10.0 * np.diag(np.ones(10))
A_ = A.view(Matrix)
v = np.random.normal(size = (10, 1))
v_ = v.view(Vector)

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
