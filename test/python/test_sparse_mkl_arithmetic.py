import sys
sys.path.append("@INVLIB_PYTHONPATH@")

import invlib
from invlib.mkl    import MklSparseCsr, MklSparseCsc
from invlib.matrix import Matrix
from invlib.vector import Vector

import numpy as np
import scipy as sp
import scipy.sparse

A  = sp.sparse.diags(np.ones(10), format = "csc")
A_ = MklSparseCsc(A)
v  = np.random.normal(size = (10, 1))
v_ = v.view(Vector)
A_.multiply(v_)

def test_matrix_multiplication_csr():

    m = np.random.randint(10, 100)
    n = np.random.randint(10, 100)

    v = np.random.normal(size = (m, 1))
    w = np.random.normal(size = (n, 1))

    A = sp.sparse.csr_matrix((m, n))
    for i in range(max(m, n)):
        j = np.random.randint(0, m)
        k = np.random.randint(0, n)
        A[j, k] = np.random.normal()

    A_ = MklSparseCsr(A)

    u  = A.dot(w)
    u_ = A_.multiply(w.view(Vector))

    assert(np.all(np.isclose(u, u_)))

    u  = A.T.dot(v)
    u_ = A_.transpose_multiply(v.view(Vector))

    assert(np.all(np.isclose(u, u_)))

def test_matrix_multiplication_csc():

    m = np.random.randint(10, 100)
    n = np.random.randint(10, 100)

    v = np.random.normal(size = (m, 1))
    w = np.random.normal(size = (n, 1))

    A = sp.sparse.csc_matrix((m, n))
    for i in range(max(m, n)):
        j = np.random.randint(0, m)
        k = np.random.randint(0, n)
        A[j, k] = np.random.normal()

    A_ = MklSparseCsc(A)

    u  = A.dot(w)
    u_ = A_.multiply(w.view(Vector))

    assert(np.all(np.isclose(u, u_)))

    u  = A.T.dot(v)
    u_ = A_.transpose_multiply(v.view(Vector))

    assert(np.all(np.isclose(u, u_)))

m = np.random.randint(10, 100)
n = np.random.randint(10, 100)

v = np.random.normal(size = (m, 1))
w = np.random.normal(size = (n, 1))

A = sp.sparse.csc_matrix((m, n))
for i in range(max(m, n)):
    j = np.random.randint(0, m)
    k = np.random.randint(0, n)
    A[j, k] = np.random.normal()

A_ = MklSparseCsc(A)

u  = A.dot(w)
u_ = A_.multiply(w.view(Vector))

assert(np.all(np.isclose(u, u_)))

u  = A.T.dot(v)
u_ = A_.transpose_multiply(v.view(Vector))

assert(np.all(np.isclose(u, u_)))
