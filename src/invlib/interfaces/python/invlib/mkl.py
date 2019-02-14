import sys
sys.path.append("@LIBINVLIB_PATH@")

import numpy  as np
import scipy  as sp
import ctypes as c

import scipy.sparse


from invlib.vector import Vector
from invlib.api    import resolve_precision, get_stride, get_c_type, \
    buffer_from_memory

class MklSparseCsr(sp.sparse.csr_matrix):

    def __init__(self, m):
        super().__init__(m)
        f = resolve_precision("create_sparse_mkl_csr", m.dtype)

        print(self.indptr.strides, self.indices.strides, self.data.strides)
        rows, cols = m.shape
        nnz        = m.nnz
        self.invlib_ptr = f(rows, cols, nnz,
                            self.indptr.ctypes.data,
                            self.indices.ctypes.data,
                            self.data.ctypes.data)

    def multiply(self, b):
        if isinstance(b, Vector):
            f   = resolve_precision("sparse_mkl_csr_multiply", self.dtype)
            ptr = f(self.invlib_ptr, b.invlib_ptr)
            return Vector(ptr, self.dtype)

        raise ValueError("Argument b must be of type invlib.vector.Vector.")

    def transpose_multiply(self, b):
        if isinstance(b, Vector):
            f   = resolve_precision("sparse_mkl_csr_transpose_multiply", self.dtype)
            ptr = f(self.invlib_ptr, b.invlib_ptr)
            return Vector(ptr, self.dtype)

        raise ValueError("Argument b must be of type invlib.vector.Vector.")

class MklSparseCsc(sp.sparse.csc_matrix):

    def __init__(self, m):
        super().__init__(m)
        f = resolve_precision("create_sparse_mkl_csc", m.dtype)

        print(self.indptr.strides, self.indices.strides, self.data.strides)
        rows, cols = m.shape
        nnz        = m.nnz
        self.invlib_ptr = f(rows, cols, nnz,
                            self.indptr.ctypes.data,
                            self.indices.ctypes.data,
                            self.data.ctypes.data)

    def multiply(self, b):
        if isinstance(b, Vector):
            f   = resolve_precision("sparse_mkl_csc_multiply", self.dtype)
            ptr = f(self.invlib_ptr, b.invlib_ptr)
            return Vector(ptr, self.dtype)

        raise ValueError("Argument b must be of type invlib.vector.Vector.")

    def transpose_multiply(self, b):
        if isinstance(b, Vector):
            f   = resolve_precision("sparse_mkl_csc_transpose_multiply", self.dtype)
            ptr = f(self.invlib_ptr, b.invlib_ptr)
            return Vector(ptr, self.dtype)

        raise ValueError("Argument b must be of type invlib.vector.Vector.")
