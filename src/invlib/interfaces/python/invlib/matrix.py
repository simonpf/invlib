"""
invlib.matrix
=============

The :code:`invlib.vector` module contains the :code:`Vector` class that provides
efficient linear algebra operations for dense matrices.
"""
import sys
sys.path.append("@LIBINVLIB_PATH@")

import numpy as np
import ctypes as c

from invlib.vector import Vector
from invlib.api    import resolve_precision, get_stride, get_c_type, \
    buffer_from_memory

################################################################################
# The Matrix class
################################################################################

class Matrix:
    """
    The :code:`Matrix` class implements dense matrices. The :code:`Matrix` class
    inherits from :code:`numpy.ndarray` and can therefore be constructed from
    every 2D :code:`ndarray` object.
    """

    @static_method
    def from_invlib_pointer(ptr, dtype):

    def _check_precision(matrix):
        """
        Check that precision of the provided matrix matches the types supported
        by invlib.

        Arguments:

            vector(:code:`np.ndarray`): The array of which to check the memory
                layout.
        """
        dtype = matrix.dtype
        if not dtype in [np.float32, np.float64]:
            raise ValueError("invlib.matrix objects can only be created from"
                             " matrices of type float32 or float64")

    def _init_dense(self, m):
        _check_precision(m)
        pass

    def _init_sparse_csc(self, m):
        _check_precision(m)
        pass

    def _init_sparse_csr(self, m):
        _check_precision(m)
        pass

    def __init__(self, m, format = None):

        # Try to deduce format from type of m.
        if format is None:
            if type(m) == np.ndarray:
                self._init_dense(m)
            elif type(m) == sp.sparse.csc_matrix:
                self._init_sparse_csc(m)
            elif type(m) == sp.sparse.csr_matrix:
                self._init_sparse_csr(m)
            else:
                raise Exception("numpy.ndarray or scipy sprase matrix required"\
                                "to create matrix.")
        else:
            if format == "dense":
                self._init_dense(np.asarray(m))
            elif format == "sparse_csc":
                try:
                    self._init_sparse_csc(m.to_csc)
                except:
                    raise ValueError("To create a matrix in sparse CSC format "\
                                     "the provided matrix must be convertible "\
                                     "to a scipy.sparse.csc_matrix matrix.")
            elif format == "sparse_csr":
                try:
                    self._init_sparse_csr(m.to_csr)
                except:
                    raise ValueError("To create a matrix in sparse CSR format "\
                                     "the provided matrix must be convertible "\
                                     "to a scipy.sparse.csr_matrix matrix.")


    def _check_memory_layout(matrix):
        """
        Check that the memory layout of a :code:`numpy.ndarray` is c-style
        and contiguous so that it can be used without copying inside
        invlib.

        Arguments:

            vector(:code:`np.ndarray`): The array of which to check the memory
                layout.
        """
        if not matrix.flags.c_contiguous:
            raise Exception("Only vectors that are contiguous and stored in "\
                            "C-order can be passed to invlib directly.")
        dtype = matrix.dtype
        m = matrix.shape[1]
        stride = get_stride(dtype)
        if not matrix.strides == (stride * m, stride):
            raise Exception("Only Matrix with a stride of 1 passed to invlib "\
                            " directly.")

    def _check_precision(matrix):
        """
        Check that precision of a :code:`numpy.ndarray` matches the types supported
        by invlib.

        Arguments:

            vector(:code:`np.ndarray`): The array of which to check the memory
                layout.
        """
        dtype = matrix.dtype
        if not dtype in [np.float32, np.float64]:
            raise ValueError("invlib.matrix objects can only be created from"
                             "numpy.ndarrays of type float32 or float64")

    def __new__(self, invlib_ptr, dtype):

        f_rows = resolve_precision("matrix_rows", dtype)
        m = f_rows(invlib_ptr)
        f_cols = resolve_precision("matrix_cols", dtype)
        n = f_cols(invlib_ptr)

        shape   = (m, n)
        stride = get_stride(dtype)
        strides = (stride * n, stride)


        b = resolve_precision("matrix_get_data_pointer", dtype)(invlib_ptr)
        ctype = get_c_type(dtype)
        b   = c.cast(b, c.POINTER(ctype))
        arr = np.ctypeslib.as_array(b, shape = (m, n))
        obj = super(Matrix, self).__new__(Matrix, shape, dtype, arr.data, 0, strides, 'C')

        self.invlib_ptr = invlib_ptr

        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return None

        if not obj.dtype in [np.float32, np.float64]:
            return np.array(obj)

        Matrix._check_precision(obj)
        Matrix._check_memory_layout(obj)
        f = resolve_precision("create_matrix", obj.dtype)
        m, n = obj.shape
        self.invlib_ptr = f(obj.ctypes.data, m, n, False)

    def multiply(self, b):
        """
        Multiply this matrix from the right by another matrix or vector.

        Arguments:

            b(:code:`invlib.matrix` or :code:`invlib.vector`): The matrix
                or vector to multiply this matrix with.

        Returns:

            The matrix or vector that results from multiplying this matrix
            from the right with another matrix or vector, respectively.

        """
        if isinstance(b, Matrix):
            f   = resolve_precision("matrix_matrix_multiply", self.dtype)
            ptr = f(self.invlib_ptr, b.invlib_ptr)
            return Matrix(ptr, self.dtype)

        elif isinstance(b, Vector):
            f   = resolve_precision("matrix_vector_multiply", self.dtype)
            ptr = f(self.invlib_ptr, b.invlib_ptr)
            return Vector(ptr, self.dtype)

        raise ValueError("Argument b must be of type invlib.matrix.Matrix or "
                         "invlib.vector.Vector.")


    def transpose_multiply(self, b):
        """
        Multiply the transpose of this matrix from the right by another matrix
        or vector.

        Arguments:

            b(:code:`invlib.matrix` or :code:`invlib.vector`): The matrix
                or vector to multiply the transpose of this matrix with.

        Returns:

            The matrix or vector that results from multiplying the transpose
            of this matrix from the right with another matrix or vector,
            respectively.
        """
        if isinstance(b, Vector):
            f   = resolve_precision("matrix_vector_multiply_transpose", self.dtype)
            ptr = f(self.invlib_ptr, b.invlib_ptr)
            return Vector(ptr, self.dtype)

        raise ValueError("Argument b must be of type invlib.vector.Vector.")
