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

    @staticmethod
    def matrix_info(ptr, dtype):

        f_rows = resolve_precision("matrix_rows", dtype)
        m = f_rows(ptr)

        f_cols = resolve_precision("matrix_cols", dtype)
        n = f_cols(ptr)

        f_nnz = resolve_precision("matrix_non_zeros", dtype)
        nnz = f_cols(ptr)

        f_format = resolve_precision("matrix_format", dtype)
        fmt = f_format(ptr)

        f_element_ptr = resolve_precision("matrix_element_pointer", dtype)
        element_ptr = f_element_ptr(ptr)

        if fmt > 0:
            f_index_ptr = resolve_precision("matrix_index_pointer", dtype)
            index_ptr   = f_index_ptr(ptr)
            f_start_ptr = resolve_precision("matrix_start_pointer", dtype)
            start_ptr   = f_start_ptr(ptr)
        else:
            index_ptr = None
            start_ptr = None
        return m, n, nnz, fmt, element_ptr, start_ptr, index_ptr

    @staticmethod
    def from_invlib_pointer(ptr, dtype):
        m, n, nnz, fmt, element_ptr, start_ptr, index_ptr = matrix_infor(ptr,
                                                                         dtype)
        # Dense format
        if fmt == 0:
            if dtype == np.float32:
                ts = "4"
            else:
                ts = "8"

            ai = {"shape" : (m,n ),
                  "typestr" : "|f" + ts,
                  "version" : 3}
            ptr.__array_interface__ = ai
            m = np.asarray(ptr)
        # CSC sparse format
        elif fmt == 1:
            m = sp.sparse.csc_matrix((element_ptr, (index_ptr, start_ptr)),
                                     shape = (m, n))


        # CSR sparse format
        elif fmt == 2:
            m = sp.sparse.csr_matrix((element_ptr, (index_ptr, start_ptr)),
                                     shape = (m, n))
        else:
            raise Exception("Currently only dense, CSC and CSR formats are "
                            " supported.")
        return Matrix(m)


    @staticmethod
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

    @staticmethod
    def _get_data_ptrs(matrix):
        if isinstance(matrix, np.ndarray):
            return [matrix.ctypes.data]
        elif isinstance(matrix, sp.sparse.csc_matrix):
            return [matrix.data.ctypes.data]
        elif isinstance(matrix, sp.sparse.csr_matrix):
            return [matrix.data.ctypes.data]
        else:
            raise ValueError("Currently only dense matrices or sparse matrices "
                             "in CSC or CSR format are supported.")

    @staticmethod
    def _get_index_ptrs(matrix):
        if isinstance(matrix, np.ndarray):
            return []
        elif isinstance(matrix, sp.sparse.csc_matrix):
            return [matrix.indices, matrix.indptr]
        elif isinstance(matrix, sp.sparse.csr_matrix):
            return [matrix.indices, matrix.indptr]
        else:
            raise ValueError("Currently only dense matrices or sparse matrices "
                             "in CSC or CSR format are supported.")

    @staticmethod
    def _list_to_ctypes_ptr(ls):
        array_type = c.c_void_p * len(ls)
        array = array_type(*ls)
        ptr = c.pointer(array)
        return ptr

    def _init_dense(self, m):
        Matrix._check_precision(m)

        data_ptrs  = Matrix._list_to_ctypes_ptr(Matrix._get_data_ptrs(m))
        index_ptrs = Matrix._list_to_ctypes_ptr(Matrix._get_index_ptrs(m))

        f = resolve_precision("create_matrix", m.dtype)
        m, n = m.shape
        nnz = m * n
        self.ptr = f(m, n, nnz, index_ptrs, data_ptrs, 0, False)

    def _init_sparse_csc(self, m):
        Matirx._check_precision(m)
        pass

    def _init_sparse_csr(self, m):
        Matrix._check_precision(m)
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
