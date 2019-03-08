"""
invlib.matrix
=============

The :code:`invlib.vector` module contains the :code:`Vector` class that provides
efficient linear algebra operations for dense matrices.
"""
import sys
sys.path.append("@LIBINVLIB_PATH@")

import numpy as np
import scipy as sp
import scipy.sparse
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
        m, n, nnz, fmt, element_ptr, start_ptr, index_ptr \
            = Matrix.matrix_info(ptr, dtype)
        # Dense format
        if fmt == 0:
            if dtype == np.float32:
                ct = c.c_float
            else:
                ct = c.c_double
            b = (ct * (m * n)).from_address(element_ptr)
            matrix = np.ctypeslib.as_array(b).reshape(m, n)

        # CSC sparse format
        elif fmt == 1:
            matrix = sp.sparse.csc_matrix((element_ptr, (index_ptr, start_ptr)),
                                     shape = (m, n))


        # CSR sparse format
        elif fmt == 2:
            matrix = sp.sparse.csr_matrix((element_ptr, (index_ptr, start_ptr)),
                                     shape = (m, n))
        else:
            raise Exception("Currently only dense, CSC and CSR formats are "
                            " supported.")
        return matrix


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

    def __init__(self, *args, **kwargs):

        if len(args) == 2:
            ptr, dtype = args
            if not ((type(ptr) == int) and (dtype in [np.float32, np.float64])):
                raise ValueError("Expected the two positional arguments "
                                 " provided to matrix constructor to be a "
                                 "ctypes pointer and numpy.dtype but found "
                                 "something else.")
            matrix = Matrix.from_invlib_pointer(ptr, dtype)
            format = None

        elif len(args) == 1:
            matrix, = args
            if "format" in kwargs:
                format = kwargs["format"]
            else:
                format = None

        fi = -1
        # Try to deduce format from type of matrix.
        if format is None:
            if type(matrix) == np.ndarray:
                fi = 0
            elif type(matrix) == sp.sparse.csc_matrix:
                fi = 1
            elif type(matrix) == sp.sparse.csr_matrix:
                fi = 2
            else:
                raise Exception("numpy.ndarray or scipy sprase matrix required"\
                                "to create matrix.")
        else:
            if format == "dense":
                matrix = np.asarray(matrix)
                fi = 0
            elif format == "sparse_csc":
                try:
                    matrix = matrix.to_csc()
                    fi = 1
                except:
                    raise ValueError("To create a matrix in sparse CSC format "\
                                     "the provided matrix must be convertible "\
                                     "to a scipy.sparse.csc_matrix matrix.")
            elif format == "sparse_csr":
                try:
                    matrix = matrix.to_csr()
                    fi = 2
                except:
                    raise ValueError("To create a matrix in sparse CSR format "\
                                     "the provided matrix must be convertible "\
                                     "to a scipy.sparse.csr_matrix matrix.")

        Matrix._check_precision(matrix)
        if fi == 0:
            Matrix._check_memory_layout(matrix)

        self.matrix = matrix
        data_ptrs   = Matrix._list_to_ctypes_ptr(Matrix._get_data_ptrs(matrix))
        index_ptrs  = Matrix._list_to_ctypes_ptr(Matrix._get_index_ptrs(matrix))
        m, n = matrix.shape
        nnz = m * n
        f = resolve_precision("create_matrix", matrix.dtype)
        self.invlib_ptr = f(m, n, nnz, index_ptrs, data_ptrs, fi, False)



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
            f   = resolve_precision("matrix_matrix_multiply", self.matrix.dtype)
            ptr = f(self.invlib_ptr, b.invlib_ptr)
            return Matrix(ptr, self.matrix.dtype)

        elif isinstance(b, Vector):
            f   = resolve_precision("matrix_vector_multiply", self.matrix.dtype)
            ptr = f(self.invlib_ptr, b.invlib_ptr)
            return Vector(ptr, self.matrix.dtype)

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
            f   = resolve_precision("matrix_vector_multiply_transpose", self.matrix.dtype)
            ptr = f(self.invlib_ptr, b.invlib_ptr)
            return Vector(ptr, self.matrix.dtype)

        raise ValueError("Argument b must be of type invlib.vector.Vector.")
