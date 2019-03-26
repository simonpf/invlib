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
from invlib.api    import resolve_precision, get_stride, buffer_from_memory, \
    get_ctypes_scalar_type, get_ctypes_index_type, get_matrix_struct

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

        f = resolve_precision("matrix_info", dtype)
        ms = f(ptr)

        m   = ms.m
        n   = ms.n
        nnz = ms.nnz
        fmt = ms.format

        if fmt == 0:
            b = (get_ctypes_scalar_type(dtype) * (m * n)).from_address(ms.data_pointers[0])
            elements = [np.ctypeslib.as_array(b).reshape(m, n)]
            indices  = None
            starts   = None

        if fmt > 0:

            b = (get_ctypes_scalar_type(dtype) * nnz).from_address(ms.data_pointers[0])
            elements  = [np.ctypeslib.as_array(b)]

            indices = []
            b = (get_ctypes_index_type() * nnz).from_address(ms.index_pointers[0])
            indices += [np.ctypeslib.as_array(b)]

            starts = []

            if fmt == 1:

                b = (get_ctypes_index_type() * (n + 1)).from_address(ms.start_pointers[0])
                starts += [np.ctypeslib.as_array(b, shape = (n + 1,))]

            if fmt == 2:

                b = (get_ctypes_index_type() * (m + 1)).from_address(ms.start_pointers[0])
                starts += [np.ctypeslib.as_array(b, shape = (m + 1,))]

            if fmt == 3:
                b = (get_ctypes_scalar_type(dtype) * nnz).from_address(ms.data_pointers[1])
                elements += [np.ctypeslib.as_array(b)]
                b = (get_ctypes_index_type() * nnz).from_address(ms.index_pointers[1])
                indices += [np.ctypeslib.as_array(b)]
                b = (get_ctypes_index_type() * (n + 1)).from_address(ms.start_pointers[0])
                starts += [np.ctypeslib.as_array(b, shape = (n + 1,))]
                b = (get_ctypes_index_type() * (m + 1)).from_address(ms.start_pointers[1])
                starts += [np.ctypeslib.as_array(b, shape = (m + 1,))]

        return m, n, nnz, fmt, elements, indices, starts

    @staticmethod
    def from_invlib_pointer(ptr, dtype):
        m, n, nnz, fmt, elements, indices, starts \
            = Matrix.matrix_info(ptr, dtype)
        # Dense format
        if fmt == 0:
            matrix = elements[0]
        # CSC sparse format
        elif fmt == 1:
            matrix = sp.sparse.csc_matrix((elements[0], indices[0], starts[0]),
                                     shape = (m, n))
        # CSR sparse format
        elif fmt == 2:
            matrix = sp.sparse.csr_matrix((elements[0], indices[0], starts[0]),
                                     shape = (m, n))
        # Hybrid format
        elif fmt == 3:
            m1 = sp.sparse.csc_matrix((elements[0], indices[0], starts[0]),
                                      shape = (m, n))
            m2 = sp.sparse.csr_matrix((elements[1], indices[1], starts[1]),
                                      shape = (m, n))
            matrix = (m1, m2)
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
        dtype = Matrix._get_dtype(matrix)
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
        dtype = Matrix._get_dtype(matrix)
        m = matrix.shape[1]
        stride = get_stride(dtype)
        if not matrix.strides == (stride * m, stride):
            raise Exception("Only Matrix with a stride of 1 passed to invlib "\
                            " directly.")

    @staticmethod
    def _get_dtype(matrix):
        if isinstance(matrix, tuple):
            return matrix[0].dtype
        else:
            return matrix.dtype

    @staticmethod
    def _get_shape(matrix):
        if isinstance(matrix, tuple):
            return matrix[0].shape
        else:
            return matrix.shape

    @staticmethod
    def _get_nnz(matrix):
        if isinstance(matrix, np.ndarray):
            return matrix.size
        elif isinstance(matrix, tuple):
            return matrix[0].nnz
        else:
            return matrix.nnz

    @staticmethod
    def _get_data_pointers(matrix):
        if isinstance(matrix, np.ndarray):
            ptrs =  [matrix.ctypes.data, None]
        elif isinstance(matrix, sp.sparse.csc_matrix):
            ptrs = [matrix.data.ctypes.data, None]
        elif isinstance(matrix, sp.sparse.csr_matrix):
            ptrs = [matrix.data.ctypes.data, None]
        elif isinstance(matrix, tuple):
            m1, m2 = matrix
            ptrs = [m1.data.ctypes.data, m2.data.ctypes.data]
        return (c.c_void_p * 2)(*ptrs)

    @staticmethod
    def _get_index_pointers(matrix):
        if isinstance(matrix, np.ndarray):
            ptrs = []
        elif isinstance(matrix, sp.sparse.csc_matrix):
            ptrs = [matrix.indices.ctypes.data, None]
        elif isinstance(matrix, sp.sparse.csr_matrix):
            ptrs = [matrix.indices.ctypes.data, None]
        elif isinstance(matrix, tuple):
            m1, m2 = matrix
            ptrs = [m1.indices.ctypes.data, m2.indices.ctypes.data]
        return (c.c_void_p * 2)(*ptrs)

    @staticmethod
    def _get_start_pointers(matrix):
        if isinstance(matrix, np.ndarray):
            ptrs = []
        elif isinstance(matrix, sp.sparse.csc_matrix):
            ptrs = [matrix.indptr.ctypes.data]
        elif isinstance(matrix, sp.sparse.csr_matrix):
            ptrs = [matrix.indptr.ctypes.data]
        elif isinstance(matrix, tuple):
            m1, m2 = matrix
            ptrs = [m1.indptr.ctypes.data, m2.indptr.ctypes.data]
        return (c.c_void_p * 2)(*ptrs)

    @staticmethod
    def _get_format(matrix):
        if isinstance(matrix, np.ndarray):
            return 0
        elif isinstance(matrix, sp.sparse.csc_matrix):
            return 1
        elif isinstance(matrix, sp.sparse.csr_matrix):
            return 2
        elif isinstance(matrix, tuple):
            return 3

    @staticmethod
    def _to_matrix_struct(matrix):
        m, n = Matrix._get_shape(matrix)

        format          = Matrix._get_format(matrix)
        data_pointers   = Matrix._get_data_pointers(matrix)
        index_pointers  = Matrix._get_index_pointers(matrix)
        start_pointers  = Matrix._get_start_pointers(matrix)

        nnz = Matrix._get_nnz(matrix)

        dtype = Matrix._get_dtype(matrix)
        ms = get_matrix_struct(dtype)
        return ms(m, n, nnz, format, data_pointers, index_pointers, start_pointers)


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

            if isinstance(matrix, Matrix):
                matrix = matrix.matrix

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
            elif type(matrix) == tuple:
                fi = 3
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
            elif format == "sparse_hyb":
                try:
                    matrix = (matrix.to_csc(),
                              matrix.to_csr())
                    fi = 3
                except:
                    raise ValueError("To create a matrix in sparse Hybrid format "\
                                     "the provided matrix must be convertible "\
                                     "to a scipy.sparse.csc_matrix and scipy.sparse"
                                     ".csr_matrix.")

        Matrix._check_precision(matrix)
        if fi == 0:
            Matrix._check_memory_layout(matrix)

        dtype = Matrix._get_dtype(matrix)
        f = resolve_precision("create_matrix", dtype)
        self._invlib_pointer = f(Matrix._to_matrix_struct(matrix), False)
        self.matrix = matrix

    @property
    def invlib_pointer(self):
        return self._invlib_pointer

    @property
    def dtype(self):
        return Matrix._get_dtype(self.matrix)

    @property
    def format(self):
        return Matrix._get_format(self.matrix)

    @property
    def nnz(self):
        return Matrix._get_nnz(self.matrix)

    @property
    def shape(self):
        return Matrix._get_shape(self.matrix)

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
        dtype = Matrix._get_dtype(self.matrix)

        if isinstance(b, Matrix):
            f   = resolve_precision("matrix_matrix_multiply", dtype)
            ptr = f(self.invlib_pointer, b.invlib_pointer)
            return Matrix(ptr, dtype)

        elif isinstance(b, Vector):
            f   = resolve_precision("matrix_vector_multiply", dtype)
            ptr = f(self.invlib_pointer, b.invlib_pointer)
            return Vector(ptr, dtype)

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
        dtype = Matrix._get_dtype(self.matrix)

        if isinstance(b, Matrix):
            f   = resolve_precision("matrix_matrix_transpose_multiply", dtype)
            ptr = f(self.invlib_pointer, b.invlib_pointer)
            return Matrix(ptr, dtype)

        if isinstance(b, Vector):
            f   = resolve_precision("matrix_vector_transpose_multiply", dtype)
            ptr = f(self.invlib_pointer, b.invlib_pointer)
            return Vector(ptr, dtype)

        raise ValueError("Argument b must be of type invlib.matrix.Matrix or "
                         "invlib.vector.Vector.")
