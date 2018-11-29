
import numpy as np
import ctypes as c

import sys
sys.path.append("@LIBINVLIB_PATH@")
from invlib.vector import Vector
from invlib.api    import resolve_precision, get_stride, get_c_type, \
    buffer_from_memory

class Matrix(np.ndarray):

    def check_memory_layout(matrix):
        if not matrix.flags.c_contiguous:
            raise Exception("Only vectors that are contiguous and stored in "\
                            "C-order can be passed to invlib directly.")
        dtype = matrix.dtype
        m = matrix.shape[1]
        stride = get_stride(dtype)
        if not matrix.strides == (stride * m, stride):
            raise Exception("Only Matrix with a stride of 1 passed to invlib "\
                            " directly.")

    def check_precision(matrix):
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

        Matrix.check_precision(obj)
        Matrix.check_memory_layout(obj)
        f = resolve_precision("create_matrix", obj.dtype)
        m, n = obj.shape
        self.invlib_ptr = f(obj.ctypes.data, m, n, False)

    def multiply(self, b):

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


    def multiply_transpose(self, b):

        if isinstance(b, Vector):
            f   = resolve_precision("matrix_vector_multiply_transpose", self.dtype)
            ptr = f(self.invlib_ptr, b.invlib_ptr)
            return Vector(ptr, self.dtype)

        raise ValueError("Argument b must be of type invlib.vector.Vector.")
