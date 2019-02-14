import sys
sys.path.append("@LIBINVLIB_PATH@")

import numpy  as np
import ctypes as c

from invlib.api import resolve_precision, get_stride, get_c_type, \
    buffer_from_memory

class Vector(np.ndarray):

    wrong_argument_error = ValueError("invlib.Vector objects can only be created"
                                      " from numpy.ndarrays of type float32 or "
                                      " float64.")
    def check_memory_layout(vector):
        if not vector.flags.c_contiguous:
            raise Exception("Only vector that are contiguous and stored in "\
                            "C-order can be passed to invlib directly.")
        dtype = vector.dtype
        if not vector.strides == (get_stride(dtype),) * (len(vector.shape)):
            raise Exception("Only vectors with a stride of 1 passed to invlib "\
                            " directly.")

    def check_precision(vector):
        dtype = vector.dtype
        if not dtype in [np.float32, np.float64]:
            raise Vector.wrong_argument_error

    def __new__(self, invlib_ptr, dtype):
        f_rows = resolve_precision("vector_rows", dtype)
        n = f_rows(invlib_ptr)

        shape   = (n, 1)
        stride  = get_stride(dtype)
        strides = (stride,) * len(shape)

        b = resolve_precision("vector_get_data_pointer", dtype)(invlib_ptr)
        ctype = get_c_type(dtype)
        b   = c.cast(b, c.POINTER(ctype))
        arr = np.ctypeslib.as_array(b, shape = shape)
        obj = super(Vector, self).__new__(Vector, shape, dtype, arr.data, 0, strides, 'C')

        self.invlib_ptr = invlib_ptr
        print("data_pointer", self.invlib_ptr)
        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return None

        Vector.check_precision(obj)
        Vector.check_memory_layout(obj)
        f = resolve_precision("create_vector", obj.dtype)
        self.invlib_ptr = f(obj.ctypes.data, obj.size, False)

    @property
    def rows(self):
        f = resolve_precision("vector_rows", self.dtype)
        return f(self.invlib_ptr)

    def dot(self, v):
        f = resolve_precision("vector_dot", self.dtype)
        return f(self.invlib_ptr, v.invlib_ptr)

    def add(self, v):

        if isinstance(v, Vector):
            f = resolve_precision("vector_add", self.dtype)
            ptr = f(self.invlib_ptr, v.invlib_ptr)
            return Vector(ptr, self.dtype)

        raise Vector.wrong_argument_error

    def subtract(self, v):
        if isinstance(v, Vector):
            f = resolve_precision("vector_subtract", self.dtype)
            ptr = f(self.invlib_ptr, v.invlib_ptr)
            return Vector(ptr, self.dtype)

        raise Vector.wrong_argument_error

    def scale(self, c):
        f = resolve_precision("vector_scale", self.dtype)
        print(self.invlib_ptr)
        f(self.invlib_ptr, c)
