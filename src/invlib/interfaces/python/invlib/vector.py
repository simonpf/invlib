"""
invlib.vector
=============

The :code:`invlib.vector` module contains the :code:`Vector` class that provides
efficient linear algebra operations for dense vectors.
"""
import sys
sys.path.append("@LIBINVLIB_PATH@")

import numpy  as np
import ctypes as c

from invlib.api import resolve_precision, get_stride, get_c_type, \
    buffer_from_memory

################################################################################
# The Vector class
################################################################################

class Vector(np.ndarray):
    """
    The :code:`Vector` is used to represent dense column vectors. The
    :code:`Vector` class inherits from :code:`numpy.ndarray` and can therefore
    be constructed from every 2D :code:`ndarray` object. 2D arrays are also
    accepts if the second dimension has size 1.

    All instances of the :code:`Vector` class are assumed to be column vector.
    """

    wrong_argument_error = ValueError("invlib.Vector objects can only be created"
                                      " from numpy.ndarrays of type float32 or "
                                      " float64.")
    def _check_memory_layout(vector):
        """
        Check that the memory layout of a :code:`numpy.ndarray` is c-style
        and contiguous so that it can be used without copying inside
        invlib.

        Arguments:

            vector(:code:`np.ndarray`): The array of which to check the memory
                layout.
        """
        if not vector.flags.c_contiguous:
            raise Exception("Only vector that are contiguous and stored in "\
                            "C-order can be passed to invlib directly.")
        dtype = vector.dtype
        print(vector.strides)
        print((get_stride(dtype),) * (len(vector.shape)))
        if not vector.strides == (get_stride(dtype),) * (len(vector.shape)):
            raise Exception("Only vectors with a stride of 1 can be passed "
                            "passed to invlib directly.")

    def _check_precision(vector):
        """
        Check that precision of a :code:`numpy.ndarray` matches the types supported
        by invlib.

        Arguments:

            vector(:code:`np.ndarray`): The array of which to check the memory
                layout.
        """
        dtype = vector.dtype
        if not dtype in [np.float32, np.float64]:
            raise Vector.wrong_argument_error

    @staticmethod
    def create_invlib_vector(obj):
        Vector._check_precision(obj)
        Vector._check_memory_layout(obj)
        f   = resolve_precision("create_vector", obj.dtype)
        ptr = f(obj.size, obj.ctypes.data, False)
        return ptr

    @staticmethod
    def from_invlib_pointer(ptr, dtype):
        f_rows = resolve_precision("vector_rows", dtype)
        n = f_rows(ptr)

        shape   = (n, 1)
        stride  = get_stride(dtype)
        strides = (stride,) * len(shape)

        b = resolve_precision("vector_element_pointer", dtype)(ptr)
        ctype = get_c_type(dtype)
        b   = c.cast(b, c.POINTER(ctype))
        arr = np.ctypeslib.as_array(b, shape = shape)
        return arr

        obj = super(Vector, self).__new__(Vector, shape, dtype, arr.data, 0, strides, 'C')

        self.invlib_ptr = ptr

    def __new__(cls, *args, **kwargs):
        if len(args) == 1:
            arr, = args
        if len(args) == 2:
            ptr, dtype = args
            arr = Vector.from_invlib_pointer(ptr, dtype)

        obj = super(Vector, cls).__new__(Vector, arr.shape, arr.dtype,
                                          arr.data, 0, arr.strides, 'C')
        obj.invlib_ptr = Vector.create_invlib_vector(obj)
        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return None

        if not obj.dtype in [np.float32, np.float64]:
            return np.array(obj)

        self.invlib_ptr = Vector.create_invlib_vector(obj)

    @property
    def rows(self):
        """
        Number of rows of the column vector.
        """
        f = resolve_precision("vector_rows", self.dtype)
        return f(self.invlib_ptr)

    def dot(self, v):
        """
        Compute the dot product of this vector with :code:`v`.

        Arguments:

            v(:code:`invlib.Vector`): The vector to compute the
                dot product with.

        Returns:

            The scalar results of the dot product.
        """
        f = resolve_precision("vector_dot", self.dtype)
        return f(self.invlib_ptr, v.invlib_ptr)

    def add(self, v):
        """
        Compute the sum of this vector and another vector.

        Arguments:

            v(:code:`invlib.Vector`): The vector to add to this vector.

        Returns:

            :code:`invlib.Vector` containing the sum of the two vectors.
        """
        if isinstance(v, Vector):
            f = resolve_precision("vector_add", self.dtype)
            ptr = f(self.invlib_ptr, v.invlib_ptr)
            return Vector(ptr, self.dtype)

        raise Vector.wrong_argument_error

    def subtract(self, v):
        """
        Compute the difference of this vector and another vector.

        Arguments:

            v(:code:`invlib.Vector`): The vector to add to this vector.

        Returns:

            :code:`invlib.Vector` containing the difference of the two vectors.
        """
        if isinstance(v, Vector):
            f = resolve_precision("vector_subtract", self.dtype)
            ptr = f(self.invlib_ptr, v.invlib_ptr)
            return Vector(ptr, self.dtype)

        raise Vector.wrong_argument_error

    def scale(self, c):
        """
        Scale vector by scalar factor.

        Arguments:

            v(:code:`float`): The scaling factor.

        """
        f = resolve_precision("vector_scale", self.dtype)
        f(self.invlib_ptr, c)
