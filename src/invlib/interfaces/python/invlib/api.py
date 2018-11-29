import ctypes as c
import os
import numpy  as np
import ctypes as c

precision_types = {np.float32 : 1,
                   np.float64 : 2}

path   = "@LIBINVLIB_PATH@"
invlib = c.CDLL(os.path.join(path, "libinvlib.so"))

def resolve_precision(fname, dtype):
    """
    invlib supports single and double precision arithmetic. The arithmetic
    type is used as suffix to the function name.

    This function returns the invlib API function corresponding to the given
    floating point precision.

    Arguments:

        fname(:code:`str`): The name of the function to resolve.

        dtype(:code:`numpy.dtype`): The floating point type to use.

    Raises:

        Exception if a :code:`numpy.dtype` value is given that is not supported.
    """
    if dtype == np.float32:
        return getattr(invlib, fname + "_float")
    elif dtype == np.float64:
        return getattr(invlib, fname + "_double")
    else:
        raise ValueError("Only numpy.float32 and numpy.float64 types are "\
                         " supported by invlib.")

strides = {np.dtype('float32') : 4,
           np.dtype('float64') : 8}

c_types = {np.dtype('float32') : c.c_float,
           np.dtype('float64') : c.c_double}

def get_stride(dtype):
    return strides[dtype]

def get_c_type(dtype):
    return c_types[dtype]

def buffer_from_memory(ptr, dtype, size):
    f = c.pythonapi.PyBuffer_FromMemory
    f.restype = ctypes.py_object
    s = strides[dtype]
    buffer    = f(ptr, s * size)
#
# Vectors
#

invlib.create_vector_float.argtypes = [c.c_void_p, c.c_uint64, c.c_bool]
invlib.create_vector_float.restype  = c.c_void_p

invlib.create_vector_double.argtypes = [c.c_void_p, c.c_uint64, c.c_bool]
invlib.create_vector_double.restype  = c.c_void_p

invlib.vector_rows_float.argtypes = [c.c_void_p]
invlib.vector_rows_float.restype  = c.c_uint64

invlib.vector_rows_double.argtypes = [c.c_void_p]
invlib.vector_rows_double.restype  = c.c_uint64

invlib.vector_get_data_pointer_float.argtypes = [c.c_void_p]
invlib.vector_get_data_pointer_float.restype  = c.c_void_p

invlib.vector_get_data_pointer_double.argtypes = [c.c_void_p]
invlib.vector_get_data_pointer_double.restype  = c.c_void_p

invlib.vector_dot_float.argtypes = [c.c_void_p, c.c_void_p]
invlib.vector_dot_float.restype  = c.c_float

invlib.vector_dot_double.argtypes = [c.c_void_p, c.c_void_p]
invlib.vector_dot_double.restype  = c.c_double

invlib.vector_add_float.argtypes = [c.c_void_p, c.c_void_p]
invlib.vector_add_float.restype  = c.c_void_p

invlib.vector_add_double.argtypes = [c.c_void_p, c.c_void_p]
invlib.vector_add_double.restype  = c.c_void_p

invlib.vector_subtract_float.argtypes = [c.c_void_p, c.c_void_p]
invlib.vector_subtract_float.restype  = c.c_void_p

invlib.vector_subtract_double.argtypes = [c.c_void_p, c.c_void_p]
invlib.vector_subtract_double.restype  = c.c_void_p

invlib.vector_scale_float.argtypes = [c.c_void_p, c.c_float]
invlib.vector_scale_float.restype  = None

invlib.vector_scale_double.argtypes = [c.c_void_p, c.c_double]
invlib.vector_scale_double.restype  = None

#
# Matrices
#

invlib.create_matrix_float.argtypes = [c.c_void_p, c.c_uint64, c.c_uint64, c.c_bool]
invlib.create_matrix_float.restype  = c.c_void_p

invlib.create_matrix_double.argtypes = [c.c_void_p, c.c_uint64, c.c_uint64, c.c_bool]
invlib.create_matrix_double.restype  = c.c_void_p

invlib.matrix_rows_float.argtypes = [c.c_void_p]
invlib.matrix_rows_float.restype  = c.c_uint64

invlib.matrix_rows_double.argtypes = [c.c_void_p]
invlib.matrix_rows_double.restype  = c.c_uint64

invlib.matrix_cols_float.argtypes = [c.c_void_p]
invlib.matrix_cols_float.restype  = c.c_uint64

invlib.matrix_cols_double.argtypes = [c.c_void_p]
invlib.matrix_cols_double.restype  = c.c_uint64

invlib.matrix_get_data_pointer_float.argtypes = [c.c_void_p]
invlib.matrix_get_data_pointer_float.restype  = c.c_void_p

invlib.matrix_get_data_pointer_double.argtypes = [c.c_void_p]
invlib.matrix_get_data_pointer_double.restype  = c.c_void_p

invlib.matrix_matrix_multiply_double.argtypes = [c.c_void_p, c.c_void_p]
invlib.matrix_matrix_multiply_double.restype  = c.c_void_p

invlib.matrix_matrix_multiply_float.argtypes = [c.c_void_p, c.c_void_p]
invlib.matrix_matrix_multiply_float.restype  = c.c_void_p

invlib.matrix_vector_multiply_double.argtypes = [c.c_void_p, c.c_void_p]
invlib.matrix_vector_multiply_double.restype  = c.c_void_p

invlib.matrix_vector_multiply_float.argtypes = [c.c_void_p, c.c_void_p]
invlib.matrix_vector_multiply_float.restype  = c.c_void_p

invlib.matrix_vector_multiply_transpose_double.argtypes = [c.c_void_p, c.c_void_p]
invlib.matrix_vector_multiply_transpose_double.restype  = c.c_void_p

invlib.matrix_vector_multiply_transpose_float.argtypes = [c.c_void_p, c.c_void_p]
invlib.matrix_vector_multiply_transpose_float.restype  = c.c_void_p
