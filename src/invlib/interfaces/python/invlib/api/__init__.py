
import ctypes as c
import os
import numpy  as np
import ctypes as c

from . import float  as sp
from . import double as dp

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
        return getattr(sp.invlib, fname)
    elif dtype == np.float64:
        return getattr(dp.invlib, fname)
    else:
        raise ValueError("Only numpy.float32 and numpy.float64 types are "\
                         " supported by invlib.")

strides = {np.dtype('float32') : 4,
           np.dtype('float64') : 8}

ctypes_scalar_types = {np.dtype('float32') : c.c_float,
                       np.dtype('float64') : c.c_double}

def get_stride(dtype):
    return strides[dtype]

def get_ctypes_scalar_type(dtype):
    return ctypes_scalar_types[dtype]

def get_ctypes_index_type():
    return c.c_uint

def buffer_from_memory(ptr, dtype, size):
    f = c.pythonapi.PyBuffer_FromMemory
    f.restype = ctypes.py_object
    s = strides[dtype]
    buffer    = f(ptr, s * size)

def get_matrix_struct(dtype):
    if dtype == np.float32:
        t = sp.matrix_struct
    else:
        t = dp.matrix_struct
    return t

def to_forward_model_struct(fm, dtype):
    if dtype == np.float32:
        t = sp.forward_model_struct
    else:
        t = dp.forward_model_struct
    return t(fm.m, fm.n,
             fm.make_jacobian_wrapper(dtype),
             fm.make_evaluate_wrapper(dtype))

def to_optimizer_struct(opt, dtype):
    if dtype == np.float32:
        t = sp.optimizer_struct
    else:
        t = dp.optimizer_struct
    return t(*opt._to_optimizer_struct(dtype))
