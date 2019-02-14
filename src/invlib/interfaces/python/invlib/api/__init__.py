
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
