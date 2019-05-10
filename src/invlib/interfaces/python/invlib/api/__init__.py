
import ctypes as c
import os
import numpy  as np
import ctypes as c

from . import cpu_float32
from . import cpu_float64

backend = {"single_precision" : cpu_float32,
           "double_precision" : cpu_float64}

def get_backend(fp_type):
    if fp_type in [np.float32, np.dtype("float32")]:
        return backend["single_precision"]
    elif fp_type in [np.float64, np.dtype("float64")]:
        return backend["double_precision"]
    else:
        raise Exception("Floating point type " + str(fp_type) + " not "
                        " supported by invlib.")

try:
    from . import mpi_float32  as mpi_sp
    from . import mpi_float64 as mpi_dp
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()
    parallel = mpi_size > 1

    if parallel:
        backend["single_precision"] = mpi_sp
        backend["double_precision"] = mpi_dp

except Exception as e:
    print("Failed loading MPI backend: ", e)
    parallel = False

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
        return getattr(backend["single_precision"].invlib, fname)
    elif dtype == np.float64:
        return getattr(backend["double_precision"].invlib, fname)
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
    b = get_backend(dtype)
    return b.matrix_struct

def to_forward_model_struct(fm, dtype):
    b = get_backend(dtype)
    t = b.forward_model_struct
    return t(fm.m, fm.n,
             fm.make_jacobian_wrapper(dtype),
             fm.make_evaluate_wrapper(dtype))

def to_optimizer_struct(opt, dtype):
    b = get_backend(dtype)
    t = b.optimizer_struct
    return t(*opt._to_optimizer_struct(dtype))
