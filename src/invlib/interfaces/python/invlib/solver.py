"""
invlib.solver
=============

Interface to the invlib conjugate gradient solver.
"""

from invlib.matrix import Matrix
from invlib.vector import Vector
from invlib.api    import resolve_precision

import numpy as np
import ctypes as c

class ConjugateGradient:

    @property
    def step_limit(self):
        return self._step_limit

    @step_limit.setter
    def step_limit(self, step_limit):
        try:
            self._step_limit = int(step_limit)
        except:
            raise Exception("Value of step limit must be convertible to int.")

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, tolerance):
        #try:
        self._tolerance = np.float64(tolerance)
        #except:
        #    raise Exception("Tolerance value must be convertible to float64.")

    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity):
        try:
            self._verbosity = int(verbosity)
        except:
            raise Exception("Verbosity value must be convertible to int.")

    @property
    def start_vector(self):
        return self._start_vector

    @start_vector.setter
    def start_vector(self, f):
        if not callable(f):
            raise Exception("The start vector callback can only be set to a "
                            " callable object.")
        else:
            self._start_vector = f

    def __init__(self,
                 tolerance  = 1e-8,
                 step_limit = 1e4,
                 verbosity  = 0):

        self._tolerance  = None
        self._step_limit = None
        self._verbosity  = None
        self._start_vector = None

        self.tolerance  = tolerance
        self.step_limit = step_limit
        self.verbosity  = verbosity

    def _make_start_vector_callback(self, f, dtype):
        ftype = c.CFUNCTYPE(c.c_void_p, c.c_void_p, c.c_void_p)

        def wrapper(v_ptr, w_ptr):
            print("incoming: ", v_ptr, w_ptr)
            w = Vector(w_ptr, dtype)
            v = Vector(v_ptr, dtype)
            print(w)

            v_ = f(w)
            v[:] = v_
            print("returning :: ", v)
            return None

        return ftype(wrapper)

    def solve(self, A, b):

        if not type(A) == Matrix:
            raise ValueError("Type of argument A must be Matrix.")

        if not type(b) == Vector:
            raise ValueError("Type of argument b must be Vector.")

        if not A.dtype == b.dtype:
            raise ValueError("Matrix A and vector b must use same floating point"
                             " type.")

        dtype = A.dtype
        f = resolve_precision("create_solver", dtype)
        ptr = f(self.tolerance, self.step_limit, self.verbosity)

        if not self.start_vector is None:
            cb = self._make_start_vector_callback(self.start_vector, dtype)
            f = resolve_precision("solver_set_start_vector_ptr", dtype)
            print(cb)
            f(ptr, cb)

        f = resolve_precision("solver_solve", dtype)
        ptr = f(ptr, A.invlib_ptr, b.invlib_ptr)

        return Vector(ptr, dtype)
