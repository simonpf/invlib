"""
invlib.solver
=============

Interface to the invlib conjugate gradient solver.
"""

from invlib.matrix import Matrix
from invlib.vector import Vector
from invlib.api    import resolve_precision

import numpy as np

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

    def __init__(self,
                 tolerance  = 1e-6,
                 step_limit = 1e4,
                 verbosity  = 0):

        self._tolerance  = None
        self._step_limit = None
        self._verbosity  = None

        self.tolerance  = tolerance
        self.step_limit = step_limit
        self.verbosity  = verbosity

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

        f = resolve_precision("solver_solve", dtype)
        ptr = f(ptr, A.invlib_ptr, b.invlib_ptr)

        return Vector.from_invlib_pointer(ptr, dtype)
