from invlib.matrix import Matrix
from invlib.vector import Vector
from invlib.api    import resolve_precision, to_forward_model_struct

import numpy as np
import ctypes as c
from abc import ABCMeta, abstractmethod

#
# Struct declarations
#

class ForwardModel(metaclass = ABCMeta):

    def __init__(self, m, n):
        self.m = m
        self.n = n

    @abstractmethod
    def evaluate(self, x):
        pass

    @abstractmethod
    def jacobian(self, x):
        pass

    def make_evaluate_wrapper(self, dtype):
        ft = c.CFUNCTYPE(c.c_void_p, c.c_void_p)

        def wrapper(x_ptr):
            x = Vector(x_ptr, dtype)
            y = self.evaluate(x)
            y = Vector(y)
            return y.invlib_pointer

        return ft(wrapper)

    def make_jacobian_wrapper(self, dtype):
        ft = c.CFUNCTYPE(c.c_void_p, c.c_void_p, c.c_void_p)

        def wrapper(x_ptr, y_ptr):
            x = Vector(x_ptr, dtype)
            y = Vector(y_ptr, dtype)

            jac, y_new = self.jacobian(x)

            y[:] = y_new
            jac  = Matrix(jac)

            return jac.invlib_pointer

        return ft(wrapper)

    def evaluate_api(self, x):

        if not isinstance(x, Vector):
            raise ValueError("Argument x must be of type invlib.Vector.")

        dtype = x.dtype
        f = resolve_precision("forward_model_evaluate", dtype)
        fs = to_forward_model_struct(self, dtype)
        ptr = f(fs, x.invlib_pointer)
        return Vector(ptr, dtype)

    def jacobian_api(self, x):

        if not isinstance(x, Vector):
            raise ValueError("Argument x must be of type invlib.Vector.")

        dtype = x.dtype

        y = Vector(np.zeros(self.m, dtype = dtype))
        f = resolve_precision("forward_model_jacobian", dtype)
        jac = f(to_forward_model_struct(self, dtype),
                x.invlib_pointer,
                y.invlib_pointer)

        return Matrix(jac, dtype), y


class LinearModel(ForwardModel):
    def __init__(self, K, x0 = None):
        super().__init__(*K.shape)
        self.K  = Matrix(K)

        if not x0 is None:
            x0 = Vector(x0)
        self.x0 = x0

    def evaluate(self, x):
        if self.x0:
            return self.K.multiply(x - self.x0)
        else:
            return self.K.multiply(x)

    def jacobian(self, x):
        y = self.evaluate(x)
        return self.K, y
