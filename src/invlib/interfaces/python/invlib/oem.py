import sys
sys.path.append("@LIBINVLIB_PATH@")

import numpy  as np
import scipy  as sp
import ctypes as c

import scipy.sparse

from invlib.api    import resolve_precision
from invlib.mkl    import MklSparseCsr, MklSparseCsc
from invlib.vector import Vector

class OEM:

    def __init__(self,
                 forward_model,
                 sa_inv,
                 se_inv,
                 x_a,
                 y):

        if not (isinstance(forward_model, MklSparseCsr) or
                isinstance(forward_model, MklSparseCsc)):
            raise Exception("Currently only linear forward models are supported.")

        if not all([isinstance(s, MklSparseCsc) for s in [sa_inv, se_inv]]):
            raise Exception("sa_inv and se_inv must be of type invlib.MklSparseCsr")

        if not all([isinstance(v, Vector) for v in [x_a, y]]):
            raise Exception("x_a and y must be of type invlib.Vector")

        self.forward_model = forward_model
        self.sa_inv        = sa_inv
        self.se_inv        = se_inv
        self.x_a           = x_a
        self.y             = y
        self.dtype         = y.dtype

    def compute(self):
        f = resolve_precision("map_linear", self.dtype)
        ptr = f(self.forward_model.invlib_ptr,
                self.sa_inv.invlib_ptr,
                self.se_inv.invlib_ptr,
                self.x_a.invlib_ptr,
                self.y.invlib_ptr)
        return Vector(ptr, self.dtype)

    def evaluate_forward_model(self, x):
        f = resolve_precision("forward_model_linear", self.dtype)
        ptr = f(self.forward_model.invlib_ptr, x.invlib_ptr)
        return Vector(ptr, self.dtype)

    def covmat_multiply(self, S, x):
        f = resolve_precision("covmat_multiply", self.dtype)
        ptr = f(S.invlib_ptr, x.invlib_ptr)
        return Vector(ptr, self.dtype)




