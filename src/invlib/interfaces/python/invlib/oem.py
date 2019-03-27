import numpy  as np
import scipy  as sp
import ctypes as c

from invlib.api           import resolve_precision, to_forward_model_struct,\
    to_optimizer_struct
from invlib.forward_model import ForwardModel
from invlib.matrix        import Matrix
from invlib.vector        import Vector
from invlib.optimizer     import GaussNewton

class OEM:

    def __init__(self,
                 forward_model,
                 s_a_inv,
                 s_e_inv,
                 x_a,
                 optimizer = GaussNewton()):

        if not isinstance(forward_model, ForwardModel):
            raise Exception("The forward model must inherit from the "
                            "invlib.FowardModel base class.")

        if not all([isinstance(s, Matrix) for s in [s_a_inv, s_e_inv]]):
            raise Exception("The covariance matrices s_a_inv and s_e_inv must "
                            "be of type invlib.Matrix")

        if not all([isinstance(v, Vector) for v in [x_a]]):
            raise Exception("x_a and y must be of type invlib.Vector")

        self.forward_model = forward_model
        self.sa_inv        = s_a_inv
        self.se_inv        = s_e_inv
        self.x_a           = x_a
        self.optimizer     = optimizer
        self.dtype         = x_a.dtype

    def compute(self, y):

        if not isinstance(y, Vector):
            raise Exception("y must be of type invlib.Vector")

        if not self.dtype == y.dtype:
            raise Exception("x_a and y must use the same dtype.")

        f = resolve_precision("oem", self.dtype)
        ptr = f(to_forward_model_struct(self.forward_model, self.dtype),
                self.sa_inv.invlib_pointer,
                self.se_inv.invlib_pointer,
                self.x_a.invlib_pointer,
                None,
                y.invlib_pointer,
                to_optimizer_struct(self.optimizer, self.dtype),
                self.optimizer.solver.to_invlib_pointer(self.dtype))
        return Vector(ptr, self.dtype)
