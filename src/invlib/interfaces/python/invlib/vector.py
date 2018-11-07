import numpy as np


class Vector(np.ndarray):

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        obj.info = info

        shape = obj.shape
        if len(shape) != 2 and shape[1] != 1:
            raise ValueError("invlib.Vector object can only be created from"
                             "numpy.ndarrays of shape (:, 1).")

        dtype = obj.dtype
        if not dtype is in [np.float32, np.float64]:
            raise ValueError("invlib.Vector objects can only be created from"
                             "numpy.ndarrays of type float32 or float64")
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.info = getattr(obj, 'info', None)
