import ctypes as c
import os

precision_types = {np.float32 : 1,
                   np.float64 : 2}

path = "@LIBINVLIB_PATH@"
invlib = c.CDLL(os.path.join(path, "libinvlib.so"))
