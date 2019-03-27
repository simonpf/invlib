import ctypes as c

class InvlibPointer:
    def __init__(self, ptr, destructor = None):
        self.ptr = ptr
        self.destructor = destructor

    def __del__(self):
        if not self.destructor is None:
            self.destructor(self.ptr)

    @property
    def _as_parameter_(self):
        return c.c_void_p(self.ptr)
