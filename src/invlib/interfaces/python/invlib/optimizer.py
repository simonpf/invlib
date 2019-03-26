from invlib.solver import ConjugateGradient

class GaussNewton:

    def _to_optimizer_struct(self, dtype):
        return (0, None)

    def __init__(self, solver = ConjugateGradient()):
        self.solver = solver
