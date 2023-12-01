import parallelproj
from array_api_compat import get_namespace

def negativePoissonLogL(exp, data):
    xp = get_namespace(exp)
    return float(xp.sum(exp - data * xp.log(exp)))

class L2L1GradientNorm:
    def __init__(self, beta: float, G: parallelproj.LinearOperator) -> None:
        self._beta = beta
        self._G = G
    
    def __call__(self, x) -> float:
        xp = get_namespace(x)
        return self._beta * xp.sum(xp.linalg.vector_norm(self._G(x), axis=0))


def prox_dual_l2l1(y, sigma):
    xp = get_namespace(y)
    gnorm = xp.linalg.vector_norm(y, axis=0)
    gnorm[gnorm < 1] = 1
    return y / gnorm

