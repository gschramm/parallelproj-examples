import abc
import numpy as np
import numpy.typing as npt
import numpy.array_api as xp


class RealLinearOperator(abc.ABC):

    def __init__(self) -> None:
        self._scale = 1.0

    @property
    def scale(self) -> float:
        return self._scale
    
    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value

    @property
    @abc.abstractmethod
    def in_shape(self) -> tuple[int,...]:
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def out_shape(self) -> tuple[int,...]:
        raise NotImplementedError

    @abc.abstractmethod
    def _forward(self, x: npt.NDArray) -> npt.NDArray:
        raise NotImplementedError
    
    @abc.abstractmethod
    def _adjoint(self, y: npt.NDArray) -> npt.NDArray:
        raise NotImplementedError

    def forward(self, x: npt.NDArray) -> npt.NDArray:
        if self._scale == 1.0:
            return self._forward(x)
        else:
            return self._forward(x) * self._scale
        
    def adjoint(self, y: npt.NDArray) -> npt.NDArray:
        if self._scale == 1.0:
            return self._adjoint(y)
        else:
            return self._adjoint(y) * self._scale

    def norm(self, num_iter=50, dev = 'cpu') -> float:
        """estimate norm of operator via power iterations

        Parameters
        ----------
        num_iter : int, optional
            number of iterations, by default 50
        dev: str, optional
            device to use, by default 'cpu'

        Returns
        -------
        float
            the estimated norm
        """

        x = xp.asarray(np.random.rand(*self.in_shape), device=dev)

        for i in range(num_iter):
            x = self.adjoint(self.forward(x))
            n = xp.linalg.vector_norm(x)
            x /= n

        return float(xp.sqrt(n))

    def adjointness_test(self, verbose: bool = False, dev = 'cpu') -> None:
        """test if adjoint is really the adjoint of forward

        Parameters
        ----------
        verbose : bool, optional
            print verbose output
        dev: str, optional
            device to use, by default 'cpu'
        """

        x = xp.asarray(np.random.rand(*self.in_shape), device=dev)
        y = xp.asarray(np.random.rand(*self.out_shape), device=dev)

        x_fwd = self.forward(x)
        y_back = self.adjoint(y)

        a = xp.sum(x_fwd * y)
        b = xp.sum(x * y_back)

        if verbose:
            print(f'<y, A x>   {a}')
            print(f'<A^T y, x> {b}')

        assert np.isclose(a, b)


class MatrixOperator(RealLinearOperator):
    """class for linear operator using dense matrix"""
    def __init__(self, A: npt.NDArray, in_shape: tuple[int,...]) -> None:
        super().__init__()
        self._A = A
        self._in_shape = in_shape
        self._out_shape = (A.shape[0],)

    @property
    def in_shape(self) -> tuple[int,...]:
        return self._in_shape
    
    @property
    def out_shape(self) -> tuple[int,...]:
        return self._out_shape
        
    def _forward(self, x: npt.NDArray) -> npt.NDArray:
        return self._A @ xp.reshape(x, x.size)
    
    def _adjoint(self, y: npt.NDArray) -> npt.NDArray:
        return xp.reshape(self._A.T @ y, self._in_shape)

class GradientOperator2D(RealLinearOperator):
    """finite difference gradient operator"""

    def __init__(self, in_shape: tuple[int, ...]) -> None:

        super().__init__()
        self._in_shape = in_shape
        self._out_shape = (len(in_shape), ) + in_shape

    @property
    def in_shape(self) -> tuple[int,...]:
        return self._in_shape
    
    @property
    def out_shape(self) -> tuple[int,...]:
        return self._out_shape
 
    def _forward(self, x):
        g = xp.zeros(self.out_shape, dtype=x.dtype, device=x.device)

        g[0, :-1, :] = x[1:,:] - x[:-1,:]
        g[1, :, :-1] = x[:,1:] - x[:,:-1]

        return g

    def _adjoint(self, y):
        tmp0 = xp.asarray(y[0, :, :], copy=True)
        tmp0[-1,:] = 0 

        tmp1 = xp.asarray(y[1, :, :], copy=True)
        tmp1[:, -1] = 0 

        div0 = xp.zeros(self.in_shape, dtype=y.dtype, device=y.device)
        div1 = xp.zeros(self.in_shape, dtype=y.dtype, device=y.device)

        div0[1:, :] = -tmp0[1:,:] + tmp0[:-1,:]
        div0[0,:] = -tmp0[0, :]

        div1[:, 1:] = -tmp1[:, 1:] + tmp1[:, :-1]
        div1[:, 0] = -tmp1[:, 0]

        return div0 + div1

class NegativePoissonLogL:
    def __init__(self, data: npt.NDArray, contamination: npt.NDArray, P: RealLinearOperator) -> None:
        self._data = data
        self._contamination = contamination
        self._P = P
    
    def __call__(self, x: npt.NDArray) -> float:
        exp = self._P.forward(x) + self._contamination
        return float(xp.sum(exp - self._data * xp.log(exp)))

class L2L1GradientNorm:
    def __init__(self, beta: float, G: RealLinearOperator) -> None:
        self._beta = beta
        self._G = G
    
    def __call__(self, x: npt.NDArray) -> float:
        return self._beta * xp.sum(xp.linalg.vector_norm(G.forward(x), axis=0))


def prox_dual_l2l1(y, sigma):
    gnorm = xp.linalg.vector_norm(y, axis=0)
    gnorm[gnorm < 1] = 1
    return y / gnorm

