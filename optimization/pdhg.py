"""minimal example of PDHG algorithm for Poisson data fidelity and non-negativity constraint"""

import math
import numpy as np
import numpy.array_api as xp
from array_api_compat import to_device
import matplotlib.pyplot as plt

from utils import MatrixOperator, NegativePoissonLogL


#-----------------------------------------------------------------------------

if __name__ == '__main__':
    np.random.seed(42)
    dev = 'cpu'
    
    img_shape = (4,3)
    num_data_bins = 20

    # setup the data operator
    A = xp.asarray(np.random.rand(num_data_bins, math.prod(img_shape)), device=dev)
    A[A < 0.7] = 0
    P = MatrixOperator(A, img_shape)
    # normalize the data operator
    P.scale = 1.0 / P.norm()

    # the ground truth image used to generate the data    
    x_true = 1000*xp.asarray(np.random.rand(*P.in_shape), device=dev)
    
    # setup known additive contamination and noise-free data
    noisefree_data = P.forward(x_true)
    contamination = xp.full(P.out_shape, 0.5*xp.mean(noisefree_data))
    noisefree_data += contamination

    # add Poisson noise to the data
    data = xp.asarray(np.random.poisson(to_device(noisefree_data, 'cpu')), device=dev, dtype = x_true.dtype)

    # PDHG parameters + init
    #sigma = 1.
    sigma = float(1 / xp.max(x_true))
    theta = 1.
    tau = 0.99 / (sigma * P.norm()**2)

    num_iter = 2000

    x = P.adjoint(data)
    xbar = x
    y = 1 - data / (P.forward(x) + contamination)

    cost_fct = NegativePoissonLogL(data, contamination, P)
    cost_arr = np.zeros(num_iter)

    x_intermed = []

    for i in range(num_iter):
        # forward step
        y += (sigma * (P.forward(xbar) + contamination))
        # apply prox of convex conj of Poisson data fidelity
        y = 0.5 * ( y + 1 - xp.sqrt((y-1)**2 + 4*sigma*data) ) 
        # backward step
        x_new = x - tau * P.adjoint(y)
        # prox of G (indicated function of non-negativity constraint) -> projection in non-negative values
        x_new[x_new < 0] = 0
        # update of xbar
        delta = x_new - x
        xbar = x_new + theta*delta
        # update of x
        x = x_new

        # compute cost
        cost_arr[i] = cost_fct(x)
        # store intermediate results
        x_intermed.append(x)

        if i % 100 == 0:
            delta_rel = float(xp.linalg.vector_norm(delta) / xp.linalg.vector_norm(x))
            print(f'iteration {i:04} / {num_iter:05} | cost {cost_arr[i]:.7e} | delta_rel {delta_rel:.7e}')


fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(cost_arr)
ax.set_xlabel('iteration')
ax.set_ylabel('cost')
fig.show()