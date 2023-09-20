"""minimal example of PDHG algorithm for Poisson data fidelity and non-negativity constraint"""

import math
import numpy as np
import numpy.array_api as xp
from array_api_compat import to_device
import matplotlib.pyplot as plt

from utils import MatrixOperator, GradientOperator2D, prox_dual_l2l1, NegativePoissonLogL

np.random.seed(42)
dev = 'cpu'

# input parameters
img_shape = (16,14)
num_data_bins = 4 * img_shape[0] * img_shape[1]
num_iter = 2000
count_factor = 1000.
beta = 3e-2

# setup the data operator
A = xp.asarray(np.random.rand(num_data_bins, math.prod(img_shape)), device=dev)
A[A < 0.7] = 0
P = MatrixOperator(A, img_shape)
# normalize the data operator
P.scale = 1.0 / P.norm()

# the ground truth image used to generate the data    
x_true = count_factor*xp.ones(P.in_shape, device=dev)
x_true[:3, :] = 0
x_true[-3:, :] = 0
x_true[:, :3] = 0
x_true[:, -3:] = 0

# setup known additive contamination and noise-free data
noisefree_data = P.forward(x_true)
contamination = xp.full(P.out_shape, 0.5*xp.mean(noisefree_data))
noisefree_data += contamination

# add Poisson noise to the data
data = xp.asarray(np.random.poisson(to_device(noisefree_data, 'cpu')), device=dev, dtype = x_true.dtype)

# setup gradient operator
G = GradientOperator2D(img_shape)
# normalize the data operator
G.scale = 1.0 / G.norm()

# setup the cost function
data_fidelity = NegativePoissonLogL(data, contamination, P)
regularizer = lambda z: float(xp.sum(xp.linalg.vector_norm(G.forward(z), axis=0)))
cost_fct = lambda z: data_fidelity(z) + beta * regularizer(z)

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#---- PDHG ----------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

# (1) intialize variables x, xbar and y

# we initialize the images with the adjoint of the data operator
x0 = P.adjoint(data)
x = xp.asarray(x0, copy = True, device=dev)
xbar = xp.asarray(x0, copy = True, device=dev)
y_data = 1 - data / (P.forward(x) + contamination)

y_reg = xp.zeros(G.out_shape, dtype = x.dtype, device=dev)

# (2) set the step sizes sigma and tau

# for Poisson data it seems that sigma = 1 is not a good choice
# instead 1/scale(reconstructed image) seems to work better
#sigma = 1.
sigma = 1*float(1 / xp.max(x0))
tau = 0.99 / (sigma * P.norm()**2)
theta = 1.

cost_pdhg = np.zeros(num_iter)

for i in range(num_iter):
    # forward step for data
    y_data += (sigma * (P.forward(xbar) + contamination))
    # apply prox of convex conj of Poisson data fidelity
    y_data = 0.5 * ( y_data + 1 - xp.sqrt((y_data-1)**2 + 4*sigma*data) ) 

    # forward step for data
    y_reg += (sigma * G.forward(xbar))
    # apply prox of convex conj of Poisson data fidelity
    y_reg = beta * prox_dual_l2l1(y_reg / beta, sigma / beta)


    # backward step
    x_new = x - tau * (P.adjoint(y_data) + G.adjoint(y_reg))
    # prox of G (indicated function of non-negativity constraint) -> projection in non-negative values
    x_new[x_new < 0] = 0
    # update of xbar
    delta = x_new - x
    xbar = x_new + theta*delta
    # update of x
    x = x_new

    # compute cost
    cost_pdhg[i] = cost_fct(x)

    if i % 100 == 0:
        delta_rel = float(xp.linalg.vector_norm(delta) / xp.linalg.vector_norm(x))
        print(f'iteration {i:04} / {num_iter:05} | cost {cost_pdhg[i]:.7e} | delta_rel {delta_rel:.7e}')

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#---- show results --------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].plot(cost_pdhg, label = 'PDHG')
ax[0].legend()
ax[1].plot(cost_pdhg)

cmin = cost_pdhg.min()
ymax = cost_pdhg[max(min(num_iter-10, 20),0):].max()
ax[1].set_ylim(cmin - 0.1*(ymax - cmin), ymax)
for axx in ax:
    axx.grid(ls = ':')
    axx.set_xlabel('iteration')
    axx.set_ylabel('cost')
fig.tight_layout()
fig.show()
