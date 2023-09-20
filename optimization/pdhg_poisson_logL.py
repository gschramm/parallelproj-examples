"""minimal example of PDHG algorithm for Poisson data fidelity and non-negativity constraint"""

import math
import numpy as np
import numpy.array_api as xp
from array_api_compat import to_device
import matplotlib.pyplot as plt

from utils import MatrixOperator, NegativePoissonLogL

np.random.seed(42)
dev = 'cpu'

# input parameters
img_shape = (4,3)
num_data_bins = 20
num_iter = 1000
count_factor = 10.

# setup the data operator
A = xp.asarray(np.random.rand(num_data_bins, math.prod(img_shape)), device=dev)
A[A < 0.7] = 0
P = MatrixOperator(A, img_shape)
# normalize the data operator
P.scale = 1.0 / P.norm()

# the ground truth image used to generate the data    
x_true = count_factor*xp.asarray(np.random.rand(*P.in_shape), device=dev)

# setup known additive contamination and noise-free data
noisefree_data = P.forward(x_true)
contamination = xp.full(P.out_shape, 0.5*xp.mean(noisefree_data))
noisefree_data += contamination

# add Poisson noise to the data
data = xp.asarray(np.random.poisson(to_device(noisefree_data, 'cpu')), device=dev, dtype = x_true.dtype)

# setup the cost function
cost_fct = NegativePoissonLogL(data, contamination, P)

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#---- PHDG ----------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

# (1) intialize variables x, xbar and y

# we initialize the images with the adjoint of the data operator
x0 = P.adjoint(data)
x = xp.asarray(x0, copy = True, device=dev)
xbar = xp.asarray(x0, copy = True, device=dev)
y_data = 1 - data / (P.forward(x) + contamination)

# (2) set the step sizes sigma and tau

# for Poisson data it seems that sigma = 1 is not a good choice
# instead 1/scale(reconstructed image) seems to work better
#sigma = 1.
sigma = 1*float(1 / xp.max(x0))
tau = 0.99 / (sigma * P.norm()**2)
theta = 1.

cost_arr = np.zeros(num_iter)

for i in range(num_iter):
    # forward step
    y_data += (sigma * (P.forward(xbar) + contamination))
    # apply prox of convex conj of Poisson data fidelity
    y_data = 0.5 * ( y_data + 1 - xp.sqrt((y_data-1)**2 + 4*sigma*data) ) 
    # backward step
    x_new = x - tau * P.adjoint(y_data)
    # prox of G (indicated function of non-negativity constraint) -> projection in non-negative values
    x_new[x_new < 0] = 0
    # update of xbar
    delta = x_new - x
    xbar = x_new + theta*delta
    # update of x
    x = x_new

    # compute cost
    cost_arr[i] = cost_fct(x)

    if i % 100 == 0:
        delta_rel = float(xp.linalg.vector_norm(delta) / xp.linalg.vector_norm(x))
        print(f'iteration {i:04} / {num_iter:05} | cost {cost_arr[i]:.7e} | delta_rel {delta_rel:.7e}')

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#---- MLEM as reference ---------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

x_mlem = xp.asarray(x0, copy = True, device=dev)
sens = P.adjoint(xp.ones_like(data))

cost_arr_mlem = np.zeros(num_iter)

for i in range(num_iter):
    exp = P.forward(x_mlem) + contamination
    x_mlem *= (P.adjoint(data / exp) / sens)

    # compute cost
    cost_arr_mlem[i] = cost_fct(x_mlem)

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#---- show results --------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].plot(cost_arr, label = 'PDHG')
ax[0].plot(cost_arr_mlem, label = 'MLEM')
ax[0].legend()
ax[1].plot(cost_arr)
ax[1].plot(cost_arr_mlem)

cmin = min(cost_arr_mlem.min(), cost_arr.min())
ymax = cost_arr_mlem[max(min(num_iter-10, 50),0):].max()
ax[1].set_ylim(cmin - 0.1*(ymax - cmin), ymax)
for axx in ax:
    axx.grid(ls = ':')
    axx.set_xlabel('iteration')
    axx.set_ylabel('negative Poisson logL')
fig.tight_layout()
fig.show()
