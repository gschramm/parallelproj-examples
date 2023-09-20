"""minimal example of PDHG algorithm for Poisson data fidelity and non-negativity constraint"""

import numpy as np
import array_api_compat.numpy as xp
from array_api_compat import to_device
import matplotlib.pyplot as plt
import parallelproj

from utils import negativePoissonLogL, GradientOperator2D, prox_dual_l2l1

np.random.seed(42)
dev = 'cpu'

# input parameters
img_shape = (32, 32)
voxel_size = (1., 1.)
radial_positions = xp.linspace(-32, 32, 64)
view_angles = xp.linspace(0, xp.pi, 180, endpoint=False)
radius = 20
img_origin = (-15.5, -15.5)
num_iter = 1000
count_factor = 500.
beta = 0.1

P = parallelproj.ParallelViewProjector2D(img_shape, radial_positions, view_angles, radius, img_origin, voxel_size)
P.scale = 1.0 / P.norm()

# the ground truth image used to generate the data    
x_true = count_factor*xp.zeros(P.in_shape, device=dev)
x_true[8:-8, 8:-8] = count_factor
x_true[12:-12, 12:-12] = 2*count_factor


# setup known additive contamination and noise-free data
noisefree_data = P(x_true)
contamination = xp.full(P.out_shape, 0.5*xp.mean(noisefree_data))
noisefree_data += contamination

# add Poisson noise to the data
data = xp.asarray(np.random.poisson(to_device(noisefree_data, 'cpu')), device=dev, dtype = x_true.dtype)

# setup gradient operator
G = GradientOperator2D(img_shape)
# normalize the data operator
G.scale = 1.0 / G.norm()

# setup the cost function
data_fidelity = lambda z: negativePoissonLogL(P(z) + contamination, data) 
regularizer = lambda z: float(xp.sum(xp.linalg.vector_norm(G(z), axis=0)))
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
y_data = 1 - data / (P(x) + contamination)
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
    # forward step
    y_data += (sigma * (P(xbar) + contamination))
    # apply prox of convex conj of Poisson data fidelity
    y_data = 0.5 * ( y_data + 1 - xp.sqrt((y_data-1)**2 + 4*sigma*data) ) 

    # forward step for data
    y_reg += (sigma * G(xbar))
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

fig, ax = plt.subplots(2, 2, figsize=(7, 7))
ax[0,0].plot(cost_pdhg, label = 'PDHG')
ax[0,0].legend()
ax[0,1].plot(cost_pdhg)

cmin = min(cost_pdhg.min(), cost_pdhg.min())
ymax = cost_pdhg[max(min(num_iter-10, 60),0):].max()
ax[0,1].set_ylim(cmin - 0.1*(ymax - cmin), ymax)
for axx in ax[0,:]:
    axx.grid(ls = ':')
    axx.set_xlabel('iteration')
    axx.set_ylabel('cost function')

ax[1,0].imshow(x_true, vmin = 0, vmax = 1.1*xp.max(x_true))
ax[1,1].imshow(x, vmin = 0, vmax = 1.1*xp.max(x_true))

fig.tight_layout()
fig.show()
