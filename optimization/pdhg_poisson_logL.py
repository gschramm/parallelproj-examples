"""minimal example of PDHG algorithm for Poisson data fidelity and non-negativity constraint"""

import numpy as np
import array_api_compat.numpy as xp
from array_api_compat import to_device
import matplotlib.pyplot as plt
import parallelproj

from utils import negativePoissonLogL

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

P = parallelproj.ParallelViewProjector2D(img_shape, radial_positions, view_angles, radius, img_origin, voxel_size)
P.scale = 1.0 / P.norm()

# the ground truth image used to generate the data    
x_true = count_factor*xp.ones(P.in_shape, device=dev)
x_true[:8, :] = 0
x_true[-8:, :] = 0
x_true[:, :8] = 0
x_true[:, -8:] = 0


# setup known additive contamination and noise-free data
noisefree_data = P(x_true)
contamination = xp.full(P.out_shape, 0.5*xp.mean(noisefree_data))
noisefree_data += contamination

# add Poisson noise to the data
data = xp.asarray(np.random.poisson(to_device(noisefree_data, 'cpu')), device=dev, dtype = x_true.dtype)

# setup the cost function
cost_fct = lambda z: negativePoissonLogL(P(z) + contamination, data)

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
    cost_pdhg[i] = cost_fct(x)

    if i % 100 == 0:
        delta_rel = float(xp.linalg.vector_norm(delta) / xp.linalg.vector_norm(x))
        print(f'iteration {i:04} / {num_iter:05} | cost {cost_pdhg[i]:.7e} | delta_rel {delta_rel:.7e}')

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#---- MLEM as reference ---------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

x_mlem = xp.asarray(x0, copy = True, device=dev)
sens = P.adjoint(xp.ones_like(data))

cost_mlem = np.zeros(num_iter)

for i in range(num_iter):
    exp = P(x_mlem) + contamination
    x_mlem *= (P.adjoint(data / exp) / sens)

    # compute cost
    cost_mlem[i] = cost_fct(x_mlem)

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#---- show results --------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

max_diff = float(xp.max(xp.abs(x - x_mlem)))

print()
print(f'max diff between PDHG and MLEM .: {max_diff:.3e}')
print(f'max x                          .: {float(xp.max(x)):.3e}')

fig, ax = plt.subplots(2, 2, figsize=(7, 7))
ax[0,0].plot(cost_pdhg, label = 'PDHG')
ax[0,0].plot(cost_mlem, label = 'MLEM')
ax[0,0].legend()
ax[0,1].plot(cost_pdhg)
ax[0,1].plot(cost_mlem)

cmin = min(cost_mlem.min(), cost_pdhg.min())
ymax = cost_mlem[max(min(num_iter-10, 60),0):].max()
ax[0,1].set_ylim(cmin - 0.1*(ymax - cmin), ymax)
for axx in ax[0,:]:
    axx.grid(ls = ':')
    axx.set_xlabel('iteration')
    axx.set_ylabel('negative Poisson logL')

ax[1,0].imshow(x_true, vmin = 0, vmax = 1.1*xp.max(x_true))
ax[1,1].imshow(x, vmin = 0, vmax = 1.1*xp.max(x_true))

fig.tight_layout()
fig.show()
