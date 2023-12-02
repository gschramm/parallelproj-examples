"""minimal example of PDHG algorithm for Poisson data fidelity and non-negativity constraint compared to MLEM"""

import numpy as np
from array_api_compat import to_device
import matplotlib.pyplot as plt
import parallelproj
from parallelproj_utils import DemoPETScannerLORDescriptor, RegularPolygonPETProjector
from utils import negativePoissonLogL

np.random.seed(42)

import numpy.array_api as xp
dev = 'cpu'

#import array_api_compat.cupy as xp
#dev = 'cuda'

# input parameters
img_shape = (64, 64, 2)
voxel_size = (4., 4., 4.)
num_iter = 1000
count_factor = 100.
sigma_fac = 1. # by default sigma = sigma_fac / max(P.adjoint(data)) where P is normalized operator

#----------------------------------------------------------------------------------------------------------------

# the ground truth image used to generate the data    
x_true = count_factor*xp.ones(img_shape, device=dev)
x_true[:8, :, :] = 0
x_true[-8:, :, :] = 0
x_true[:, :30, :] = 0
x_true[:, -30:, :] = 0

# setup an attenuation image
x_attn = xp.full(img_shape, 0.01)* xp.astype(x_true > 0, xp.float32)

lor_descriptor =  DemoPETScannerLORDescriptor(xp, dev, num_rings=2, radial_trim=181)
P = RegularPolygonPETProjector(lor_descriptor, img_shape, voxel_size=voxel_size)

# calcuate an attenuation sinogram
attn_sino = xp.exp(-P(x_attn))

# add the correction for attenuation and resolution to the operator
P = parallelproj.CompositeLinearOperator((parallelproj.ElementwiseMultiplicationOperator(attn_sino),P, parallelproj.GaussianFilterOperator(img_shape, 0.6)))

# rescale the operator to norm 1
P.scale = 1.0 / P.norm(xp, dev=dev)

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
sigma = sigma_fac * float(1 / xp.max(x0))
tau = 0.99 / (sigma * P.norm(xp, dev=dev)**2)
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
        print(f'PDHG iteration {i:04} / {num_iter:05} | cost {cost_pdhg[i]:.7e} | delta_rel {delta_rel:.7e}')

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

    if i % 100 == 0:
        delta_rel = float(xp.linalg.vector_norm(delta) / xp.linalg.vector_norm(x))
        print(f'MLEM iteration {i:04} / {num_iter:05} | cost {cost_mlem[i]:.7e}')

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#---- show results --------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

sl = img_shape[2] // 2

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

ax[1,0].imshow(np.asarray(to_device(x_true[:,:,sl], 'cpu')), vmin = 0, vmax = 1.1*xp.max(x_true))
ax[1,1].imshow(np.asarray(to_device(x[:,:,sl], 'cpu')), vmin = 0, vmax = 1.1*xp.max(x_true))

fig.tight_layout()
fig.savefig('fig1.png')
