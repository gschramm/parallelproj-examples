"""minimal example of PDHG algorithm for Poisson data fidelity and non-negativity constraint"""

import numpy as np
from array_api_compat import to_device
from scipy.optimize import fmin_powell
import matplotlib.pyplot as plt
import parallelproj
from parallelproj_utils import DemoPETScannerLORDescriptor, RegularPolygonPETProjector
from utils import negativePoissonLogL, prox_dual_l2l1

np.random.seed(42)

import numpy.array_api as xp
dev = 'cpu'

#import array_api_compat.cupy as xp
#dev = 'cpu'

# input parameters
img_shape = (32, 32, 1)
voxel_size = (4., 4., 4.)
num_iter = 5000
count_factor = 100.
sigma_fac = 1. # by default sigma = sigma_fac / max(P.adjoint(data)) where P is normalized operator
beta = 0.1
#----------------------------------------------------------------------------------------------------------------

# the ground truth image used to generate the data    
x_true = count_factor*xp.ones(img_shape, device=dev)
x_true[:2, :, :] = 0
x_true[-2:, :, :] = 0
x_true[:, :14, :] = 0
x_true[:, -14:, :] = 0

# setup an attenuation image
x_attn = xp.full(img_shape, 0.01)* xp.astype(x_true > 0, xp.float32)

lor_descriptor =  DemoPETScannerLORDescriptor(xp, dev, num_rings=1, radial_trim=221)
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

# setup gradient operator
G = parallelproj.FiniteForwardDifference(img_shape)
# normalize the data operator
G.scale = 1.0 / G.norm(xp, dev=dev)

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
sigma = sigma_fac * float(1 / xp.max(x0))
tau = 0.99 / (sigma * P.norm(xp, dev=dev)**2)
theta = 1.

cost_pdhg = xp.zeros(num_iter, device = 'cpu')

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
        print(f'PDHG iteration {i:04} / {num_iter:05} | cost {float(cost_pdhg[i]):.7e} | delta_rel {delta_rel:.7e}')

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#---- check against bute force optimization -------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

# fmin_powell needs to operate on flat numpy CPU arrays
x0_powell = np.asarray(to_device(x, 'cpu')).ravel()
cost_fct_wrapper = lambda z: cost_fct(xp.reshape(xp.asarray(z, device=dev), P.in_shape)) + int(xp.min(xp.asarray(z, device=dev)) < 0)*1e6 
ref = xp.reshape(xp.asarray(fmin_powell(cost_fct_wrapper, x0_powell, maxiter = 10), device = dev), P.in_shape)

ref_cost = cost_fct(ref)

print(f'PDHG cost : {float(cost_pdhg[-1]):.10e}')
print(f'ref  cost : {ref_cost:.10e}')
print(f'rel  diff : {((float(cost_pdhg[-1]) - ref_cost) / abs(ref_cost)):.10e}')

RMSE = xp.sqrt(xp.sum((x - ref)**2))
PSNR = 20 * float(xp.log10(xp.max(xp.abs(x_true))) / RMSE)

print(f'PSNR PDHG vs ref: {PSNR:.4e}')

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#---- show results --------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

it = np.arange(1, num_iter + 1)

fig, ax = plt.subplots(2, 4, figsize=(12, 6))
ax[0,0].plot(it, np.asarray(to_device(cost_pdhg, 'cpu')), label = 'PDHG')
ax[0,0].legend()
ax[0,1].plot(it[500:], np.asarray(to_device(cost_pdhg[500:], 'cpu')))
ax[0,2].plot(it[-100:], np.asarray(to_device(cost_pdhg[-100:], 'cpu')))
ax[0,3].set_axis_off()

for axx in ax[0,:-1]:
    axx.grid(ls = ':')
    axx.set_xlabel('iteration')
    axx.set_ylabel('cost function')
    axx.axhline(ref_cost, ls = '--', color = 'k')
for axx in ax[1, :]:
    axx.set_axis_off()

im0 = ax[1,0].imshow(np.asarray(to_device(x_true, 'cpu')), vmin = 0, vmax = 1.1*xp.max(x_true), cmap = 'Greys')
im1 = ax[1,1].imshow(np.asarray(to_device(x, 'cpu')), vmin = 0, vmax = 1.1*xp.max(x_true), cmap = 'Greys')
im2 = ax[1,2].imshow(np.asarray(to_device(ref, 'cpu')), vmin = 0, vmax = 1.1*xp.max(x_true), cmap = 'Greys')
dmax = float(xp.max(xp.abs(x - ref)))
im3 = ax[1,3].imshow(np.asarray(to_device(x - ref, 'cpu')), cmap = 'seismic', vmin = -dmax, vmax = dmax)

fig.colorbar(im0, ax = ax[1,0], location = 'bottom')
fig.colorbar(im1, ax = ax[1,1], location = 'bottom')
fig.colorbar(im2, ax = ax[1,2], location = 'bottom')
fig.colorbar(im3, ax = ax[1,3], location = 'bottom')

ax[1,0].set_title('ground truth')
ax[1,1].set_title(f'PHDG {num_iter} it.')
ax[1,2].set_title('ref')
ax[1,3].set_title('PDHG - ref')

fig.tight_layout()
fig.savefig('fig2.png')
