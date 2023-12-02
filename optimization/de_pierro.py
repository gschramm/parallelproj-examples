"""minimal example of PDHG algorithm for Poisson data fidelity and non-negativity constraint compared to MLEM"""

import numpy as np
from array_api_compat import to_device
import matplotlib.pyplot as plt
from scipy.optimize import fmin_powell
import parallelproj
from parallelproj_utils import DemoPETScannerLORDescriptor, RegularPolygonPETProjector
from utils import negativePoissonLogL

np.random.seed(42)

import numpy.array_api as xp
dev = 'cpu'

#import array_api_compat.cupy as xp
#dev = 'cuda'

# input parameters
img_shape = (32, 32, 1)
voxel_size = (4., 4., 4.)
num_iter = 5000
count_factor = 100.
beta = 0.001

#----------------------------------------------------------------------------------------------------------------

# the ground truth image used to generate the data    
x_true = count_factor*xp.ones(img_shape, device=dev)
x_true[:8, :, :] = 0
x_true[-8:, :, :] = 0
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

# setup an intensity prior image
x_prior = 1.0 * x_true

# setup the cost function
cost_fct = lambda z: negativePoissonLogL(P(z) + contamination, data) + 0.5*beta*float(xp.sum((z - x_prior)**2))

# we initialize the images with the adjoint of the data operator
x0 = P.adjoint(data)

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#---- de Pierro recon with quadratic intensity prior ----------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

x_dep = xp.asarray(x0, copy = True, device=dev)
B = P.adjoint(xp.ones_like(data)) - beta * x_prior



cost_dep = np.zeros(num_iter)

for i in range(num_iter):
    exp = P(x_dep) + contamination
    A = x_dep * P.adjoint(data / exp)

    x_dep = 2*A / (xp.sqrt(B**2 + 4*beta*A) + B)

    # compute cost
    cost_dep[i] = cost_fct(x_dep)

    if i % 100 == 0:
        print(f'dePierro iteration {i:04} / {num_iter:05} | cost {cost_dep[i]:.7e}')


#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#---- check against brute force optimization ------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

# fmin_powell needs to operate on flat numpy CPU arrays
x0_powell = np.asarray(to_device(x_dep, 'cpu')).ravel()
cost_fct_wrapper = lambda z: cost_fct(xp.reshape(xp.asarray(z, device=dev), P.in_shape)) + int(xp.min(xp.asarray(z, device=dev)) < 0)*1e6 
ref = xp.reshape(xp.asarray(fmin_powell(cost_fct_wrapper, x0_powell, maxiter = 20), device = dev), P.in_shape)

ref_cost = cost_fct(ref)

print(f'dePierro cost : {float(cost_dep[-1]):.10e}')
print(f'ref      cost : {ref_cost:.10e}')
print(f'rel      diff : {((float(cost_dep[-1]) - ref_cost) / abs(ref_cost)):.10e}')

RMSE = xp.sqrt(xp.sum((x_dep - ref)**2))
PSNR = 20 * float(xp.log10(xp.max(xp.abs(x_true))) / RMSE)

print(f'PSNR dePierro vs ref: {PSNR:.4e}')

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#---- show results --------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

it = np.arange(1, num_iter + 1)

fig, ax = plt.subplots(2, 4, figsize=(12, 6))
ax[0,0].plot(it, np.asarray(to_device(cost_dep, 'cpu')), label = 'dePierro')
ax[0,0].legend()
ax[0,1].plot(it[500:], np.asarray(to_device(cost_dep[500:], 'cpu')))
ax[0,2].plot(it[-100:], np.asarray(to_device(cost_dep[-100:], 'cpu')))
ax[0,3].set_axis_off()

for axx in ax[0,:-1]:
    axx.grid(ls = ':')
    axx.set_xlabel('iteration')
    axx.set_ylabel('cost function')
    axx.axhline(ref_cost, ls = '--', color = 'k')
for axx in ax[1, :]:
    axx.set_axis_off()

im0 = ax[1,0].imshow(np.asarray(to_device(x_true, 'cpu')), vmin = 0, vmax = 1.1*xp.max(x_true), cmap = 'Greys')
im1 = ax[1,1].imshow(np.asarray(to_device(x_dep, 'cpu')), vmin = 0, vmax = 1.1*xp.max(x_true), cmap = 'Greys')
im2 = ax[1,2].imshow(np.asarray(to_device(ref, 'cpu')), vmin = 0, vmax = 1.1*xp.max(x_true), cmap = 'Greys')
dmax = float(xp.max(xp.abs(x_dep - ref)))
im3 = ax[1,3].imshow(np.asarray(to_device(x_dep - ref, 'cpu')), cmap = 'seismic', vmin = -dmax, vmax = dmax)

fig.colorbar(im0, ax = ax[1,0], location = 'bottom')
fig.colorbar(im1, ax = ax[1,1], location = 'bottom')
fig.colorbar(im2, ax = ax[1,2], location = 'bottom')
fig.colorbar(im3, ax = ax[1,3], location = 'bottom')

ax[1,0].set_title('ground truth')
ax[1,1].set_title(f'dePierro {num_iter} it.')
ax[1,2].set_title('ref')
ax[1,3].set_title('dePierro - ref')

fig.tight_layout()
fig.savefig('fig3.png')
