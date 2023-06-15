# ## example that shows how to MLEM reconstruction on cupy GPU arrays

# +
#import numpy as xp
#import scipy.ndimage as ndi
import cupy as xp
import cupyx.scipy.ndimage as ndi

from parallelproj.operators import CompositeLinearOperator, ElementwiseMultiplicationOperator, GaussianFilterOperator
from parallelproj.projectors import ParallelViewProjector2D
from parallelproj.utils import tonumpy

import matplotlib.pyplot as plt

from utils import generate_random_image
# -

# ### setup a batch of "random" ground truth images

# +
# seed the random generator, since we are using random images
xp.random.seed(0)
# image dimensions
n0, n1, n2 = (1, 128, 128)
num_images = 5
img_shape = (n0, n1, n2)

# voxel size of the image
voxel_size = xp.array([2., 2., 2.]).astype(xp.float32)

# image origin -> world coordinates of the [0,0,0] voxel
img_origin = ((-xp.array(img_shape) / 2 + 0.5) * voxel_size).astype(xp.float32)

img_batch = xp.zeros((num_images, n0, n1, n2), dtype=xp.float32)

for i in range(num_images):
    img_batch[i, ...] = generate_random_image(n0, n1, n2, xp, ndi)
# -

# ### setup a simple "2D" parallel view non-tof projector

# +
# setup the coordinates for projections along parallel views
num_rad = 223
num_phi = 190
# "radius" of the scanner in mm
scanner_R = 350.

# radial coordinates of the projection views in mm
r = xp.linspace(-200, 200, num_rad, dtype=xp.float32)
view_angles = xp.linspace(0, xp.pi, num_phi, endpoint=False, dtype=xp.float32)

projector = ParallelViewProjector2D(img_shape, r, view_angles, scanner_R,
                                    img_origin, voxel_size, xp)
# -

# ### show the projector geometry

# +
fig_proj = projector.show_views(image=img_batch[0, ...], cmap='Greys')
# -

# ### generate the attenuation images, attenuation sinograms, and sensitivity sinograms for our forward model

# +
att_img_batch = xp.zeros((num_images, n0, n1, n2), dtype=xp.float32)
att_sino_batch = xp.zeros((num_images, num_phi, num_rad), dtype=xp.float32)

for i in range(num_images):
    # the attenuation coefficients in 1/mm
    att_img_batch[i, ...] = 0.01 * (img_batch[i, ...] > 0).astype(xp.float32)
    att_sino_batch[i, ...] = xp.exp(-projector(att_img_batch[i, ...]))

# generate a constant sensitivity sinogram
sens_sino_batch = xp.full((num_images, ) + projector.out_shape,
                          0.3,
                          dtype=xp.float32)

# generate sinograms of multiplicative corrections (attention times sensitivity)
mult_corr_batch = att_sino_batch * sens_sino_batch
# -

# ###setup the complete forward model consisting of image-based resolution model projector and multiplication by sensitivity

# +
image_space_filter = GaussianFilterOperator(projector.in_shape,
                                            ndi,
                                            xp,
                                            sigma=4.5 / (2.35 * voxel_size))

# setup a projector including an image-based resolution model
projector_with_res_model = CompositeLinearOperator(
    (projector, image_space_filter))
# -

# +
# apply the forward model to generate noise-free data
img_fwd_batch = xp.zeros((num_images, num_phi, num_rad), dtype=xp.float32)
add_corr_batch = xp.zeros((num_images, num_phi, num_rad), dtype=xp.float32)
adjoint_ones_batch = xp.zeros((num_images, n0, n1, n2), dtype=xp.float32)

for i in range(num_images):
    img_fwd_batch[i, ...] = mult_corr_batch[i, ...] * projector_with_res_model(
        img_batch[i, ...])

    # generate a constant contamination sinogram
    add_corr_batch[i, ...] = xp.full(img_fwd_batch[i, ...].shape,
                                     0.5 * img_fwd_batch[i, ...].mean(),
                                     dtype=xp.float32)

# generate noisy data
data_batch = xp.random.poisson(img_fwd_batch + add_corr_batch)
# -

# ### generate the sensitivity images (adjoint operator applied to a sinogram of ones)

# +
# create the sensitivity images (adjoint applied to "ones")
for i in range(num_images):
    adjoint_ones_batch[i, ...] = projector_with_res_model.adjoint(
        mult_corr_batch[i, ...])

# -

# ### show all input data

# +
fig, ax = plt.subplots(4, num_images, figsize=(2.5 * num_images, 2.5 * 4))
for i in range(num_images):
    im0 = ax[0, i].imshow(tonumpy(img_batch[i, n0 // 2, ...], xp),
                          cmap='Greys',
                          vmin=0,
                          vmax=float(1.2 * img_batch.max()))
    im1 = ax[1, i].imshow(tonumpy(att_img_batch[i, n0 // 2, ...], xp),
                          cmap='Greys',
                          vmin=0,
                          vmax=float(att_img_batch.max()))
    im2 = ax[2, i].imshow(tonumpy(mult_corr_batch[i, ...], xp),
                          cmap='Greys',
                          vmin=0,
                          vmax=float(mult_corr_batch.max()))
    im3 = ax[3, i].imshow(tonumpy(data_batch[i, ...], xp),
                          cmap='Greys',
                          vmin=0,
                          vmax=float(data_batch.max()))

    cb0 = fig.colorbar(im0, fraction=0.03, location='bottom')
    cb1 = fig.colorbar(im1, fraction=0.03, location='bottom')
    cb2 = fig.colorbar(im2, fraction=0.03, location='bottom')
    cb3 = fig.colorbar(im3, fraction=0.03, location='bottom')

    ax[0, i].set_title(f'ground truth image {i:03}', fontsize='medium')
    ax[1, i].set_title(f'attenuation image {i:03}', fontsize='medium')
    ax[2, i].set_title(f'mult. corr. sinogram {i:03}', fontsize='medium')
    ax[3, i].set_title(f'noisy data sinogram {i:03}', fontsize='medium')

for axx in ax.ravel():
    axx.axis('off')
fig.tight_layout()
# -

### run MLEM

# +
num_iter = 100

x0_batch = xp.ones(img_batch.shape, dtype=xp.float32)
x_batch = x0_batch.copy()

for it in range(num_iter):
    print(f'it {(it+1):04} / {num_iter:04}', end='\r')
    for ib in range(num_images):
        exp = mult_corr_batch[ib, ...] * projector_with_res_model(
            x_batch[ib, ...]) + add_corr_batch[ib, ...]
        x_batch[ib, ...] *= (projector_with_res_model.adjoint(
            mult_corr_batch[ib, ...] * data_batch[ib, ...] / exp) /
                             adjoint_ones_batch[ib, ...])
print('')
# -

# ### show all MLEM reconstructions

# +
figm, axm = plt.subplots(1,
                         num_images,
                         figsize=(2.5 * num_images, 2.5 * 1),
                         squeeze=False)
for i in range(num_images):
    im0 = axm[0, i].imshow(tonumpy(x_batch[i, n0 // 2, ...], xp),
                           cmap='Greys',
                           vmin=0,
                           vmax=float(1.2 * img_batch.max()))

    cb0 = figm.colorbar(im0, fraction=0.03, location='bottom')

    axm[0, i].set_title(f'MLEM {i:03} - {num_iter:03} it.', fontsize='medium')

for axx in axm.ravel():
    axx.axis('off')
figm.tight_layout()

plt.show()
# -
