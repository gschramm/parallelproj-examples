# # Unrolled MLEM and variational networks for Poisson projection data

# ## Part 1: MLEM using cupy GPU arrays
#
# In this part, we will:
# - simulate a data base of random 3D objects
# - setup a projector for a "small" generic 3D non-TOF PET scanner 
# - generate noisy measured projection data
# - use MLEM to reconstruct all data sets

# +
#--- import of external python modules we need ---
import numpy as np

import cupy as cp
import cupyx.scipy.ndimage as ndi

from parallelproj.operators import CompositeLinearOperator, GaussianFilterOperator
from parallelproj.utils import tonumpy

import matplotlib.pyplot as plt

#--- import of custom python modules ---
from utils import ParallelViewProjector3D
from shapes import generate_random_3d_image
# -

# ### 1.1 setup a batch of "random" ground truth images
#
# - use a custom function to generate a data base of random 3D blob-like images
# - we store all images directly as cupy GPU arrays

# +
# seed the random generator, since we are using random images
seed = 0
cp.random.seed(seed)
np.random.seed(seed)

#----------------------------------------------------
#--- image input parameters -------------------------

# set the total number of images to be generated
num_images = 30

# set the image size in trans-axial and axial direction
n_trans = 128
n_ax = 16

# set the voxel size of the images (in mm)
voxel_size = cp.array([2., 2., 2.]).astype(cp.float32)

#----------------------------------------------------
#----------------------------------------------------

# setup a tuple containing the image shape (dimensions)
img_shape = (n_trans, n_trans, n_ax)


# setup the "image origin" -> world coordinates of the [0,0,0] voxel
img_origin = ((-cp.array(img_shape) / 2 + 0.5) * voxel_size).astype(cp.float32)

# allocated an empty cupy GPU array for all 3D images
img_dataset = cp.zeros((num_images, ) + img_shape, dtype=cp.float32)
# generate the dataset of random 3D images
for i in range(num_images):
    img_dataset[i, ...] = generate_random_3d_image(n_trans, n_ax, cp, ndi)
    
print(f'image data base shape (# images, # n0, # n1, # n2) :{img_dataset.shape}')
# -

# ### 1.2 setup a simple "3D" parallel view non-tof projector
#
# - setup a projector for a generic 3D parallel view projector
# - to accelerate projection times, we setup a projector with a small ring difference

# +
# setup the coordinates for projections along parallel views

num_rad = 111  # number of radial elements in the sinogram
num_phi = 190  # number of views in the sinogram
scanner_R = 350.  # "radius" of the scanner in mm


rmax = 1.4 * float(voxel_size[0] * n_trans / 2)
ring_spacing = 4.

#-------------------------------------------------------------
#-------------------------------------------------------------

# radial coordinates of the projection views in mm
r = cp.linspace(-rmax, rmax, int(2 * rmax / 3.), dtype=cp.float32)
view_angles = cp.linspace(0, cp.pi, num_phi, endpoint=False, dtype=cp.float32)


num_rings = int((img_shape[2] - 1) * voxel_size[2] / ring_spacing) + 1
ring_positions = cp.linspace(img_origin[2],
                             img_origin[2] +
                             (img_shape[2] - 1) * voxel_size[2],
                             num_rings,
                             dtype=cp.float32)

projector = ParallelViewProjector3D(img_shape,
                                    r,
                                    view_angles,
                                    ring_positions,
                                    scanner_R,
                                    img_origin,
                                    voxel_size,
                                    cp,
                                    max_ring_diff=5)

print(f'projector input (image) shape (# n0, # n1, # n2) :{projector.in_shape}')
print(f'projector output (sinogram) shape (# views, # radial elements, # projection planes) :{projector.out_shape}')
# -

# ### 1.3 setup of multiplicative correction (attenuation and sensitivity) sinograms
#
# Setup the multiplicative correction sinograms we need in our forward model:
#
# $$y = A \,x + s $$ 
#
# with the forward operator
#
# $$ A = M \, PG$$
#
# where $G$ is a Gaussian convolution in image space, $P$ is the (geometrical) forward projection, and $M$ is a diagonal matrix (point-wise multiplication with attenuation and sensitivity sinogram). 

# +
# allocate memory for attenuation images and sinograms
att_img_dataset = cp.zeros((num_images, ) + img_shape, dtype=cp.float32)
att_sino_dataset = cp.zeros((num_images, ) + projector.out_shape,
                            dtype=cp.float32)

for i in range(num_images):
    # the attenuation coefficients in 1/mm
    # assume that our objects contain water attenuation
    att_img_dataset[i,
                    ...] = 0.01 * (img_dataset[i, ...] > 0).astype(cp.float32)
    att_sino_dataset[i, ...] = cp.exp(-projector(att_img_dataset[i, ...]))

# generate a constant sensitivity sinogram
# this values can be used to control the number of simulated counts (the noise level) 
sens_value = 0.2
sens_sino_dataset = cp.full((num_images, ) + projector.out_shape,
                            sens_value,
                            dtype=cp.float32)

# multiply each sensitivty sinogram with a random factor between 0.5 and 2.0
# to simulate differences in injected dose and acq. time
for i in range(num_images):
    sens_sino_dataset[i, ...] *= float(cp.random.uniform(0.75, 1.25))

# generate sinograms of multiplicative corrections (attention times sensitivity)
mult_corr_dataset = att_sino_dataset * sens_sino_dataset
# -

# ### 1.4 setup an image-based resolution model
#
# - setup the image-based Gaussian convolution operator $G$ that we use to simulate the limited resolution of our scanner

# +
simulated_resolution_mm = 4.5

image_space_filter = GaussianFilterOperator(projector.in_shape,
                                            ndi,
                                            cp,
                                            sigma=simulated_resolution_mm / (2.35 * voxel_size))

# setup a projector including an image-based resolution model
# in the notation of our forward model this is (PG)
projector_with_res_model = CompositeLinearOperator(
    (projector, image_space_filter))
# -

# ### 1.5 apply the forward model to our simulated ground truth images
#
# we evaluate $$M\,PG x$$ for all simulated images

# +
# allocated memory for the noise-free forward projections
img_fwd_dataset = cp.zeros((num_images, ) + projector_with_res_model.out_shape,
                           dtype=cp.float32)

# apply the forward model to generate noise-free data
for i in range(num_images):
    img_fwd_dataset[i,
                    ...] = mult_corr_dataset[i,
                                             ...] * projector_with_res_model(
                                                 img_dataset[i, ...])
# -

# ### 1.6 generate the additive (scatter + randoms) correction sinograms
#
# - we setup the expectation of the additive correction sinograms $s$ assuming a constant background contamination

# +
add_corr_dataset = cp.zeros(
    (num_images, ) + projector_with_res_model.out_shape, dtype=cp.float32)


for i in range(num_images):
    # generate a constant contamination sinogram
    add_corr_dataset[i, ...] = cp.full(img_fwd_dataset[i, ...].shape,
                                       0.5 * img_fwd_dataset[i, ...].mean(),
                                       dtype=cp.float32)
# -

# ### 1.7 generate noisy projection data following a Poisson distribution
#
# $$d = \text{Poisson}(A\,x + s)$$

# generate noisy data
data_dataset = cp.random.poisson(img_fwd_dataset + add_corr_dataset)

# ### 1.8 generate the sensitivity images (adjoint operator applied to a sinogram of ones)
#
# For MLEM, we need the "sensitivy images" 
#
# $$ b = A^H \mathbb{1} = (PG)^H M^H \mathbb{1} $$
#
#

# +
# create the sensitivity images (adjoint applied to "ones")
adjoint_ones_dataset = cp.zeros((num_images, ) + img_shape, dtype=cp.float32)

for i in range(num_images):
    adjoint_ones_dataset[i, ...] = projector_with_res_model.adjoint(
        mult_corr_dataset[i, ...])
# -

# ### 1.9 show the first 5 data sets
#
# - show the first 5 simulated activity images, attenuation images, multiplicative correction sinograms and noisy emission sinograms
# - we show the "central slice" of the image and the "central slice" of the direct sinogram planes

# +
vmax = float(1.1 * img_dataset.max())
fig, ax = plt.subplots(4, 5, figsize=(2.5 * 5, 2.5 * 4))

img_sl = n_ax // 2
sino_sl = num_rings // 2

for i in range(5):
    im0 = ax[0, i].imshow(tonumpy(img_dataset[i, ..., img_sl], cp),
                          cmap='Greys',
                          vmin=0,
                          vmax=vmax)
    im1 = ax[1, i].imshow(tonumpy(att_img_dataset[i, ..., img_sl], cp),
                          cmap='Greys',
                          vmin=0,
                          vmax=float(att_img_dataset.max()))
    im2 = ax[2, i].imshow(tonumpy(mult_corr_dataset[i, ..., sino_sl], cp),
                          cmap='Greys',
                          vmin=0,
                          vmax=float(mult_corr_dataset.max()))
    im3 = ax[3, i].imshow(tonumpy(data_dataset[i, ..., sino_sl], cp),
                          cmap='Greys',
                          vmin=0,
                          vmax=float(data_dataset.max()))

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

# ## 1.10 reconstruct all simulated data sets using early-stopped MLEM
#
# we use 100 MLEM updates
#
# $$ x^+ = \frac{x}{A^H \mathbb{1}} A^H \frac{d}{A\,x + s}$$
#
# $$ x^+ = \frac{x}{(PG)^H M^H \mathbb{1}} (PG)^H M^H \frac{d}{M\, PG \,x + s}$$
#
# to reconstruct all our simulated data sets.
#
# **Note that our MLEM implementation runs directly on our GPU meaning that there is no memory transfer between host and GPU.**
#

# +
num_iter_mlem = 100

x0_mlem_dataset = cp.ones(img_dataset.shape, dtype=cp.float32)
x_mlem_dataset = x0_mlem_dataset.copy()

for it in range(num_iter_mlem):
    print(f'cupy  MLEM it {(it+1):04} / {num_iter_mlem:04}', end='\r')
    for ib in range(num_images):
        exp = mult_corr_dataset[ib, ...] * projector_with_res_model(
            x_mlem_dataset[ib, ...]) + add_corr_dataset[ib, ...]
        x_mlem_dataset[ib, ...] *= (projector_with_res_model.adjoint(
            mult_corr_dataset[ib, ...] * data_dataset[ib, ...] / exp) /
                                    adjoint_ones_dataset[ib, ...])
print('')
# -

# ### 1.11 show MLEM reconstructions
#
# we show the MLEM reconstructions of the first 5 data sets and compare them to the ground truth images

# +
figm, axm = plt.subplots(3, 5, figsize=(2.5 * 5, 2.5 * 3))
for i in range(5):
    im0 = axm[0, i].imshow(tonumpy(img_dataset[i, ..., img_sl], cp),
                           cmap='Greys',
                           vmin=0,
                           vmax=vmax)
    im1 = axm[1, i].imshow(tonumpy(x_mlem_dataset[i, ..., img_sl], cp),
                           cmap='Greys',
                           vmin=0,
                           vmax=vmax)
    im2 = axm[2, i].imshow(tonumpy(
        ndi.gaussian_filter(x_mlem_dataset[i, ..., img_sl], 1.3), cp),
                           cmap='Greys',
                           vmin=0,
                           vmax=vmax)

    cb0 = figm.colorbar(im0, fraction=0.03, location='bottom')
    cb1 = figm.colorbar(im1, fraction=0.03, location='bottom')
    cb2 = figm.colorbar(im2, fraction=0.03, location='bottom')

    axm[0, i].set_title(f'ground truth image {i:03}', fontsize='medium')
    axm[1, i].set_title(f'MLEM {i:03}', fontsize='medium')
    axm[2, i].set_title(f'post-smoothed MLEM {i:03}', fontsize='medium')

for axx in axm.ravel():
    axx.axis('off')
figm.tight_layout()
# -

# ---
# ---
# ---
# ---
# ---
# ---
# ---
# ---
# ---

# ## Part 2: MLEM using an unrolled torch network

# +
# define a torch module that performs an MLEM update - possible with zero copy
import torch
# import a custom torch module that computes an MLEM update
from torch_utils import PoissonEMModule

em_module = PoissonEMModule(projector_with_res_model)

# +
# convert our cupy GPU arrays to torch GPU arrays
# pytorch and cuda support zero copy data exchange
# see https://docs.cupy.dev/en/stable/user_guide/interoperability.html#pytorch
# (mini batch) data in torch should have the shape (batch, channel, spatial/data shape)
# which is why we add extra dummy channel dimension via unsqueeze(1)

img_dataset_t = torch.from_dlpack(img_dataset).unsqueeze(1)
data_dataset_t = torch.from_dlpack(data_dataset).unsqueeze(1)
mult_corr_dataset_t = torch.from_dlpack(mult_corr_dataset).unsqueeze(1)
add_corr_dataset_t = torch.from_dlpack(add_corr_dataset).unsqueeze(1)
adjoint_ones_dataset_t = torch.from_dlpack(adjoint_ones_dataset).unsqueeze(1)

# initialize a batch array for the reconstructions
x0_mlem_dataset_t = torch.ones(img_dataset_t.shape,
                               dtype=torch.float32,
                               device=data_dataset_t.device)
x_mlem_dataset_t = torch.clone(x0_mlem_dataset_t)

# -

# run MLEM using a custom defined EM_Module
for it in range(num_iter_mlem):
    print(f'torch MLEM it {(it+1):04} / {num_iter_mlem:04}', end='\r')
    x_mlem_dataset_t = em_module.forward(x_mlem_dataset_t, data_dataset_t,
                                         mult_corr_dataset_t,
                                         add_corr_dataset_t,
                                         adjoint_ones_dataset_t)
print('')
# +
# convert the torch MLEM reconstruction array back to cupy for visualization
x_mlem_dataset_torch = cp.ascontiguousarray(
    cp.from_dlpack(x_mlem_dataset_t.detach())).squeeze(1)

# calculate the max difference between the cupy and torch MLEM implementation
print(
    f'max cupy - torch MLEM diff: {cp.abs(x_mlem_dataset - x_mlem_dataset_torch).max()}'
)

# +
# visualize the torch reconstructions
figm, axm = plt.subplots(3, 5, figsize=(2.5 * 5, 2.5 * 3))
for i in range(5):
    im0 = axm[0, i].imshow(tonumpy(img_dataset[i, ..., img_sl], cp),
                           cmap='Greys',
                           vmin=0,
                           vmax=vmax)
    im1 = axm[1, i].imshow(tonumpy(x_mlem_dataset[i, ..., img_sl], cp),
                           cmap='Greys',
                           vmin=0,
                           vmax=vmax)
    im2 = axm[2, i].imshow(tonumpy(x_mlem_dataset_torch[i, ..., img_sl], cp),
                           cmap='Greys',
                           vmin=0,
                           vmax=vmax)

    cb0 = figm.colorbar(im0, fraction=0.03, location='bottom')
    cb1 = figm.colorbar(im1, fraction=0.03, location='bottom')
    cb2 = figm.colorbar(im2, fraction=0.03, location='bottom')

    axm[0, i].set_title(f'ground truth image {i:03}', fontsize='medium')
    axm[1, i].set_title(f'cupy MLEM {i:03} - {num_iter_mlem:03} it.',
                        fontsize='medium')
    axm[2, i].set_title(f'torch MLEM {i:03} - {num_iter_mlem:03} it.',
                        fontsize='medium')

for axx in axm.ravel():
    axx.axis('off')
figm.tight_layout()
# -

# ---
# ---
# ---
# ---
# ---
# ---
# ---
# ---
# ---

# ## Part 3: Supervised training of a unrolled variational network

# +
from torch_utils import UnrolledVarNet, Unet3D
import torchmetrics

# setup a simple CNN that maps an image batch onto an image batch
conv_net = Unet3D(num_features=16, num_downsampling_layers=3)

# setup the unrolled variational network consiting of block combining MLEM and conv-net updates
var_net = UnrolledVarNet(em_module, num_blocks=5, neural_net=conv_net)

num_updates = 1001
batch_size = 5
num_train = int(0.8 * num_images)
learning_rate = 1e-3
data_range = float(img_dataset_t.max())
ssim_fn = torchmetrics.StructuralSimilarityIndexMeasure(
    data_range=data_range).to(x_mlem_dataset_t.device)
psnr_fn = torchmetrics.PeakSignalNoiseRatio(data_range=data_range).to(
    x_mlem_dataset_t.device)

#loss_fn = torch.nn.MSELoss()
loss_fn = torch.nn.L1Loss()

optimizer = torch.optim.Adam(var_net.parameters(), lr=learning_rate)

training_loss = np.zeros(num_updates)
validation_loss = []
validation_ssim = []
validation_psnr = []

# feed a mini batch through the network
for update in range(num_updates):
    var_net.train()
    # select a random mini batch from the traning data sets
    i_batch = np.random.choice(np.arange(num_train),
                               size=batch_size,
                               replace=False)
    y_train_t = var_net.forward(x_mlem_dataset_t[i_batch, ...],
                                data_dataset_t[i_batch, ...],
                                mult_corr_dataset_t[i_batch, ...],
                                add_corr_dataset_t[i_batch, ...],
                                adjoint_ones_dataset_t[i_batch, ...],
                                verbose=False)

    loss = loss_fn(y_train_t, img_dataset_t[i_batch, ...])
    training_loss[update] = loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # calculate the validation loss
    if update % 10 == 0:
        var_net.eval()
        with torch.no_grad():
            y_val_t = var_net.forward(x_mlem_dataset_t[num_train:, ...],
                                      data_dataset_t[num_train:, ...],
                                      mult_corr_dataset_t[num_train:, ...],
                                      add_corr_dataset_t[num_train:, ...],
                                      adjoint_ones_dataset_t[num_train:, ...],
                                      verbose=False)

        validation_loss.append(
            loss_fn(y_val_t, img_dataset_t[num_train:, ...]).item())
        validation_ssim.append(
            ssim_fn(y_val_t, img_dataset_t[num_train:, ...]).item())
        validation_psnr.append(
            psnr_fn(y_val_t, img_dataset_t[num_train:, ...]).item())

        print(
            f'update: {update:05} / {num_updates:05} - train loss: {loss.item():.2E} - val loss {validation_loss[-1]:.2E} - val ssim {validation_ssim[-1]:.2E} - val psnr {validation_psnr[-1]:.2E}',
            end='\r')

print(f'\nconv net weights {var_net._neural_net_weight}')

print(f'min training   loss {training_loss.min():.2E}')
print(f'min validation loss {min(validation_loss):.2E}')
print(f'max validation ssim {max(validation_ssim):.2E}')
print(f'max validation psnr {max(validation_psnr):.2E}')

# +
# plot the training and validation loss
i_train = np.arange(1, num_updates + 1)
i_val = np.arange(len(validation_loss)) * 10 + 1

figl, axl = plt.subplots(1, 3, figsize=(10, 4), sharex=True)
axl[0].plot(training_loss, '.-', label='training loss')
axl[0].plot(i_val, validation_loss, '.-', label='validation loss')
axl[0].set_ylim(0, training_loss[10:].max())
axl[0].legend()
axl[0].grid(ls=':')
axl[0].set_title('losses')
axl[1].plot(i_val, validation_ssim, '.-')
axl[1].grid(ls=':')
axl[1].set_title('validation ssim')
axl[2].plot(i_val, validation_psnr, '.-')
axl[2].grid(ls=':')
axl[2].set_title('validation PSNR')
axl[0].set_xlabel('update')
axl[1].set_xlabel('update')
axl[2].set_xlabel('update')
figl.tight_layout()

# +
# inference and plots of results
var_net.eval()

with torch.no_grad():
    i_train = np.arange(num_train)
    y_train_t = var_net.forward(x_mlem_dataset_t[i_train, ...],
                                data_dataset_t[i_train, ...],
                                mult_corr_dataset_t[i_train, ...],
                                add_corr_dataset_t[i_train, ...],
                                adjoint_ones_dataset_t[i_train, ...],
                                verbose=False)

with torch.no_grad():
    i_val = np.arange(num_train, num_images)
    y_val_t = var_net.forward(x_mlem_dataset_t[i_val, ...],
                              data_dataset_t[i_val, ...],
                              mult_corr_dataset_t[i_val, ...],
                              add_corr_dataset_t[i_val, ...],
                              adjoint_ones_dataset_t[i_val, ...],
                              verbose=False)

y_train_dataset = cp.ascontiguousarray(cp.from_dlpack(
    y_train_t.detach())).squeeze(1)
y_val_dataset = cp.ascontiguousarray(cp.from_dlpack(
    y_val_t.detach())).squeeze(1)

# visualize the torch reconstructions
figm, axm = plt.subplots(4, 5, figsize=(2.5 * 5, 2.5 * 4))
for i in range(5):
    im0 = axm[0, i].imshow(tonumpy(img_dataset[i, ..., img_sl], cp),
                           cmap='Greys',
                           vmin=0,
                           vmax=vmax)
    im1 = axm[1, i].imshow(tonumpy(x_mlem_dataset[i, ..., img_sl], cp),
                           cmap='Greys',
                           vmin=0,
                           vmax=vmax)
    im2 = axm[2, i].imshow(tonumpy(
        ndi.gaussian_filter(x_mlem_dataset[i, ..., img_sl], 1.3), cp),
                           cmap='Greys',
                           vmin=0,
                           vmax=vmax)
    im3 = axm[3, i].imshow(tonumpy(y_train_dataset[i, ..., img_sl], cp),
                           cmap='Greys',
                           vmin=0,
                           vmax=vmax)

    cb0 = figm.colorbar(im0, fraction=0.03, location='bottom')
    cb1 = figm.colorbar(im1, fraction=0.03, location='bottom')
    cb2 = figm.colorbar(im2, fraction=0.03, location='bottom')
    cb3 = figm.colorbar(im3, fraction=0.03, location='bottom')

    axm[0, i].set_title(f'ground truth image {i:03}', fontsize='medium')
    axm[1, i].set_title(f'MLEM {i:03} - {num_iter_mlem:03} it.',
                        fontsize='medium')
    axm[2, i].set_title(f'p.s. MLEM {i:03}', fontsize='medium')
    axm[3, i].set_title(f'varnet {i:03}', fontsize='medium')

for axx in axm.ravel():
    axx.axis('off')
figm.suptitle('First 5 training data sets', fontsize=20)
figm.tight_layout()

figv, axv = plt.subplots(4, 5, figsize=(2.5 * 5, 2.5 * 4))
for i in range(5):
    j = i + num_train
    im0 = axv[0, i].imshow(tonumpy(img_dataset[j, ..., img_sl], cp),
                           cmap='Greys',
                           vmin=0,
                           vmax=vmax)
    im1 = axv[1, i].imshow(tonumpy(x_mlem_dataset[j, ..., img_sl], cp),
                           cmap='Greys',
                           vmin=0,
                           vmax=vmax)
    im2 = axv[2, i].imshow(tonumpy(
        ndi.gaussian_filter(x_mlem_dataset[j, ..., img_sl], 1.3), cp),
                           cmap='Greys',
                           vmin=0,
                           vmax=vmax)
    im3 = axv[3, i].imshow(tonumpy(y_val_dataset[i, ..., img_sl], cp),
                           cmap='Greys',
                           vmin=0,
                           vmax=vmax)

    cb0 = figv.colorbar(im0, fraction=0.03, location='bottom')
    cb1 = figv.colorbar(im1, fraction=0.03, location='bottom')
    cb2 = figv.colorbar(im2, fraction=0.03, location='bottom')
    cb3 = figv.colorbar(im3, fraction=0.03, location='bottom')

    axv[0, i].set_title(f'ground truth image {j:03}', fontsize='medium')
    axv[1, i].set_title(f'MLEM {j:03} - {num_iter_mlem:03} it.',
                        fontsize='medium')
    axv[2, i].set_title(f'p.s. MLEM {j:03}', fontsize='medium')
    axv[3, i].set_title(f'varnet {j:03}', fontsize='medium')

for axx in axv.ravel():
    axx.axis('off')
figv.suptitle('First 5 validation data sets', fontsize=20)
figv.tight_layout()

plt.show()
