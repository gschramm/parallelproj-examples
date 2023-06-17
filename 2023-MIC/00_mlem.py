# # Unrolled MLEM and variational networks for Poisson projection data

# ## Part 1: MLEM using cupy GPU arrays

# +
import numpy as np

import cupy as cp
import cupyx.scipy.ndimage as ndi

from parallelproj.operators import CompositeLinearOperator, GaussianFilterOperator
from parallelproj.projectors import ParallelViewProjector2D
from parallelproj.utils import tonumpy

import matplotlib.pyplot as plt

from utils import generate_random_image
# -

# ### setup a batch of "random" ground truth images

# +
# seed the random generator, since we are using random images
seed = 0
cp.random.seed(seed)
np.random.seed(seed)
# image dimensions
n = 128
num_images = 30
img_shape = (1, n, n)

# voxel size of the image
voxel_size = cp.array([2., 2., 2.]).astype(cp.float32)

# image origin -> world coordinates of the [0,0,0] voxel
img_origin = ((-cp.array(img_shape) / 2 + 0.5) * voxel_size).astype(cp.float32)

img_dataset = cp.zeros((num_images, ) + img_shape, dtype=cp.float32)

for i in range(num_images):
    img_dataset[i, 0, ...] = generate_random_image(n, cp, ndi)
# -

# ### setup a simple "2D" parallel view non-tof projector

# +
# setup the coordinates for projections along parallel views
num_rad = 223
num_phi = 190
# "radius" of the scanner in mm
scanner_R = 350.

# radial coordinates of the projection views in mm
r = cp.linspace(-200, 200, num_rad, dtype=cp.float32)
view_angles = cp.linspace(0, cp.pi, num_phi, endpoint=False, dtype=cp.float32)

projector = ParallelViewProjector2D(img_shape, r, view_angles, scanner_R,
                                    img_origin, voxel_size, cp)
# -

# ### show the projector geometry

fig_proj = projector.show_views(image=img_dataset[0, ...], cmap='Greys')

# ### generate the attenuation images, attenuation sinograms, and sensitivity sinograms for our forward model

# +
att_img_dataset = cp.zeros((num_images, ) + img_shape, dtype=cp.float32)
att_sino_dataset = cp.zeros((num_images, num_phi, num_rad), dtype=cp.float32)

for i in range(num_images):
    # the attenuation coefficients in 1/mm
    att_img_dataset[i,
                    ...] = 0.01 * (img_dataset[i, ...] > 0).astype(cp.float32)
    att_sino_dataset[i, ...] = cp.exp(-projector(att_img_dataset[i, ...]))

# generate a constant sensitivity sinogram
sens_sino_dataset = cp.full((num_images, ) + projector.out_shape,
                            1.,
                            dtype=cp.float32)

# generate sinograms of multiplicative corrections (attention times sensitivity)
mult_corr_dataset = att_sino_dataset * sens_sino_dataset

# +
image_space_filter = GaussianFilterOperator(projector.in_shape,
                                            ndi,
                                            cp,
                                            sigma=4.5 / (2.35 * voxel_size))

# setup a projector including an image-based resolution model
projector_with_res_model = CompositeLinearOperator(
    (projector, image_space_filter))

# +
# apply the forward model to generate noise-free data
img_fwd_dataset = cp.zeros((num_images, num_phi, num_rad), dtype=cp.float32)
add_corr_dataset = cp.zeros((num_images, num_phi, num_rad), dtype=cp.float32)
adjoint_ones_dataset = cp.zeros((num_images, ) + img_shape, dtype=cp.float32)

for i in range(num_images):
    img_fwd_dataset[i,
                    ...] = mult_corr_dataset[i,
                                             ...] * projector_with_res_model(
                                                 img_dataset[i, ...])

    # generate a constant contamination sinogram
    add_corr_dataset[i, ...] = cp.full(img_fwd_dataset[i, ...].shape,
                                       0.5 * img_fwd_dataset[i, ...].mean(),
                                       dtype=cp.float32)

# generate noisy data
data_dataset = cp.random.poisson(img_fwd_dataset + add_corr_dataset)
# -

# ### generate the sensitivity images (adjoint operator applied to a sinogram of ones)

# create the sensitivity images (adjoint applied to "ones")
for i in range(num_images):
    adjoint_ones_dataset[i, ...] = projector_with_res_model.adjoint(
        mult_corr_dataset[i, ...])

# ### show the first 5 data sets

# +
vmax = float(1.1 * img_dataset.max())
fig, ax = plt.subplots(4, 5, figsize=(2.5 * 5, 2.5 * 4))
for i in range(5):
    im0 = ax[0, i].imshow(tonumpy(img_dataset[i, 0, ...], cp),
                          cmap='Greys',
                          vmin=0,
                          vmax=vmax)
    im1 = ax[1, i].imshow(tonumpy(att_img_dataset[i, 0, ...], cp),
                          cmap='Greys',
                          vmin=0,
                          vmax=float(att_img_dataset.max()))
    im2 = ax[2, i].imshow(tonumpy(mult_corr_dataset[i, ...], cp),
                          cmap='Greys',
                          vmin=0,
                          vmax=float(mult_corr_dataset.max()))
    im3 = ax[3, i].imshow(tonumpy(data_dataset[i, ...], cp),
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

# ## run MLEM

# +
num_iter = 100

x0_mlem_dataset = cp.ones(img_dataset.shape, dtype=cp.float32)
x_mlem_dataset = x0_mlem_dataset.copy()

for it in range(num_iter):
    print(f'cupy MLEM it {(it+1):04} / {num_iter:04}', end='\r')
    for ib in range(num_images):
        exp = mult_corr_dataset[ib, ...] * projector_with_res_model(
            x_mlem_dataset[ib, ...]) + add_corr_dataset[ib, ...]
        x_mlem_dataset[ib, ...] *= (projector_with_res_model.adjoint(
            mult_corr_dataset[ib, ...] * data_dataset[ib, ...] / exp) /
                                    adjoint_ones_dataset[ib, ...])
# -

# ### show all MLEM reconstructions

# +
figm, axm = plt.subplots(2, 5, figsize=(2.5 * 5, 2.5 * 2))
for i in range(5):
    im0 = axm[0, i].imshow(tonumpy(img_dataset[i, 0, ...], cp),
                           cmap='Greys',
                           vmin=0,
                           vmax=vmax)
    im1 = axm[1, i].imshow(tonumpy(x_mlem_dataset[i, 0, ...], cp),
                           cmap='Greys',
                           vmin=0,
                           vmax=vmax)

    cb0 = figm.colorbar(im0, fraction=0.03, location='bottom')
    cb1 = figm.colorbar(im1, fraction=0.03, location='bottom')

    axm[0, i].set_title(f'ground truth image {i:03}', fontsize='medium')
    axm[1, i].set_title(f'MLEM {i:03} - {num_iter:03} it.', fontsize='medium')

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
for it in range(num_iter):
    print(f'torch it {(it+1):04} / {num_iter:04}', end='\r')
    x_mlem_dataset_t = em_module.forward(x_mlem_dataset_t, data_dataset_t,
                                         mult_corr_dataset_t,
                                         add_corr_dataset_t,
                                         adjoint_ones_dataset_t)

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
    im0 = axm[0, i].imshow(tonumpy(img_dataset[i, 0, ...], cp),
                           cmap='Greys',
                           vmin=0,
                           vmax=vmax)
    im1 = axm[1, i].imshow(tonumpy(x_mlem_dataset[i, 0, ...], cp),
                           cmap='Greys',
                           vmin=0,
                           vmax=float(1.2 * img_dataset.max()))
    im2 = axm[2, i].imshow(tonumpy(x_mlem_dataset_torch[i, 0, ...], cp),
                           cmap='Greys',
                           vmin=0,
                           vmax=vmax)

    cb0 = figm.colorbar(im0, fraction=0.03, location='bottom')
    cb1 = figm.colorbar(im1, fraction=0.03, location='bottom')
    cb2 = figm.colorbar(im2, fraction=0.03, location='bottom')

    axm[0, i].set_title(f'ground truth image {i:03}', fontsize='medium')
    axm[1, i].set_title(f'cupy MLEM {i:03} - {num_iter:03} it.',
                        fontsize='medium')
    axm[2, i].set_title(f'torch MLEM {i:03} - {num_iter:03} it.',
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

# setup a simple CNN that maps an image batch onto an image batch
conv_net = Unet3D(num_features=16, num_downsampling_layers=3, batch_norm=True)

# setup the unrolled variational network consiting of block combining MLEM and conv-net updates
var_net = UnrolledVarNet(em_module, num_blocks=5, neural_net=conv_net)
var_net.eval()

num_epochs = 501
batch_size = 8
num_train = int(0.7 * num_images)
learning_rate = 1e-3
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(var_net.parameters(), lr=learning_rate)

training_loss = np.zeros(num_epochs)
validation_loss = []

# feed a mini batch through the network
for epoch in range(num_epochs):
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
    training_loss[epoch] = loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # calculate the validation loss
    if epoch % 10 == 0:
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

        print(
            f'epoch: {epoch:05} / {num_epochs:05} - train loss: {loss.item():.2E} - val loss {validation_loss[-1]:.2E}',
            end='\r')

print(f'\nconv net weights {var_net._neural_net_weight}')

print(f'min training   loss {training_loss.min():.2E}')
print(f'min validation loss {min(validation_loss):.2E}')

# +
# plot the training and validation loss
i_train = np.arange(1, num_epochs + 1)
i_val = np.arange(len(validation_loss)) * 10 + 1

figl, axl = plt.subplots(1, 1, figsize=(8, 4))
axl.plot(training_loss, '.-', label='training loss')
axl.plot(i_val, validation_loss, '.-', label='validation loss')
axl.set_ylim(0, training_loss[10:].max())
axl.legend()
axl.grid(ls=':')

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
    im0 = axm[0, i].imshow(tonumpy(img_dataset[i, 0, ...], cp),
                           cmap='Greys',
                           vmin=0,
                           vmax=vmax)
    im1 = axm[1, i].imshow(tonumpy(x_mlem_dataset[i, 0, ...], cp),
                           cmap='Greys',
                           vmin=0,
                           vmax=vmax)
    im2 = axm[2, i].imshow(tonumpy(
        ndi.gaussian_filter(x_mlem_dataset[i, 0, ...], 1.3), cp),
                           cmap='Greys',
                           vmin=0,
                           vmax=vmax)
    im3 = axm[3, i].imshow(tonumpy(y_train_dataset[i, 0, ...], cp),
                           cmap='Greys',
                           vmin=0,
                           vmax=vmax)

    cb0 = figm.colorbar(im0, fraction=0.03, location='bottom')
    cb1 = figm.colorbar(im1, fraction=0.03, location='bottom')
    cb2 = figm.colorbar(im2, fraction=0.03, location='bottom')
    cb3 = figm.colorbar(im3, fraction=0.03, location='bottom')

    axm[0, i].set_title(f'ground truth image {i:03}', fontsize='medium')
    axm[1, i].set_title(f'MLEM {i:03} - {num_iter:03} it.', fontsize='medium')
    axm[2, i].set_title(f'p.s. MLEM {i:03}', fontsize='medium')
    axm[3, i].set_title(f'varnet {i:03}', fontsize='medium')

for axx in axm.ravel():
    axx.axis('off')
figm.suptitle('First 5 training data sets', fontsize=20)
figm.tight_layout()

figv, axv = plt.subplots(4, 5, figsize=(2.5 * 5, 2.5 * 4))
for i in range(5):
    j = i + num_train
    im0 = axv[0, i].imshow(tonumpy(img_dataset[j, 0, ...], cp),
                           cmap='Greys',
                           vmin=0,
                           vmax=vmax)
    im1 = axv[1, i].imshow(tonumpy(x_mlem_dataset[j, 0, ...], cp),
                           cmap='Greys',
                           vmin=0,
                           vmax=vmax)
    im2 = axv[2, i].imshow(tonumpy(
        ndi.gaussian_filter(x_mlem_dataset[j, 0, ...], 1.3), cp),
                           cmap='Greys',
                           vmin=0,
                           vmax=vmax)
    im3 = axv[3, i].imshow(tonumpy(y_val_dataset[i, 0, ...], cp),
                           cmap='Greys',
                           vmin=0,
                           vmax=vmax)

    cb0 = figv.colorbar(im0, fraction=0.03, location='bottom')
    cb1 = figv.colorbar(im1, fraction=0.03, location='bottom')
    cb2 = figv.colorbar(im2, fraction=0.03, location='bottom')
    cb3 = figv.colorbar(im3, fraction=0.03, location='bottom')

    axv[0, i].set_title(f'ground truth image {j:03}', fontsize='medium')
    axv[1, i].set_title(f'MLEM {j:03} - {num_iter:03} it.', fontsize='medium')
    axv[2, i].set_title(f'p.s. MLEM {j:03}', fontsize='medium')
    axv[3, i].set_title(f'varnet {j:03}', fontsize='medium')

for axx in axv.ravel():
    axx.axis('off')
figv.suptitle('First 5 validation data sets', fontsize=20)
figv.tight_layout()
