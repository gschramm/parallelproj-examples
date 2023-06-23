# # Unrolled MLEM and variational networks for Poisson projection data
#
# ---
# ---
# ---
#
# **The learning objective of this tutorial is to understand how to set up and train an unrolled variational network
# to reconstruct projection data following a Poisson distribution (e.g. PET or SPECT data) using pytorch.**
#
# ![](figs/unrolled_varnet.png)
#
# This tutorial is split into three parts to understand the essential steps, tools and building block we need for the setup and training process of our model.
#
# In **Part 1**, we will learn how to perform an iterative MLEM reconstruction directly on GPU arrays which are commonly used when training neural networks. 
#
# In **Part 2**, we will implement the same MLEM reconstruction using a series of custom pytorch neural network modules.
#
# In **Part 3**, we will combine our custom MLEM modules with a trainable neural network into an unrolled variational network that can be trained with pytorch.
#
# ---
# ---
# ---

# ## Part 1: MLEM using cupy GPU arrays
#
# In this part, we will:
# - *simulate a data base* of 3D PET images based on the brainweb phantom
# - setup a projector for a "small" generic 3D non-TOF PET scanner
# - generate noisy measured projection data
# - implement MLEM to reconstruct all data sets using GPU arrays
#
# Understanding how to implement MLEM using GPU arrays is important since those arrays are commonly used when training neural networks. To handle GPU arrays, we will use the [cupy](https://docs.cupy.dev/en/stable/reference/index.html) python package which implements most of the numpy functionality for GPU CUDA arrays.
#
#

# +
#--- import of external python modules we need ---
import numpy as np
from pathlib import Path

import cupy as cp
import cupyx.scipy.ndimage as ndi

from parallelproj.operators import CompositeLinearOperator, GaussianFilterOperator
from parallelproj.utils import tonumpy

import matplotlib.pyplot as plt
import nibabel as nib

#--- import of custom python modules ---
from utils import ParallelViewProjector3D, download_data
# -

# ### 1.0 Data download 

# download the data we need for this tutorial
# the data is downloaded into the "data" folder
download_data()

# ### 1.1 Setup a batch of 3D brain images
#
# - we read 60 "small" 3D images that we downloaded (20 subjects with 3 different random contrasts) 
# - we store all images directly as cupy GPU arrays

# +
# seed the random generator, since we are using random images
seed = 0
cp.random.seed(seed)
np.random.seed(seed)

#----------------------------------------------------
#--- image input parameters -------------------------

# set the number of images to be loaded into our data base
# (we have 60 in total)
num_images = 60

#----------------------------------------------------
#----------------------------------------------------

img_dataset = []
att_img_dataset = []
subject_dirs = sorted(list(Path('data').glob('subject??')))

# load our "data base" of simulated images based in the brainweb phantom
for i in range(num_images):
    subject_index = i // 3
    image_index = i % 3
    print(
        f'loading image {(i+1):03} {subject_dirs[subject_index]} image_{image_index:03}.nii.gz',
        end='\r')
    tmp = nib.load(subject_dirs[subject_index] /
                   f'image_{image_index}.nii.gz').get_fdata()
    scale = tmp.max()

    # the images come in 1mm^3 voxels, to reduce comp. time we "downsample" by a factor
    # of 2 in all directions and select only a few slices
    img_dataset.append(
        cp.swapaxes(
            cp.pad(cp.asarray(tmp[::2, ::2, 75:107:2] / scale),
                   ((1, 1), (1, 1), (0, 0))), 0, 1))
    tmp = nib.load(subject_dirs[subject_index] /
                   'attenuation_image.nii.gz').get_fdata()
    att_img_dataset.append(
        cp.swapaxes(
            cp.pad(cp.asarray(tmp[::2, ::2, 75:107:2]),
                   ((1, 1), (1, 1), (0, 0))), 0, 1))
print('')

img_dataset = cp.array(img_dataset, dtype=cp.float32)
att_img_dataset = cp.array(att_img_dataset, dtype=cp.float32)

# set the voxel size of the images (in mm)
voxel_size = cp.array([2., 2., 2.]).astype(cp.float32)

# setup a tuple containing the image shape (dimensions)
img_shape = img_dataset.shape[1:]

# setup the "image origin" -> world coordinates of the [0,0,0] voxel
img_origin = ((-cp.array(img_shape) / 2 + 0.5) * voxel_size).astype(cp.float32)

print(
    f'image data base shape (# images, # n0, # n1, # n2) :{img_dataset.shape}')
# -

# ### 1.2 Setup of a simple "3D" parallel view non-tof projector
#
# - set up a projector for a generic 3D non-TOF parallel view projector
# - to accelerate projection times, we set up a projector with a small axial FOV and small ring difference

# +
# setup the coordinates for projections along parallel views

num_rad = 111  # number of radial elements in the sinogram
num_phi = 190  # number of views in the sinogram
scanner_R = 350.  # "radius" of the scanner in mm

rmax = 1.4 * float(voxel_size[0] * img_dataset.shape[1] / 2)
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

print(
    f'projector input (image) shape (# n0, # n1, # n2) :{projector.in_shape}')
print(
    f'projector output (sinogram) shape (# views, # radial elements, # projection planes) :{projector.out_shape}'
)
# -

# ### 1.3 Setup of multiplicative correction (attenuation and sensitivity) sinograms
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
att_sino_dataset = cp.zeros((num_images, ) + projector.out_shape,
                            dtype=cp.float32)

for i in range(num_images):
    # the attenuation coefficients in 1/mm
    # assume that our objects contain water attenuation
    att_sino_dataset[i, ...] = cp.exp(-projector(att_img_dataset[i, ...]))

# generate a constant sensitivity sinogram
# this value can be used to control the number of simulated counts (the noise level)
sens_value = 5.0
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

# ### 1.4 Setup an image-based resolution model
#
# - we use a simplistic image-based Gaussian convolution operator $G$ to simulate the limited resolution of our scanner

# +
simulated_resolution_mm = 4.5

image_space_filter = GaussianFilterOperator(projector.in_shape,
                                            ndi,
                                            cp,
                                            sigma=simulated_resolution_mm /
                                            (2.35 * voxel_size))

# setup a projector including an image-based resolution model
# in the notation of our forward model this is (PG)
projector_with_res_model = CompositeLinearOperator(
    (projector, image_space_filter))
# -

# ### 1.5 Apply the forward model to our simulated ground truth images
#
# - we evaluate $M\,PG\,x$ for all simulated images

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

# ### 1.6 Generate additive (scatter + randoms) correction sinograms
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

# ### 1.7 Generate noisy projection data following a Poisson distribution
#
# - we generate noisy data $d = \text{Poisson}(A\,x + s) = \text{Poisson}(M\,PG\,x + s)$

# generate noisy data
data_dataset = cp.random.poisson(img_fwd_dataset + add_corr_dataset)

# ### 1.8 Generate the sensitivity images (adjoint operator applied to a sinogram of ones)
#
# For MLEM, we calculate the "sensitivy images"
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

# ### 1.9 Show the first 5 data sets
#
# - show the first 5 simulated activity images, attenuation images, multiplicative correction sinograms and noisy emission sinograms
# - we show the "central slice" of the image and the "central slice" of the direct sinogram planes

# +
vmax = float(1.1 * img_dataset.max())
fig, ax = plt.subplots(4, 5, figsize=(2.5 * 5, 2.5 * 4))

img_sl = img_dataset.shape[-1] // 2
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

# ### 1.10 Reconstruct all simulated data sets using early-stopped MLEM
#
# we use 200 MLEM updates given by
#
# $$ x^+ = \frac{x}{A^H \mathbb{1}} A^H \frac{d}{A\,x + s}$$
#
# $$ x^+ = \frac{x}{(PG)^H M^H \mathbb{1}} (PG)^H M^H \frac{d}{M\, PG \,x + s}$$
#
# to reconstruct all our simulated data sets.
#
# **Note that:**
# - our MLEM implementation runs directly on our GPU meaning that there is no memory transfer between host and GPU
# - we reconstruct all 60 data sets at once
#

# +
num_iter_mlem = 200

x0_mlem_dataset = cp.ones(img_dataset.shape, dtype=cp.float32)
x_mlem_dataset = x0_mlem_dataset.copy()

# MLEM iteration loop
for it in range(num_iter_mlem):
    print(f'cupy  MLEM it {(it+1):04} / {num_iter_mlem:04}', end='\r')
    # loop over all data sets to be reconstructed
    for ib in range(num_images):
        exp = mult_corr_dataset[ib, ...] * projector_with_res_model(
            x_mlem_dataset[ib, ...]) + add_corr_dataset[ib, ...]
        x_mlem_dataset[ib, ...] *= (projector_with_res_model.adjoint(
            mult_corr_dataset[ib, ...] * data_dataset[ib, ...] / exp) /
                                    adjoint_ones_dataset[ib, ...])
print('')
# -

# ### 1.11 Display the first 5 MLEM reconstructions
#
# - we show the MLEM reconstructions of the first 5 data sets and compare them to the ground truth images

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

# ### 1.12 Summary of Part 1
#
# So far we have:
# - setup a "simple" forward model $Ax + s$
# - generated a data set of PET brain images, and corresponding emission and correction sinograms
# - used MLEM to reconstruct all data sets directly on our GPU

# ---
# ---
# ---
# ---
# ---
# ---
# ---
# ---
# ---

# ## Part 2: MLEM using an unrolled pytorch network
#
# ---
# ---
# ---
#
# ![](figs/unrolled_mlem.png)
#
# In this part, we will implement the series of MLEM updates as an unrolled pytorch network.
# In other words, we will setup and use a custom pytorch module that calculates a single MLEM update on a batch of data sets. Repetitive use (stacking) of those modules allows us to re-produce the MLEM reconstructions obtained of part 1.
#
# This is an important pre-cursor for setting up a trainable unrolled variational network including trainable parameters which we will do in Part 3.

# +
# import pytorch
import torch
# import a custom torch module that computes an MLEM update
from torch_utils import PoissonEMModule

# setup the custom torch PoissonEMModule that takes the operator (PG) as input
em_module = PoissonEMModule(projector_with_res_model)
# -

# ### 2.1 Convert cupy GPU arrays to pytorch GPU tensors
#
# To use pytorch, we have to convert our cupy GPU arrays to pytorch GPU tensors. Fortunately, this is possible with zero copy data exchage - see [here](https://docs.cupy.dev/en/stable/user_guide/interoperability.html#pytorch) for details.
#
# **Note:**
#
# - in this notebook, we use the suffix *_t* for pytorch GPU tensors
# - pytorch neural network modules operator on batches of data with shape *(batch size, number of channels, spatial size)*
# - since we deal with single channel images, we simply add a dummy channel axis of length 1

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
# set up a variable that will contain all our torch MLEM reconstructions
x_mlem_dataset_t = torch.clone(x0_mlem_dataset_t)

# -

# ### 2.2 Pytorch module-based MLEM
#
# We execute MLEM reconstructions of all data sets, by passing our data repetitively through our custom PoissonEMModules.
# Note that the module acts on a batch of data sets.

# run MLEM using a custom defined EM_Module
for it in range(num_iter_mlem):
    print(f'torch MLEM it {(it+1):04} / {num_iter_mlem:04}', end='\r')
    x_mlem_dataset_t = em_module.forward(x_mlem_dataset_t, data_dataset_t,
                                         mult_corr_dataset_t,
                                         add_corr_dataset_t,
                                         adjoint_ones_dataset_t)
print('')

# ### 2.3 Compare cupy and pytorch MLEM results
#
# If our implementation of the pytorch module based MLEM reconstruction is correct, the difference to the results obtained in Part 1 should be "small" (< 1e-5). Note that the difference is not exactly zero due to parallel addition of floating point numbers with limited precision.

# +
# convert the torch MLEM reconstruction array back to cupy for comparison and visualization
x_mlem_dataset_torch = cp.ascontiguousarray(
    cp.from_dlpack(x_mlem_dataset_t.detach())).squeeze(1)

# calculate the max difference between the cupy and torch MLEM implementation
print(
    f'max cupy MLEM - torch MLEM difference: {cp.abs(x_mlem_dataset - x_mlem_dataset_torch).max()}'
)
# -

# ### 2.4 Display the pytorch and cupy MLEM reconstructions

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

# ### 2.5 Summary of Part 2
#
# In part 2 we have:
# - set up / used a custom pytorch module that computes a single MLEM update on a data batch
# - used a forward pass through a series / stack of those module to implement several MLEM iterations
#
# Now, we are ready to setup an unrolled variational network consisting of PoissonEMModules and neural network modules with trainable parameters (see part 3).

# ---
# ---
# ---
# ---
# ---
# ---
#

# ## Part 3: Supervised training of an unrolled variational network
#
# ---
# ---
# ---
#
# In the previous part, we have seen how to set ut / use a custom Poisson EM modules to compute an MLEM update. In this part, we will the Poisson EM block and a trainable neural network (e.g. a Unet) into blocks forming an unrolled variational network.
#
# ![](figs/unrolled_varnet.png)

# ### 3.1 Set up of simple convolutional neural network (3D Unet)
#
# - we set up a demo neural network (a 3D Unet) that maps a batch image tensor of shape (num_batch,1,n0,n1,n2) onto a tensor of the same shape 

# +
from torch_utils import Unet3D

# number of features in the highest level of the Unet
num_features = 16
# number of downsampling layers in the Unet
num_downsampling_layers = 3

# setup a simple CNN that maps a tensor with shape [batch_size, 1, spatial shape] onto a tensor with the same shape
conv_net = Unet3D(num_features=num_features,
                  num_downsampling_layers=num_downsampling_layers)

print(conv_net)
# -

# ### 3.2 Setup of an unrolled variational network
#
# - we combine the custom Poisson EM update module and our trainable neural network into an unrolled variational network

# +
from torch_utils import UnrolledVarNet

# number of unrolled block in our variational network
num_blocks = 6

# setup the unrolled variational network consiting of block combining MLEM and conv-net updates
var_net = UnrolledVarNet(em_module, num_blocks=num_blocks, neural_net=conv_net)
# -

# ### 3.3 Setup of training parameters and validation metrics 

# +
import torchmetrics

#---------------------------
#--- training parameters ---
#---------------------------

# number of parameter updates
num_updates = 1201
# mini batch size
batch_size = 5
# number of images to use for training
num_train = int(0.8 * num_images)
# learning rate of the optimizer
learning_rate = 1e-3
optimizer = torch.optim.Adam(var_net.parameters(), lr=learning_rate)
# loss function for supervised training
loss_fn = torch.nn.MSELoss()
#loss_fn = torch.nn.L1Loss()

training_loss = np.zeros(num_updates)

#---------------------------
#--- validation metrics ----
#---------------------------

# metrics to track on validation images
data_range = float(img_dataset_t.max())
ssim_fn = torchmetrics.StructuralSimilarityIndexMeasure(
    data_range=data_range).to(x_mlem_dataset_t.device)
psnr_fn = torchmetrics.PeakSignalNoiseRatio(data_range=data_range).to(
    x_mlem_dataset_t.device)

validation_loss = []
validation_ssim = []
validation_psnr = []
# -

# ### 3.4 Training Loop
#
# - we train our model by looping over mini-batches of data and by using pytorch's autograd functionality

# +
# feed a mini batch through the network
for update in range(num_updates):
    var_net.train()
    
    # randomly select indices forming our training mini batch
    i_batch = np.random.choice(np.arange(num_train),
                               size=batch_size,
                               replace=False)
    
    # feed forward pass of the training mini batch though our model
    y_train_t = var_net.forward(x_mlem_dataset_t[i_batch, ...],
                                data_dataset_t[i_batch, ...],
                                mult_corr_dataset_t[i_batch, ...],
                                add_corr_dataset_t[i_batch, ...],
                                adjoint_ones_dataset_t[i_batch, ...],
                                verbose=False)

    # compute the loss between our predicted data and the reference data
    # !! in this tutorial, we use the ground truth images as reference
    #    which are not available in real world applications,
    #    but can be replaced with any kind of higher-quality images !!
    loss = loss_fn(y_train_t, img_dataset_t[i_batch, ...])
    training_loss[update] = loss.item()

    # calculate the gradient of the loss function and back-propagate it
    # through the entire network to update the network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    
    #-------------------------------------------
    #--- validation step -----------------------
    #-------------------------------------------
    
    if update % 10 == 0:
        var_net.eval()
        with torch.no_grad():
            # feed forward pass of validation data
            y_val_t = var_net.forward(x_mlem_dataset_t[num_train:, ...],
                                      data_dataset_t[num_train:, ...],
                                      mult_corr_dataset_t[num_train:, ...],
                                      add_corr_dataset_t[num_train:, ...],
                                      adjoint_ones_dataset_t[num_train:, ...],
                                      verbose=False)
            
        # calculate the validation loss and all validation metrics
        validation_loss.append(
            loss_fn(y_val_t, img_dataset_t[num_train:, ...]).item())
        validation_ssim.append(
            ssim_fn(y_val_t, img_dataset_t[num_train:, ...]).item())
        validation_psnr.append(
            psnr_fn(y_val_t, img_dataset_t[num_train:, ...]).item())

        # save the model state dict (weights) if the valdiation loss improves
        if validation_loss[-1] == min(validation_loss):
            torch.save(var_net.state_dict(), 'best_var_net_state.ckpt')

        print(
            f'update: {update:05} / {num_updates:05} - train loss: {loss.item():.2E} - val loss {validation_loss[-1]:.2E} - val ssim {validation_ssim[-1]:.2E} - val psnr {validation_psnr[-1]:.2E}',
            end='\r')


        
# save the last state of our variational model
torch.save(var_net.state_dict(), 'last_var_net_state.ckpt')
# -

# ### 3.5 Print the best validation loss and metrics

# +
print(f'\nconv net fusion weights {var_net._neural_net_weight}\n')

# print the optimal values of the losses and metrics
print(f'min training   loss {training_loss.min():.2E}')
print(f'min validation loss {min(validation_loss):.2E}')
print(f'max validation ssim {max(validation_ssim):.2E}')
print(f'max validation psnr {max(validation_psnr):.2E}')
# -

# ### 3.6 Plot validation loss and metrics

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
# -

# ### 3.7 Inference - predictions using the model with best validation loss

# +
# load the model weights that gave the best validation loss
var_net.load_state_dict(torch.load('best_var_net_state.ckpt'))
# don't forget to use the model in "eval" mode which changes the
# behavior of certain layers (e.g. batchnorm or dropout)
var_net.eval()

with torch.no_grad():
    i_train = np.arange(num_train)
    # prediction of training data
    y_train_t = var_net.forward(x_mlem_dataset_t[i_train, ...],
                                data_dataset_t[i_train, ...],
                                mult_corr_dataset_t[i_train, ...],
                                add_corr_dataset_t[i_train, ...],
                                adjoint_ones_dataset_t[i_train, ...],
                                verbose=False)

with torch.no_grad():
    i_val = np.arange(num_train, num_images)
    # prediction of validation data
    y_val_t = var_net.forward(x_mlem_dataset_t[i_val, ...],
                              data_dataset_t[i_val, ...],
                              mult_corr_dataset_t[i_val, ...],
                              add_corr_dataset_t[i_val, ...],
                              adjoint_ones_dataset_t[i_val, ...],
                              verbose=False)

# calculate the validation PSNRs and SSIMs for all images
val_psnr = [
    psnr_fn(y_val_t[i, ...], img_dataset_t[num_train + i, ...]).item()
    for i in range(y_val_t.shape[0])
]

val_ssim = [
    ssim_fn(y_val_t[i, ...], img_dataset_t[num_train + i, ...]).item()
    for i in range(y_val_t.shape[0])
]

# convert the predictions from torch GPU tensors into cupy arrays (for display)
y_train_dataset = cp.ascontiguousarray(cp.from_dlpack(
    y_train_t.detach())).squeeze(1)
y_val_dataset = cp.ascontiguousarray(cp.from_dlpack(
    y_val_t.detach())).squeeze(1)
# -

# ### 3.8 Visualize the results of the predictions of training and validation data

# +
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

    axv[0, i].set_title(f'ground truth image {j:03}', fontsize='small')
    axv[1, i].set_title(f'MLEM {j:03} - {num_iter_mlem:03} it.',
                        fontsize='small')
    axv[2, i].set_title(f'p.s. MLEM {j:03}', fontsize='small')
    axv[3, i].set_title(
        f'varnet {j:03} PSNR {val_psnr[i]:.1f} SSIM {val_ssim[i]:.2f}',
        fontsize='small')

for axx in axv.ravel():
    axx.axis('off')
figv.suptitle('First 5 validation data sets', fontsize=20)
figv.tight_layout()

plt.show()
