"""example that shows how to implement an unrolled MLEM in pytorch using a mini batch of data"""

import torch

import numpy as np
import cupy as xp
import cupy as cp
import cupyx.scipy.ndimage as ndi

from parallelproj.operators import CompositeLinearOperator, GaussianFilterOperator, LinearOperator
from parallelproj.projectors import ParallelViewProjector2D
from parallelproj.utils import tonumpy

import matplotlib.pyplot as plt
import collections


def generate_random_image(n0, n1, n2):
    bg_img = xp.zeros((n0, n1, n2)).astype(xp.float32)
    bg_img[:, (n1 // 4):((3 * n1) // 4), (n2 // 4):((3 * n2) // 4)] = 1

    tmp = bg_img.copy()
    tmp *= xp.random.rand(n0, n1, n2).astype(xp.float32)
    tmp = (ndi.gaussian_filter(tmp, 1.5) > 0.52)

    label_img, num_labels = ndi.label(tmp)

    img = xp.zeros((n0, n1, n2)).astype(xp.float32)
    for i in range(1, num_labels + 1):
        inds = xp.where(label_img == i)
        img[inds] = 2 * xp.random.rand() + 1

    return bg_img + img


class LinearOperatorForwardLayer(torch.autograd.Function):
    """forward layer mapping using a custom linear operator

    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on mini batch tensors.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, operator: LinearOperator):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation.
        """

        ctx.set_materialize_grads(False)
        ctx.operator = operator

        num_batch = x.shape[0]

        # convert pytorch input tensor into cupy array
        cp_x = cp.ascontiguousarray(cp.from_dlpack(x.detach()))
        cp_y = cp.zeros((num_batch, ) + operator.out_shape, dtype=cp.float32)

        # apply operator across mini batch
        for i in range(cp_x.shape[0]):
            cp_y[i, ...] = operator(cp_x[i, ...])

        return torch.from_dlpack(cp_y)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        For details on how to implement the backward pass, see
        https://pytorch.org/docs/stable/notes/extending.html#how-to-use
        """

        if grad_output is None:
            return None, None
        else:
            operator = ctx.operator

            num_batch = grad_output.shape[0]

            # convert torch array to cupy array
            cp_grad_output = cp.from_dlpack(grad_output.detach())

            cp_x = cp.zeros((num_batch, ) + operator.in_shape,
                            dtype=cp.float32)

            # apply adjoint operator across mini batch
            for i in range(cp_x.shape[0]):
                cp_x[i, ...] = operator.adjoint(cp_grad_output[i, ...])

            # since forward takes two input arguments (x, operator)
            # we have to return two arguments (the latter is None)
            return torch.from_dlpack(cp_x), None


class LinearOperatorAdjointLayer(torch.autograd.Function):
    """ adjoint of the LinearOperatorForwardLayer
    
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on mini batch tensors.
    """

    @staticmethod
    def forward(ctx, x, operator: LinearOperator):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation.
        """

        ctx.set_materialize_grads(False)
        ctx.operator = operator

        num_batch = x.shape[0]

        # convert pytorch input tensor into cupy array
        cp_x = cp.ascontiguousarray(cp.from_dlpack(x.detach()))

        cp_y = cp.zeros((num_batch, ) + operator.in_shape, dtype=cp.float32)

        # apply operator across mini batch
        for i in range(cp_x.shape[0]):
            cp_y[i, ...] = operator.adjoint(cp_x[i, ...])

        return torch.from_dlpack(cp_y)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        For details on how to implement the backward pass, see
        https://pytorch.org/docs/stable/notes/extending.html#how-to-use
        """

        if grad_output is None:
            return None, None
        else:
            operator = ctx.operator

            num_batch = grad_output.shape[0]

            # convert torch array to cupy array
            cp_grad_output = cp.from_dlpack(grad_output.detach())

            cp_x = cp.zeros((num_batch, ) + operator.out_shape,
                            dtype=cp.float32)

            # apply adjoint operator across mini batch
            for i in range(cp_x.shape[0]):
                cp_x[i, ...] = operator(cp_grad_output[i, ...])

            # since forward takes two input arguments (x, operator)
            # we have to return two arguments (the latter is None)
            return torch.from_dlpack(cp_x), None


#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------


class PoissonEMModule(torch.nn.Module):

    def __init__(self, projector: LinearOperator) -> None:
        super().__init__()
        self._fwd_layer = LinearOperatorForwardLayer.apply
        self._adjoint_layer = LinearOperatorAdjointLayer.apply
        self._projector = projector

    def forward(self, x: torch.Tensor, data: torch.Tensor,
                multiplicative_correction: torch.Tensor,
                additive_correction: torch.Tensor,
                adjoint_ones: torch.Tensor) -> torch.Tensor:
        """Poisson EM step

        Parameters
        ----------
        x : torch.Tensor
            minibatch of 3D images with dimension (batch_size, n0, n1, n2)
        data : torch.Tensor
            emission data (batch_size, data_size)
        multiplicative_correction : torch.Tensor
            multiplicative corrections in forward model (batch_size, data_size)
        additive_correction : torch.Tensor
            additive corrections in forward model (batch_size, data_size)
        adjoint_ones : torch.Tensor
            adjoint applied to "ones" (sensitivity images) of size (batch_size, n0, n1, n2)

        Returns
        -------
        torch.Tensor
            minibatch of 3D images with dimension (batch_size, n0, n1, n2)
        """

        exp = multiplicative_correction * self._fwd_layer(
            x, self._projector) + additive_correction

        y = x * (self._adjoint_layer(multiplicative_correction * data / exp,
                                     self._projector) / adjoint_ones)

        return y


#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------


class UnrolledVarNet(torch.nn.Module):
    """unrolled varaitional network consisting of a PoissonEMModule and neural network blocks"""

    def __init__(self,
                 poisson_em_module: torch.nn.Module,
                 num_iterations: int,
                 neural_net: torch.nn.Module | None = None,
                 init_net_weight: float = 1.0) -> None:
        super().__init__()
        self._poisson_em_module = poisson_em_module
        self._neural_net = neural_net

        self._num_iterations = num_iterations
        self._neural_net_weight = torch.nn.Parameter(
            torch.tensor(init_net_weight))

    def forward(self,
                x: torch.Tensor,
                data: torch.Tensor,
                multiplicative_correction: torch.Tensor,
                additive_correction: torch.Tensor,
                adjoint_ones: torch.Tensor,
                verbose: bool = False) -> torch.Tensor:
        """Poisson EM step

        Parameters
        ----------
        x : torch.Tensor
            minibatch of 3D images with dimension (batch_size, 1, n0, n1, n2)
        data : torch.Tensor
            emission data (batch_size, data_size)
        multiplicative_correction : torch.Tensor
            multiplicative corrections in forward model (batch_size, data_size)
        additive_correction : torch.Tensor
            additive corrections in forward model (batch_size, data_size)
        adjoint_ones : torch.Tensor
            adjoint applied to "ones" (sensitivity images) of size (batch_size, n0, n1, n2)
        verbose : bool, optional
            print progress, by default False

        Returns
        -------
        torch.Tensor
            minibatch of 3D images with dimension (batch_size, n0, n1, n2)
        """

        y = torch.clone(x)

        for i in range(self._num_iterations):
            if verbose:
                print(f'iteration {(i+1):04} / {num_iter:04}', end='\r')
            y = self._poisson_em_module(y, data, multiplicative_correction,
                                        additive_correction, adjoint_ones)

            # pytorch convnets expect input tensors of shape (batch_size, num_channels, spatial_shape)
            # here we just add a dummy channel dimension
            if self._neural_net is not None:
                y_net = self._neural_net(y.unsqueeze(1))[:, 0, ...]
                y = torch.nn.ReLU()(y + self._neural_net_weight * y_net)

        if verbose: print('')

        return y


#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

# setup a test image
torch.manual_seed(0)

# image dimensions
n0, n1, n2 = (1, 128, 128)
num_images = 5
img_shape = (n0, n1, n2)

# voxel size of the image
voxel_size = xp.array([2., 2., 2.]).astype(xp.float32)

# image origin -> world coordinates of the [0,0,0] voxel
img_origin = ((-xp.array(img_shape) / 2 + 0.5) * voxel_size).astype(xp.float32)

# setup the coordinates for projections along parallel views
num_rad = 223
num_phi = 190
# "radius" of the scanner in mm
scanner_R = 350.

# radial coordinates of the projection views in mm
r = xp.linspace(-200, 200, num_rad, dtype=xp.float32)
view_angles = xp.linspace(0, xp.pi, num_phi, endpoint=False, dtype=xp.float32)

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

# setup a 2 test images
img_batch = xp.zeros((num_images, n0, n1, n2), dtype=xp.float32)

for i in range(num_images):
    img_batch[i, ...] = generate_random_image(n0, n1, n2)

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

projector = ParallelViewProjector2D(img_shape, r, view_angles, scanner_R,
                                    img_origin, voxel_size, xp)

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# generate the attenuation image and sinogram

att_img_batch = xp.zeros((num_images, n0, n1, n2), dtype=xp.float32)
att_sino_batch = xp.zeros((num_images, num_phi, num_rad), dtype=xp.float32)

for i in range(num_images):
    # the attenuation coefficients in 1/mm
    att_img_batch[i, ...] = 0.01 * (img_batch[i, ...] > 0).astype(xp.float32)
    att_sino_batch[i, ...] = xp.exp(-projector(att_img_batch[i, ...]))

# generate a constant sensitivity sinogram
sens_sino_batch = xp.full((num_images, ) + projector.out_shape,
                          1.,
                          dtype=xp.float32)

# generate sinograms of multiplicative corrections (attention times sensitivity)
mult_corr_batch = att_sino_batch * sens_sino_batch

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

# setup the complete forward model consisting of image-based resolution model
# projector and multiplication by sensitivity

image_space_filter = GaussianFilterOperator(projector.in_shape,
                                            ndi,
                                            xp,
                                            sigma=4.5 / (2.35 * voxel_size))

# setup a projector including an image-based resolution model
projector_with_res_model = CompositeLinearOperator(
    (projector, image_space_filter))

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

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

# create the sensitivity images (adjoint applied to "ones")
for i in range(num_images):
    adjoint_ones_batch[i, ...] = projector_with_res_model.adjoint(
        mult_corr_batch[i, ...])

# create torch "mini-batch" tensors
img_batch_t = torch.from_dlpack(img_batch)
data_batch_t = torch.from_dlpack(data_batch)
mult_corr_batch_t = torch.from_dlpack(mult_corr_batch)
add_corr_batch_t = torch.from_dlpack(add_corr_batch)
adjoint_ones_batch_t = torch.from_dlpack(adjoint_ones_batch)

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

# run torch MLEM using torch layers
x0_batch_t = torch.ones((num_images, ) + projector_with_res_model.in_shape,
                        device=data_batch_t.device,
                        dtype=torch.float32)

# setup the module for the Poisson EM update (data fidelity)
em_module = PoissonEMModule(projector_with_res_model)

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# setup the unrolled MLEM network
num_iter_mlem = 100
mlem_net = UnrolledVarNet(em_module,
                          num_iterations=num_iter_mlem,
                          neural_net=None)

# perform MLEM with unrolled network
x_mlem_batch_t = mlem_net.forward(x0_batch_t,
                                  data_batch_t,
                                  mult_corr_batch_t,
                                  add_corr_batch_t,
                                  adjoint_ones_batch_t,
                                  verbose=False)

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

# setup a minial convolutional network
device = 'cuda:0'
dtype = torch.float32
num_features = 7
num_hidden_layers = 5
kernel_size = (1, 3, 3)
conv_net = collections.OrderedDict()
conv_net['conv_0a'] = torch.nn.Conv3d(1,
                                      num_features,
                                      kernel_size,
                                      padding='same',
                                      device=device,
                                      dtype=dtype)
conv_net['conv_0b'] = torch.nn.Conv3d(num_features,
                                      num_features,
                                      kernel_size,
                                      padding='same',
                                      device=device,
                                      dtype=dtype)
conv_net['relu_0'] = torch.nn.PReLU(device=device, dtype=dtype)

for i in range(num_hidden_layers):
    conv_net[f'conv_{i+1}a'] = torch.nn.Conv3d(num_features,
                                               num_features,
                                               kernel_size,
                                               padding='same',
                                               device=device,
                                               dtype=dtype)
    conv_net[f'conv_{i+1}b'] = torch.nn.Conv3d(num_features,
                                               num_features,
                                               kernel_size,
                                               padding='same',
                                               device=device,
                                               dtype=dtype)
    conv_net[f'relu_{i+1}'] = torch.nn.PReLU(device=device, dtype=dtype)

conv_net['conv_f'] = torch.nn.Conv3d(num_features,
                                     1, (1, 1, 1),
                                     padding='same',
                                     device=device,
                                     dtype=dtype)
conv_net['relu_f'] = torch.nn.PReLU(device=device, dtype=dtype)

conv_net = torch.nn.Sequential(conv_net)

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

# train the varnet
var_net = UnrolledVarNet(em_module, num_iterations=20, neural_net=conv_net)

num_epochs = 2000
learning_rate = 1e-3
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(var_net.parameters(), lr=learning_rate)

training_loss = np.zeros(num_epochs)

# feed a minibatch through the network
for epoch in range(num_epochs):
    x_batch_t = var_net.forward(x_mlem_batch_t,
                                data_batch_t,
                                mult_corr_batch_t,
                                add_corr_batch_t,
                                adjoint_ones_batch_t,
                                verbose=False)

    loss = loss_fn(x_batch_t, img_batch_t)
    training_loss[epoch] = loss.item()

    if epoch % 10 == 0:
        print(f'{epoch:05} / {num_epochs:05}: {loss.item():.2E}', end='\r')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('')

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

# convert recon to cupy array for visualization
x_mlem_batch = cp.ascontiguousarray(cp.from_dlpack(x_mlem_batch_t.detach()))
x_batch = cp.ascontiguousarray(cp.from_dlpack(x_batch_t.detach()))

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

im_kwargs = dict(origin='lower', cmap='Greys')

for i in range(4):
    fig2, ax2 = plt.subplots(2, 4, figsize=(4 * 4, 2 * 4))
    im00 = ax2[0, 0].imshow(tonumpy(img_fwd_batch[i, ...], xp),
                            cmap='Greys',
                            vmax=1.25 * float(img_fwd_batch[i, ...].max()))
    im01 = ax2[0, 1].imshow(tonumpy(data_batch[i, ...], xp),
                            cmap='Greys',
                            vmax=1.25 * float(img_fwd_batch[i, ...].max()))
    im02 = ax2[0, 2].imshow(tonumpy(mult_corr_batch[i, ...], xp),
                            cmap='Greys',
                            vmin=0,
                            vmax=1)
    im03 = ax2[0, 3].imshow(tonumpy(add_corr_batch[i, ...], xp), cmap='Greys')

    im10 = ax2[1, 0].imshow(tonumpy(img_batch[i, ...].squeeze(), xp).T,
                            vmax=1.2 * img_batch[i, ...].max(),
                            **im_kwargs)
    im11 = ax2[1, 1].imshow(
        tonumpy(adjoint_ones_batch[i, ...].squeeze(), xp).T, **im_kwargs)
    im12 = ax2[1, 2].imshow(tonumpy(x_mlem_batch[i, ...].squeeze(), xp).T,
                            vmax=1.2 * img_batch[i, ...].max(),
                            **im_kwargs)
    im13 = ax2[1, 3].imshow(tonumpy(x_batch[i, ...].squeeze(), xp).T,
                            vmax=1.2 * img_batch[i, ...].max(),
                            **im_kwargs)

    ax2[0, 0].set_title('Ax', fontsize='small')
    ax2[0, 1].set_title('d = Poisson(Ax + s)', fontsize='small')
    ax2[0, 2].set_title('mult. correction sinogram', fontsize='small')
    ax2[0, 3].set_title('add. correction sinogram', fontsize='small')
    ax2[1, 0].set_title('image - x', fontsize='small')
    ax2[1, 1].set_title('sensitivity image - A^H 1', fontsize='small')
    ax2[1, 2].set_title(f'MLEM - {num_iter_mlem} it.', fontsize='small')
    ax2[1, 3].set_title(f'output of varnet', fontsize='small')

    cb00 = fig2.colorbar(im00, fraction=0.03)
    cb01 = fig2.colorbar(im01, fraction=0.03)
    cb02 = fig2.colorbar(im02, fraction=0.03)
    cb03 = fig2.colorbar(im03, fraction=0.03)
    cb10 = fig2.colorbar(im10, fraction=0.03)
    cb11 = fig2.colorbar(im11, fraction=0.03)
    cb12 = fig2.colorbar(im12, fraction=0.03)
    cb13 = fig2.colorbar(im13, fraction=0.03)

    for i in range(3):
        ax2[0, i].set_xlabel('radial bin')
        ax2[0, i].set_ylabel('view number')
    for i in range(4):
        ax2[1, i].set_xlabel('x1')
        ax2[1, i].set_ylabel('x2')

    fig2.tight_layout()
    fig2.show()

fig3, ax3 = plt.subplots(1, 1)
ax3.plot(training_loss)
ax3.set_xlabel('epoch')
ax3.set_ylabel('training loss')
ax3.set_ylim(None, training_loss[20:].max())
ax3.grid(ls=':')
fig3.tight_layout()
fig3.show()