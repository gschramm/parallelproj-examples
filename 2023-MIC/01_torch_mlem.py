"""example that shows how to implement an unrolled MLEM in pytorch using a mini batch of data"""

import torch

import cupy as xp
import cupy as cp
import cupyx.scipy.ndimage as ndi

from parallelproj.operators import CompositeLinearOperator, GaussianFilterOperator, LinearOperator
from parallelproj.projectors import ParallelViewProjector2D
from parallelproj.utils import tonumpy

import matplotlib.pyplot as plt


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

# setup a test image

# image dimensions
n0, n1, n2 = (1, 128, 128)
img_shape = (n0, n1, n2)

# voxel size of the image
voxel_size = xp.array([2., 2., 2.]).astype(xp.float32)

# image origin -> world coordinates of the [0,0,0] voxel
img_origin = ((-xp.array(img_shape) / 2 + 0.5) * voxel_size).astype(xp.float32)

# setup a 2 test images
img1 = xp.zeros((n0, n1, n2)).astype(xp.float32)
img1[:, (n1 // 4):((3 * n1) // 4), (n2 // 4):((3 * n2) // 4)] = 1
img1[:, (n1 // 4):(n1 // 3), (n2 // 4):(n2 // 3)] = 3

img2 = xp.zeros((n0, n1, n2)).astype(xp.float32)
img2[:, :(n1 // 2), :(n2 // 2)] = 1
img2[:, :(n1 // 8), :(n2 // 8)] = 2

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

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

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# generate the attenuation image and sinogram

# the attenuation coefficients in 1/mm
att_img1 = 0.01 * (img1 > 0).astype(xp.float32)
att_sino1 = xp.exp(-projector(att_img1))

att_img2 = 0.01 * (img2 > 0).astype(xp.float32)
att_sino2 = xp.exp(-projector(att_img2))

# generate a constant sensitivity sinogram
sens_sino1 = xp.full(projector.out_shape, 1., dtype=xp.float32)
sens_sino2 = xp.full(projector.out_shape, 1., dtype=xp.float32)

# generate sinograms of multiplicative corrections (attention times sensitivity)
mult_corr1 = att_sino1 * sens_sino1
mult_corr2 = att_sino2 * sens_sino2

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
img_fwd1 = mult_corr1 * projector_with_res_model(img1)
img_fwd2 = mult_corr2 * projector_with_res_model(img2)

# generate a constant contamination sinogram
add_corr1 = xp.full(img_fwd1.shape, 0.5 * img_fwd1.mean(), dtype=xp.float32)
add_corr2 = xp.full(img_fwd2.shape, 0.75 * img_fwd2.mean(), dtype=xp.float32)

# generate noisy data
data1 = xp.random.poisson(img_fwd1 + add_corr1)
data2 = xp.random.poisson(img_fwd2 + add_corr2)

# create the sensitivity images (adjoint applied to "ones")
adjoint_ones1 = projector_with_res_model.adjoint(mult_corr1)
adjoint_ones2 = projector_with_res_model.adjoint(mult_corr2)

# create torch "mini-batch" tensors
data_t = torch.from_dlpack(cp.stack((data1, data2)))
mult_corr_t = torch.from_dlpack(cp.stack((mult_corr1, mult_corr2)))
add_corr_t = torch.from_dlpack(cp.stack((add_corr1, add_corr2)))
adjoint_ones_t = torch.from_dlpack(cp.stack((adjoint_ones1, adjoint_ones2)))

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

# run torch MLEM using torch layers

num_iter = 400

x0_t = torch.ones((2, ) + projector_with_res_model.in_shape,
                  device=data_t.device,
                  dtype=torch.float32)

x_t = torch.clone(x0_t)

em_module = PoissonEMModule(projector_with_res_model)

# setup our unrolled MLEM network
for i in range(num_iter):
    print(f'iteration {(i+1):04} / {num_iter:04}', end='\r')
    x_t = em_module(x_t, data_t, mult_corr_t, add_corr_t, adjoint_ones_t)
print('')

# convert recon to cupy array
x = cp.ascontiguousarray(cp.from_dlpack(x_t.detach()))
img = cp.stack((img1, img2))
img_fwd = cp.stack((img_fwd1, img_fwd2))
data = cp.ascontiguousarray(cp.from_dlpack(data_t.detach()))
adjoint_ones = cp.ascontiguousarray(cp.from_dlpack(adjoint_ones_t.detach()))
mult_corr = cp.ascontiguousarray(cp.from_dlpack(mult_corr_t.detach()))
add_corr = cp.ascontiguousarray(cp.from_dlpack(add_corr_t.detach()))

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

im_kwargs = dict(origin='lower', cmap='Greys')

for i in range(2):
    fig2, ax2 = plt.subplots(2, 4, figsize=(4 * 4, 2 * 4))
    im00 = ax2[0, 0].imshow(tonumpy(img_fwd[i, ...], xp),
                            cmap='Greys',
                            vmax=1.25 * float(img_fwd[i, ...].max()))
    im01 = ax2[0, 1].imshow(tonumpy(data[i, ...], xp),
                            cmap='Greys',
                            vmax=1.25 * float(img_fwd[i, ...].max()))
    im02 = ax2[0, 2].imshow(tonumpy(mult_corr[i, ...], xp),
                            cmap='Greys',
                            vmin=0,
                            vmax=1)
    im03 = ax2[0, 3].imshow(tonumpy(add_corr[i, ...], xp), cmap='Greys')

    im10 = ax2[1, 0].imshow(tonumpy(img[i, ...].squeeze(), xp).T,
                            vmax=1.2 * img[i, ...].max(),
                            **im_kwargs)
    im11 = ax2[1, 1].imshow(
        tonumpy(adjoint_ones[i, ...].squeeze(), xp).T, **im_kwargs)
    im12 = ax2[1, 2].imshow(tonumpy(x[i, ...].squeeze(), xp).T,
                            vmax=1.2 * img[i, ...].max(),
                            **im_kwargs)
    im13 = ax2[1, 3].imshow(tonumpy(
        ndi.gaussian_filter(x[i, ...], 6.0 / (2.35 * voxel_size)).squeeze(),
        xp).T,
                            vmax=1.2 * img[i, ...].max(),
                            **im_kwargs)

    ax2[0, 0].set_title('Ax', fontsize='small')
    ax2[0, 1].set_title('d = Poisson(Ax + s)', fontsize='small')
    ax2[0, 2].set_title('mult. correction sinogram', fontsize='small')
    ax2[0, 3].set_title('add. correction sinogram', fontsize='small')
    ax2[1, 0].set_title('image - x', fontsize='small')
    ax2[1, 1].set_title('sensitivity image - A^H 1', fontsize='small')
    ax2[1, 2].set_title(f'MLEM - {num_iter} it.', fontsize='small')
    ax2[1, 3].set_title(f'post-smoothed MLEM - {num_iter} it.',
                        fontsize='small')

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