import torch

import cupy as xp
import cupy as cp
import cupyx.scipy.ndimage as ndi

from parallelproj.operators import CompositeLinearOperator, ElementwiseMultiplicationOperator, GaussianFilterOperator, LinearOperator
from parallelproj.projectors import ParallelViewProjector2D
from parallelproj.utils import tonumpy

import matplotlib.pyplot as plt


class LinearOperatorForwardLayer(torch.autograd.Function):
    """forward layer mapping using a custom linear operator

    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
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

        # convert pytorch input tensor into cupy array
        cp_x = cp.ascontiguousarray(cp.from_dlpack(x.detach()))

        # a custom function that maps from cupy array to cupy array
        cp_y = operator(cp_x)

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

            # convert torch array to cupy array
            cp_grad_output = cp.from_dlpack(grad_output.detach())

            # since forward takes three input arguments (x, projector, subset)
            # we have to return three arguments (the latter is None)
            return torch.from_dlpack(operator.adjoint(cp_grad_output)), None


class LinearOperatorAdjointLayer(torch.autograd.Function):
    """ adjoint of the LinearOperatorForwardLayer
    
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
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

        # convert pytorch input tensor into cupy array
        cp_x = cp.ascontiguousarray(cp.from_dlpack(x.detach()))

        # a custom function that maps from cupy array to cupy array
        cp_y = operator.adjoint(cp_x)

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

            # convert torch array to cupy array
            cp_grad_output = cp.from_dlpack(grad_output.detach())

            # since forward takes three input arguments (x, projector, subset)
            # we have to return three arguments (the latter is None)
            return torch.from_dlpack(operator(cp_grad_output, )), None


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

# setup a random image
img = xp.zeros((n0, n1, n2)).astype(xp.float32)
img[:, (n1 // 4):((3 * n1) // 4), (n2 // 4):((3 * n2) // 4)] = 1
img[:, (n1 // 4):(n1 // 3), (n2 // 4):(n2 // 3)] = 3

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

# setup the coordinates for projections along parallel views
num_rad = 223
num_phi = 190
scanner_R = 350.

# radial coordinates of the projection views in mm
r = xp.linspace(-200, 200, num_rad, dtype=xp.float32)
# "radius" of the scanner in mm

projector = ParallelViewProjector2D(img_shape, r, num_phi, scanner_R,
                                    img_origin, voxel_size, xp)

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# generate the attenuation image and sinogram

# the attenuation coefficients in 1/mm
att_img = 0.01 * (img > 0).astype(xp.float32)
att_sino = xp.exp(-projector(att_img))

# generate a constant sensitivity sinogram
sens_sino = xp.full(projector.out_shape, 1., dtype=xp.float32)

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

# setup the complete forward model consisting of image-based resolution model
# projector and multiplication by sensitivity

image_space_filter = GaussianFilterOperator(projector.in_shape,
                                            ndi,
                                            sigma=4.5 / (2.35 * voxel_size))

sens_operator = ElementwiseMultiplicationOperator(sens_sino * att_sino)

fwd_model = CompositeLinearOperator(
    (sens_operator, projector, image_space_filter))

fwd_model.adjointness_test(xp, verbose=True)
fwd_model_norm = fwd_model.norm(xp)

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

# apply the forward model to generate noise-free data
img_fwd = fwd_model(img)

# generate a constant contamination sinogram
contamination = xp.full(img_fwd.shape, 0.5 * img_fwd.mean(), dtype=xp.float32)

# generate noisy data
data = xp.random.poisson(img_fwd + contamination)

# do ack projection (the adjoint of the forward projection)
data_back = fwd_model.adjoint(data)

# convert the data and contamination sinograms to torch tensor
contamination_t = torch.from_dlpack(contamination)
data_t = torch.from_dlpack(data)

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

# run torch MLEM using torch layers

num_iter = 400

dtype = torch.float32
device = torch.device("cuda:0")

fwd_layer = LinearOperatorForwardLayer.apply
adjoint_layer = LinearOperatorAdjointLayer.apply

x0_t = torch.ones(fwd_model.in_shape, device=device, dtype=dtype)
x_t = torch.clone(x0_t)

sens_image_t = adjoint_layer(
    torch.ones(fwd_model.out_shape, dtype=dtype, device=device), fwd_model)

for i in range(num_iter):
    print(f'it {(i+1):04} / {num_iter:04}', end='\r')
    exp_t = fwd_layer(x_t, fwd_model) + contamination_t
    x_t *= (adjoint_layer(data_t / exp_t, fwd_model) / sens_image_t)

# convert recon to cupy array
x = cp.ascontiguousarray(cp.from_dlpack(x_t.detach()))

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

im_kwargs = dict(origin='lower', cmap='Greys')

fig2, ax2 = plt.subplots(2, 4, figsize=(4 * 4, 2 * 4))
im00 = ax2[0, 0].imshow(tonumpy(img_fwd, xp),
                        cmap='Greys',
                        vmax=1.25 * float(img_fwd.max()))
im01 = ax2[0, 1].imshow(tonumpy(data, xp),
                        cmap='Greys',
                        vmax=1.25 * float(img_fwd.max()))
im02 = ax2[0, 2].imshow(tonumpy(att_sino, xp), cmap='Greys', vmin=0, vmax=1)
ax2[0, 3].set_axis_off()

im10 = ax2[1, 0].imshow(tonumpy(img[0, ...], xp).T,
                        vmax=1.2 * img.max(),
                        **im_kwargs)
im11 = ax2[1, 1].imshow(tonumpy(data_back[0, ...], xp).T, **im_kwargs)
im12 = ax2[1, 2].imshow(tonumpy(x[0, ...], xp).T,
                        vmax=1.2 * img.max(),
                        **im_kwargs)
im13 = ax2[1, 3].imshow(tonumpy(
    ndi.gaussian_filter(x, 5.5 / (2.35 * voxel_size))[0, ...], xp).T,
                        vmax=1.2 * img.max(),
                        **im_kwargs)

ax2[0, 0].set_title('Ax', fontsize='small')
ax2[0, 1].set_title('d = Poisson(Ax + s)', fontsize='small')
ax2[0, 2].set_title('attenuation sinogram', fontsize='small')
ax2[1, 0].set_title('image - x', fontsize='small')
ax2[1, 1].set_title('back projection of data - A^H d', fontsize='small')
ax2[1, 2].set_title(f'MLEM - {num_iter} it.', fontsize='small')
ax2[1, 3].set_title(f'post-smoothed MLEM - {num_iter} it.', fontsize='small')

cb00 = fig2.colorbar(im00, fraction=0.03)
cb01 = fig2.colorbar(im01, fraction=0.03)
cb02 = fig2.colorbar(im02, fraction=0.03)
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