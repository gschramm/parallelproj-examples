import torch
import cupy as cp
import collections

from parallelproj.operators import LinearOperator


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
    """unrolled varaitional network consisting of a PoissonEMModule and neural network block
       using weight sharing across blocks
    """

    def __init__(self,
                 poisson_em_module: torch.nn.Module,
                 num_blocks: int,
                 neural_net: torch.nn.Module | None = None,
                 init_net_weight: float = 1.0) -> None:
        """Unrolled variational network with Poisson MLEM data fidelity update

        Parameters
        ----------
        poisson_em_module : torch.nn.Module
            custom torch module that computes an MLEM update
        num_blocks : int
            number of unrolled blocks
        neural_net : torch.nn.Module | None, optional
            a torch neural network with trainable parameters
            mapping an image batch to an image batch
        init_net_weight : float, optional
            initial weight of the network based update when added to
            the MLEM update, by default 1.0
        """
        super().__init__()
        self._poisson_em_module = poisson_em_module
        self._neural_net = neural_net

        self._num_blocks = num_blocks
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

        for i in range(self._num_blocks):
            if verbose:
                print(f'iteration {(i+1):04} / {self._num_iterations:04}',
                      end='\r')
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


def simple_conv_net(num_hidden_layers=5,
                    num_features=7,
                    kernel_size=(1, 3, 3),
                    device='cuda:0',
                    dtype=torch.float32):
    """simple sequential CNN with PReLU activation"""

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
    conv_net['act_0'] = torch.nn.PReLU(device=device, dtype=dtype)

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
        conv_net[f'act_{i+1}'] = torch.nn.PReLU(device=device, dtype=dtype)

    conv_net['conv_f'] = torch.nn.Conv3d(num_features,
                                         1, (1, 1, 1),
                                         padding='same',
                                         device=device,
                                         dtype=dtype)
    conv_net['act_f'] = torch.nn.PReLU(device=device, dtype=dtype)

    conv_net = torch.nn.Sequential(conv_net)

    return conv_net