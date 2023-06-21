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
        num_ch = x.shape[1]

        # convert pytorch input tensor into cupy array
        cp_x = cp.ascontiguousarray(cp.from_dlpack(x.detach()))
        cp_y = cp.zeros((num_batch, num_ch) + operator.out_shape,
                        dtype=cp.float32)

        # apply operator across mini batch and channels
        for ib in range(num_batch):
            for ich in range(num_ch):
                cp_y[ib, ich, ...] = operator(cp_x[ib, ich, ...])

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
            num_ch = grad_output.shape[1]

            # convert torch array to cupy array
            cp_grad_output = cp.from_dlpack(grad_output.detach())

            cp_x = cp.zeros((num_batch, num_ch) + operator.in_shape,
                            dtype=cp.float32)

            # apply adjoint operator across mini batch
            for ib in range(num_batch):
                for ich in range(num_ch):
                    cp_x[ib, ich,
                         ...] = operator.adjoint(cp_grad_output[ib, ich, ...])

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
        num_ch = x.shape[1]

        # convert pytorch input tensor into cupy array
        cp_x = cp.ascontiguousarray(cp.from_dlpack(x.detach()))
        cp_y = cp.zeros((num_batch, num_ch) + operator.in_shape,
                        dtype=cp.float32)

        # apply operator across mini batch
        for ib in range(num_batch):
            for ich in range(num_ch):
                cp_y[ib, ich, ...] = operator.adjoint(cp_x[ib, ich, ...])

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
            num_ch = grad_output.shape[1]

            # convert torch array to cupy array
            cp_grad_output = cp.from_dlpack(grad_output.detach())

            cp_x = cp.zeros((num_batch, num_ch) + operator.out_shape,
                            dtype=cp.float32)

            # apply adjoint operator across mini batch
            for ib in range(num_batch):
                for ich in range(num_ch):
                    cp_x[ib, ich, ...] = operator(cp_grad_output[ib, ich, ...])

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
            minibatch of 3D images with dimension (batch_size, 1, n0, n1, n2)
        data : torch.Tensor
            emission data (batch_size, 1, data_size)
        multiplicative_correction : torch.Tensor
            multiplicative corrections in forward model (batch_size, 1, data_size)
        additive_correction : torch.Tensor
            additive corrections in forward model (batch_size, 1, data_size)
        adjoint_ones : torch.Tensor
            adjoint applied to "ones" (sensitivity images) of size (batch_size, 1, n0, n1, n2)

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
                 neural_net: torch.nn.Module | None = None):
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
        """
        super().__init__()
        self._poisson_em_module = poisson_em_module
        self._neural_net = neural_net

        self._num_blocks = num_blocks
        self._neural_net_weight = torch.nn.Parameter(
            0.1 * torch.ones(self._num_blocks))

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
                y_net = self._neural_net(y)
                y = torch.nn.ReLU()(y + self._neural_net_weight[i] * y_net)

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


#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------


class Unet3D(torch.nn.Module):

    def __init__(self,
                 device='cuda:0',
                 num_features: int = 8,
                 num_downsampling_layers: int = 3,
                 kernel_size: tuple[int, int, int] = (3, 3, 3),
                 batch_norm: bool = False,
                 dropout_rate: float = 0.,
                 dtype=torch.float32) -> None:

        super().__init__()

        self._device = device
        self._num_features = num_features
        self._kernel_size = kernel_size
        self._dtype = dtype
        self._num_downsampling_layers = num_downsampling_layers
        self._batch_norm = batch_norm

        self._encoder_blocks = torch.nn.ModuleList()
        self._upsample_blocks = torch.nn.ModuleList()
        self._decoder_blocks = torch.nn.ModuleList()

        self._dropout = torch.nn.Dropout(dropout_rate)

        # first encoder block that takes input
        self._encoder_blocks.append(
            self._conv_block(1, num_features, num_features))

        for i in range(self._num_downsampling_layers):
            self._encoder_blocks.append(
                self._conv_block((2**i) * num_features,
                                 (2**(i + 1)) * num_features,
                                 (2**(i + 1)) * num_features))

        for i in range(self._num_downsampling_layers):
            n = self._num_downsampling_layers - i
            self._upsample_blocks.append(
                torch.nn.ConvTranspose3d((2**n) * num_features,
                                         (2**(n - 1)) * num_features,
                                         kernel_size=(2, 2, 2),
                                         stride=2,
                                         device=device))

            self._decoder_blocks.append(
                self._conv_block((2**n) * num_features,
                                 (2**(n - 1)) * num_features,
                                 (2**(n - 1)) * num_features))

        self._final_conv = torch.nn.Conv3d(num_features,
                                           1, (1, 1, 1),
                                           padding='same',
                                           device=self._device,
                                           dtype=self._dtype)

    def _conv_block(self, num_features_in, num_features_mid, num_features_out):
        conv_block = collections.OrderedDict()

        conv_block['conv_1'] = torch.nn.Conv3d(num_features_in,
                                               num_features_mid,
                                               self._kernel_size,
                                               padding='same',
                                               device=self._device,
                                               dtype=self._dtype)

        conv_block['activation_1'] = torch.nn.LeakyReLU()

        conv_block['conv_2'] = torch.nn.Conv3d(num_features_mid,
                                               num_features_out,
                                               self._kernel_size,
                                               padding='same',
                                               device=self._device,
                                               dtype=self._dtype)

        if self._batch_norm:
            conv_block['batch_norm'] = torch.nn.BatchNorm3d(
                num_features_out, device=self._device)

        conv_block['activation_2'] = torch.nn.LeakyReLU()

        conv_block = torch.nn.Sequential(conv_block)

        return conv_block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_down = []
        x_up = []

        x_down.append(self._encoder_blocks[0](x))

        for i in range(self._num_downsampling_layers):
            x_down.append(self._encoder_blocks[i + 1](torch.nn.MaxPool3d(
                (2, 2, 2))(x_down[i])))

        x_up = self._dropout(x_down[-1])

        for i in range(self._num_downsampling_layers):
            x_up = self._decoder_blocks[i](torch.cat(
                [x_down[-(i + 2)], self._upsample_blocks[i](x_up)], dim=1))

        xout = self._final_conv(x_up)

        return xout