import torch
import cupy as cp

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

