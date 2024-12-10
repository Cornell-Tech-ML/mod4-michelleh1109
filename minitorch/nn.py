from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw

    tile_size = kh * kw

    # Reshape the tensor to separate pooling regions
    # Step 1: Reshape into (batch, channel, new_height, kh, new_width, kw)
    output = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)

    # Step 2: Permute dimensions to (batch, channel, new_height, new_width, kh, kw)
    output = output.permute(0, 1, 2, 4, 3, 5)

    # Step 3: Combine kh and kw into a single dimension (kernel_height * kernel_width)
    output = output.contiguous().view(batch, channel, new_height, new_width, tile_size)

    return output, new_height, new_width

def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform 2D average pooling.

    Args:
    ----
        input: Tensor of shape (batch, channel, height, width)
        kernel: Tuple (kernel_height, kernel_width) specifying pooling dimensions

    Returns:
    -------
        Pooled Tensor of shape (batch, channel, new_height, new_width)
    """
    # Reshape the input using tile function
    tiled, new_height, new_width = tile(input, kernel)

    # Compute the average along the last dimension (pooling region)
    pooled = tiled.mean(4).contiguous()

    return pooled.view(input.shape[0], input.shape[1], new_height, new_width)

class Max(Function):
    """Max Function for forward and backward computation."""

    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for max.

        Args:
        ----
            ctx: Context for saving intermediate results.
            input: Input tensor.
            dim: Dimension to compute max over (None for global max).

        Returns:
        -------
            Tensor with the max value(s).

        """
        # Compute max along the specified dimension
        b = input.f.max_reduce(input, int(dim.item()))
        # Save mask for backward pass
        ctx.save_for_backward(input.f.eq_zip(input, b))
        return b

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for max.

        Args:
        ----
            ctx: Context with saved intermediate values.
            grad_output: Gradient of the loss w.r.t. output.

        Returns:
        -------
            Gradient of the loss w.r.t. input.
        """
        
