from typing import Tuple, Optional

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

max_reduce = FastOps.reduce(operators.max, -1e9)

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

def argmax(input: Tensor, axis: int) -> Tensor:
    """Compute the argmax as a one-hot tensor along a specified axis.

    Parameters
    ----------
    input : Tensor
        The input tensor for which the argmax will be computed.
    axis : int
        The axis along which to compute the argmax.

    Returns
    -------
        Tensor
            A one-hot tensor with the same shape as the input, where the maximum
            indices along the specified axis are marked as 1, and all others are 0.
    
    """
    max_values = max_reduce(input, axis)
    return input == max_values

class Max(Function):
    """Function to compute the max operation in both forward and backward passes"""

    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward pass of the max function.

        Parameters
        ----------
        ctx : Context
            A context object to store values for the backward pass.
        input : Tensor
            The input tensor.
        dim : Tensor
            The dimension along which to compute the max.

        Returns
        -------
            Tensor
                A tensor containing the maximum values along the specified dimension.
       
        """
        # Save input tensor and dimension for backward pass
        ctx.save_for_backward(input, dim)
        # Compute max values along the specified dimension
        return max_reduce(input, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass of the max function.

        Parameters
        ----------
        ctx : Context
            A context object containing saved values from the forward pass.
        grad_output : Tensor
            The gradient of the output with respect to the final objective.

        Returns
        -------
            Tuple[Tensor, float]
                Gradient of the input tensor and the dimension parameter.
        
        """
        input, dim = ctx.saved_values
        # Use argmax to generate a one-hot tensor for backpropagation
        return grad_output * argmax(input, int(dim.item())), 0.0

        
def max(input: Tensor, dim: int) -> Tensor:
    """Compute the maximum value along a specific axis.

    Parameters
    ----------
    input : Tensor
        The input tensor for which the maximum is to be computed.
    dim : Optional[int]
        The axis along which the maximum is computed. If None, the maximum of
        the flattened tensor is computed.

    Returns
    -------
        Tensor
            A tensor containing the maximum values along the specified axis.
    
    """
    return Max.apply(input, input._ensure_tensor(dim))

def softmax(input: Tensor, dim: int) -> Tensor:
    """Apply the softmax function over a specific axis of the tensor.

    Parameters
    ----------
    input : Tensor
        The input tensor to which softmax will be applied.
    dim : int
        The axis along which to compute softmax.

    Returns
    -------
        Tensor
            A tensor with values normalized along the specified axis.
    
    """
    # Exponentiate the values in the tensor and normalize by their sum
    exp = input.exp()
    return exp / exp.sum(dim)

def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the logarithm of the softmax function for numerical stability.

    Parameters
    ----------
    input : Tensor
        The input tensor for which the log softmax will be computed.
    dim : int
        The axis along which to compute the log softmax.

    Returns
    -------
        Tensor
            A tensor with the log softmax values along the specified axis.
   
    """
    max_values = max(input, dim)
    normalized_input = input - max_values
    exp_sum = normalized_input.exp().sum(dim)
    log_exp_sum = exp_sum.log() + max_values # what is this step
    return input - log_exp_sum

def maxpool2d(input: Tensor, window: Tuple[int, int]) -> Tensor:
    """Perform 2D max pooling over the input tensor.

    Parameters
    ----------
    input : Tensor
        A 4D tensor with dimensions (batch, channel, height, width).
    window : Tuple[int, int]
        The size of the pooling window as (window_height, window_width).

    Returns
    -------
        Tensor
            A tensor with reduced spatial dimensions after max pooling.
    
    """
    pooled_tensor, out_height, out_width = tile(input, window)
    max_pooled = max(pooled_tensor, 4)
    return max_pooled.contiguous().view(
        input.shape[0], input.shape[1], out_height, out_width
    )

def dropout(input: Tensor, prob: float, ignore: bool = False) -> Tensor:
    """Apply dropout to randomly zero out elements in the input tensor.

    Parameters
    ----------
    input : Tensor
        The input tensor to which dropout will be applied.
    prob : float
        The probability of dropping out elements (0 to 1).
    ignore : bool
        Whether to apply dropout (set to False during inference).

    Returns
    -------
        Tensor
            A tensor with random elements zeroed out, scaled to retain expected
            values.
    
    """
    if ignore:
        return input
    mask = rand(input.shape) > prob
    return input * mask

