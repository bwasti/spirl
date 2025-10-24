"""
Batch-invariant operations with backward pass support.

This module adds gradient support to vLLM's deterministic batch_invariant mode
by registering backward operations that also use vLLM's deterministic kernels.

Key architecture:
- Forward: Uses vLLM's batch_invariant Triton kernels (deterministic)
- Backward: Also uses vLLM's batch_invariant kernels (deterministic)

This achieves bitwise-deterministic RL training where both rollouts (forward)
and training (forward + backward) produce identical results.

Usage:
    from vllm.model_executor.layers.batch_invariant import init_batch_invariance
    from batch_invariant_backward import patch_batch_invariant_with_gradients

    # Initialize vLLM's deterministic mode first
    init_batch_invariance()

    # Then patch in gradient support
    patch_batch_invariant_with_gradients()

    # Now all operations are deterministic AND support gradients
    model = MyModel()
    output = model(input)  # deterministic forward
    loss = compute_loss(output)
    loss.backward()  # gradients work with deterministic backward!
"""

import torch


# ============================================================================
# Backward operation implementations for autograd
# ============================================================================

def matmul_backward_impl(grad_output, self, other, output_mask):
    """
    Backward pass for matmul: y = matmul(a, b)
    Returns: (grad_a, grad_b)

    Args:
        grad_output: Gradient from downstream
        self: First input tensor (a)
        other: Second input tensor (b)
        output_mask: List of bools indicating which gradients to compute [self, other]

    grad_a = grad_output @ b.T
    grad_b = a.T @ grad_output

    Uses torch.matmul which is overridden by vLLM's batch_invariant mode!
    """
    grad_self = grad_other = None

    # output_mask is a list [compute_grad_self, compute_grad_other]
    compute_grad_self = output_mask[0] if len(output_mask) > 0 else True
    compute_grad_other = output_mask[1] if len(output_mask) > 1 else True

    if compute_grad_self:
        # grad_self = grad_output @ other.T
        if other.ndim == 2:
            grad_self = torch.matmul(grad_output, other.t())
        elif other.ndim == 3:
            grad_self = torch.matmul(grad_output, other.transpose(-2, -1))
        else:
            grad_self = torch.matmul(grad_output, other.transpose(-2, -1))

    if compute_grad_other:
        # grad_other = self.T @ grad_output
        if self.ndim == 2:
            grad_other = torch.matmul(self.t(), grad_output)
        elif self.ndim == 3:
            grad_other = torch.matmul(self.transpose(-2, -1), grad_output)
        else:
            grad_other = torch.matmul(self.transpose(-2, -1), grad_output)

    return grad_self, grad_other


def linear_backward_impl(grad_output, input, weight, output_mask):
    """
    Backward pass for linear: y = input @ weight.T + bias
    Returns: (grad_input, grad_weight, grad_bias)

    Args:
        grad_output: Gradient from downstream (actually the saved input!)
        input: Input tensor (actually grad_output!)
        weight: Weight tensor
        output_mask: List of bools indicating which gradients to compute [input, weight, bias]

    PyTorch passes args in weird order: (saved_input, grad_output, weight, output_mask)
    So we swap the first two args in our implementation.
    """
    # Swap: PyTorch passes (saved_input, grad_output, ...) but we want (grad_output, input, ...)
    input, grad_output = grad_output, input

    grad_input = grad_weight = grad_bias = None

    # output_mask is a list [compute_grad_input, compute_grad_weight, compute_grad_bias]
    compute_grad_input = output_mask[0] if len(output_mask) > 0 else True
    compute_grad_weight = output_mask[1] if len(output_mask) > 1 else True
    compute_grad_bias = output_mask[2] if len(output_mask) > 2 else True

    if compute_grad_input:
        # grad_input = grad_output @ weight
        grad_input = torch.matmul(grad_output, weight)

    if compute_grad_weight:
        # PyTorch linear: y = x @ W.T + b where W is [out, in]
        # Backward: grad_W = grad_y.T @ x
        # grad_output: (batch, out), input: (batch, in)
        # grad_output.T @ input: (out, batch) @ (batch, in) = (out, in) ✓

        # Handle multi-dimensional inputs
        if input.ndim == 3:
            # Reshape for matmul: (batch, seq, in) -> (batch*seq, in)
            input_2d = input.reshape(-1, input.shape[-1])
            grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
            # grad_output_2d: (batch*seq, out), input_2d: (batch*seq, in)
            # grad_output_2d.T @ input_2d: (out, batch*seq) @ (batch*seq, in) = (out, in) ✓
            grad_weight = torch.matmul(grad_output_2d.transpose(0, 1), input_2d)
        else:
            # input: (batch, in), grad_output: (batch, out)
            # grad_output.T @ input: (out, batch) @ (batch, in) = (out, in) ✓
            grad_weight = torch.matmul(grad_output.transpose(0, 1), input)

    if compute_grad_bias:
        # grad_bias = sum(grad_output) along all dims except last
        grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))

    return grad_input, grad_weight, grad_bias


# ============================================================================
# Registration
# ============================================================================

_batch_invariant_backward_MODE = False
_batch_invariant_backward_LIB = None


def patch_batch_invariant_with_gradients():
    """Patch vLLM's batch_invariant mode to support gradients.

    This function adds backward pass support to vLLM's existing batch_invariant
    implementations by registering the backward operations. vLLM handles all the
    forward passes, we just add gradient support.
    """
    global _batch_invariant_backward_MODE, _batch_invariant_backward_LIB

    if _batch_invariant_backward_MODE:
        return

    # Get vLLM's batch_invariant library (already created by init_batch_invariance)
    from vllm.model_executor.layers import batch_invariant as vllm_bi

    if not hasattr(vllm_bi, '_batch_invariant_LIB') or vllm_bi._batch_invariant_LIB is None:
        raise RuntimeError(
            "vLLM's batch_invariant mode is not initialized. "
            "Call init_batch_invariance() first."
        )

    # Use vLLM's existing library - don't destroy it!
    _batch_invariant_backward_LIB = vllm_bi._batch_invariant_LIB

    # Just add the backward operations - everything else is already handled by vLLM
    _batch_invariant_backward_LIB.impl("aten::matmul_backward", matmul_backward_impl, "CUDA")
    _batch_invariant_backward_LIB.impl("aten::linear_backward", linear_backward_impl, "CUDA")

    _batch_invariant_backward_MODE = True


def enable_batch_invariant_backward_mode():
    """Legacy name for patch_batch_invariant_with_gradients()."""
    patch_batch_invariant_with_gradients()


def disable_batch_invariant_backward_mode():
    """Disable batch invariant backward mode."""
    global _batch_invariant_backward_MODE, _batch_invariant_backward_LIB

    if _batch_invariant_backward_LIB is not None:
        _batch_invariant_backward_LIB._destroy()

    _batch_invariant_backward_MODE = False
    _batch_invariant_backward_LIB = None


def is_batch_invariant_backward_mode_enabled():
    """Check if batch invariant backward mode is enabled."""
    return _batch_invariant_backward_MODE
