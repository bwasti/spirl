"""
Batch-invariant operations with backward pass support.

This module wraps vLLM's deterministic Triton kernels with PyTorch autograd support,
enabling both deterministic forward passes AND gradient computation for RL training.

Key insight: Only the forward pass needs to be deterministic for reproducible rollouts.
The backward pass can use standard (faster) PyTorch operations since gradient computation
doesn't need to be deterministic.

Architecture:
- Forward: Uses vLLM's batch_invariant Triton kernels (deterministic)
- Backward: Uses standard PyTorch operations (fast, non-deterministic but correct)

Supported operations:
- mm, addmm, matmul, bmm: Matrix multiplication variants
- linear: Linear layers (most important for transformer models)
- log_softmax, softmax: Activation functions
- mean: Reduction operations

Usage:
    from batch_invariant_backward import enable_batch_invariant_backward_mode
    from vllm.model_executor.layers.batch_invariant import disable_batch_invariant_mode

    # Disable vLLM's mode (no backward) and enable ours (with backward)
    disable_batch_invariant_mode()
    enable_batch_invariant_backward_mode()

    # Now all operations are deterministic AND support gradients
    model = MyModel()
    output = model(input)  # deterministic forward
    loss = compute_loss(output)
    loss.backward()  # gradients work!
"""

import torch
import torch.nn.functional as F
from vllm.triton_utils import tl, triton


# ============================================================================
# Matrix Multiplication Backward
# ============================================================================

@triton.jit
def matmul_kernel_persistent_backward_a(
    grad_output_ptr,
    b_ptr,
    grad_a_ptr,
    M, N, K,
    stride_gom, stride_gon,
    stride_bk, stride_bn,
    stride_gam, stride_gak,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """
    Backward pass for matrix multiplication: compute grad_a.
    grad_a = grad_output @ b.T
    """
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_k

    for tile_id in range(start_pid, num_tiles, NUM_SMS):
        pid_m = tile_id // num_pid_k
        pid_k = tile_id % num_pid_k

        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        offs_bn = tl.arange(0, BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

        for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
            offs_n = n * BLOCK_SIZE_N + offs_bn

            # Load grad_output[m, n]
            go_ptrs = grad_output_ptr + (offs_am[:, None] * stride_gom + offs_n[None, :] * stride_gon)
            go_mask = (offs_am[:, None] < M) & (offs_n[None, :] < N)
            grad_out = tl.load(go_ptrs, mask=go_mask, other=0.0)

            # Load b[k, n] (will transpose to b.T[n, k])
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
            b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
            b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)

            # grad_a = grad_output @ b.T
            # We need: grad_out[m, n] @ b.T[n, k] = grad_out[m, n] @ b[k, n].T
            accumulator += tl.dot(grad_out, b_vals.trans())

        # Store grad_a
        ga_ptrs = grad_a_ptr + (offs_am[:, None] * stride_gam + offs_k[None, :] * stride_gak)
        ga_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
        grad_a = accumulator.to(grad_a_ptr.dtype.element_ty)
        tl.store(ga_ptrs, grad_a, mask=ga_mask)


@triton.jit
def matmul_kernel_persistent_backward_b(
    a_ptr,
    grad_output_ptr,
    grad_b_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_gom, stride_gon,
    stride_gbk, stride_gbn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """
    Backward pass for matrix multiplication: compute grad_b.
    grad_b = a.T @ grad_output
    """
    start_pid = tl.program_id(axis=0)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_k * num_pid_n

    for tile_id in range(start_pid, num_tiles, NUM_SMS):
        pid_k = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_am = tl.arange(0, BLOCK_SIZE_M)

        accumulator = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)

        for m in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
            offs_m = m * BLOCK_SIZE_M + offs_am

            # Load a[m, k]
            a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)

            # Load grad_output[m, n]
            go_ptrs = grad_output_ptr + (offs_m[:, None] * stride_gom + offs_bn[None, :] * stride_gon)
            go_mask = (offs_m[:, None] < M) & (offs_bn[None, :] < N)
            grad_out = tl.load(go_ptrs, mask=go_mask, other=0.0)

            # grad_b = a.T @ grad_output
            accumulator += tl.dot(a_vals.trans(), grad_out)

        # Store grad_b
        gb_ptrs = grad_b_ptr + (offs_k[:, None] * stride_gbk + offs_bn[None, :] * stride_gbn)
        gb_mask = (offs_k[:, None] < K) & (offs_bn[None, :] < N)
        grad_b = accumulator.to(grad_b_ptr.dtype.element_ty)
        tl.store(gb_ptrs, grad_b, mask=gb_mask)


class MatmulPersistentFunction(torch.autograd.Function):
    """Autograd function for deterministic matrix multiplication with gradients."""

    @staticmethod
    def forward(ctx, a, b, bias=None):
        """Forward pass: c = a @ b (+ bias)"""
        from vllm.model_executor.layers.batch_invariant import matmul_persistent

        # Save tensors for backward
        ctx.save_for_backward(a, b, bias)

        # Use vLLM's deterministic forward
        return matmul_persistent(a, b, bias=bias)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: compute gradients for a, b, and bias.

        Note: Backward doesn't need to be deterministic, only forward does.
        """
        a, b, bias = ctx.saved_tensors
        grad_a = grad_b = grad_bias = None

        # grad_a = grad_output @ b.T
        if ctx.needs_input_grad[0]:
            grad_a = torch.mm(grad_output, b.t())

        # grad_b = a.T @ grad_output
        if ctx.needs_input_grad[1]:
            grad_b = torch.mm(a.t(), grad_output)

        # grad_bias = sum(grad_output, dim=0)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)

        return grad_a, grad_b, grad_bias


def mm_batch_invariant_backward(a, b):
    """Matrix multiply with backward support."""
    return MatmulPersistentFunction.apply(a, b, None)


def addmm_batch_invariant_backward(bias, a, b):
    """Matrix multiply with bias and backward support."""
    return MatmulPersistentFunction.apply(a, b, bias)


# ============================================================================
# Linear Layer Backward
# ============================================================================

class LinearBatchInvariantFunction(torch.autograd.Function):
    """Autograd function for linear layer with batch invariance."""

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        """Forward: output = input @ weight.T (+ bias)"""
        ctx.save_for_backward(input, weight, bias)

        # Handle 3D input tensors (batch, seq, hidden)
        input_shape = input.shape
        ctx.input_shape = input_shape

        if input.ndim == 3:
            # Reshape to 2D
            input_2d = input.reshape(-1, input.shape[-1])
            output = mm_batch_invariant_backward(input_2d, weight.t())
            output = output.reshape(*input_shape[:-1], output.shape[-1])
        else:
            output = mm_batch_invariant_backward(input, weight.t())

        if bias is not None:
            output = output + bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for linear layer.

        Note: Backward doesn't need to be deterministic, only forward does.
        """
        input, weight, bias = ctx.saved_tensors
        input_shape = ctx.input_shape
        grad_input = grad_weight = grad_bias = None

        # Reshape grad_output if needed
        if grad_output.ndim == 3:
            grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        else:
            grad_output_2d = grad_output

        # grad_input = grad_output @ weight
        if ctx.needs_input_grad[0]:
            if input.ndim == 3:
                input_2d = input.reshape(-1, input.shape[-1])
                grad_input_2d = torch.mm(grad_output_2d, weight)
                grad_input = grad_input_2d.reshape(input_shape)
            else:
                grad_input = torch.mm(grad_output_2d, weight)

        # grad_weight = grad_output.T @ input
        if ctx.needs_input_grad[1]:
            if input.ndim == 3:
                input_2d = input.reshape(-1, input.shape[-1])
                grad_weight = torch.mm(grad_output_2d.t(), input_2d)
            else:
                grad_weight = torch.mm(grad_output_2d.t(), input)

        # grad_bias = sum(grad_output, dim=[0, ..., -2])
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(dim=0)

        return grad_input, grad_weight, grad_bias


def linear_batch_invariant_backward(input, weight, bias=None):
    """Linear layer with batch invariance and backward support."""
    return LinearBatchInvariantFunction.apply(input, weight, bias)


# ============================================================================
# BMM (Batched Matrix Multiply) Backward
# ============================================================================

class BmmBatchInvariantFunction(torch.autograd.Function):
    """Autograd function for batched matrix multiplication."""

    @staticmethod
    def forward(ctx, a, b):
        """Forward: c = a @ b for each batch."""
        ctx.save_for_backward(a, b)

        # Process each batch separately
        results = []
        for i in range(a.shape[0]):
            results.append(MatmulPersistentFunction.apply(a[i], b[i], None))

        return torch.stack(results, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for batched matmul.

        Note: Backward doesn't need to be deterministic, only forward does.
        """
        a, b = ctx.saved_tensors
        grad_a = grad_b = None

        if ctx.needs_input_grad[0]:
            grad_a_list = []
            for i in range(a.shape[0]):
                grad_a_list.append(torch.mm(grad_output[i], b[i].t()))
            grad_a = torch.stack(grad_a_list, dim=0)

        if ctx.needs_input_grad[1]:
            grad_b_list = []
            for i in range(a.shape[0]):
                grad_b_list.append(torch.mm(a[i].t(), grad_output[i]))
            grad_b = torch.stack(grad_b_list, dim=0)

        return grad_a, grad_b


def bmm_batch_invariant_backward(a, b, out=None):
    """Batched matrix multiply with backward support."""
    result = BmmBatchInvariantFunction.apply(a, b)
    if out is not None:
        out.copy_(result)
        return out
    return result


# ============================================================================
# Matmul (General) Backward
# ============================================================================

def matmul_batch_invariant_backward(a, b, out=None):
    """General matmul with batch invariance and backward support."""
    if a.ndim == 2 and b.ndim == 2:
        result = MatmulPersistentFunction.apply(a, b, None)
        if out is not None:
            out.copy_(result)
            return out
        return result
    elif a.ndim == 3 and b.ndim == 2:
        # Handle F.linear case: (batch, seq, in_features) @ (out_features, in_features).T
        # Reshape 3D to 2D, do matmul, reshape back
        batch_size, seq_len, in_features = a.shape
        a_2d = a.reshape(-1, in_features)  # (batch*seq, in_features)
        c_2d = MatmulPersistentFunction.apply(a_2d, b.t(), None)
        result = c_2d.reshape(batch_size, seq_len, -1)
        if out is not None:
            out.copy_(result)
            return out
        return result
    elif a.ndim == 3 and b.ndim == 3:
        return bmm_batch_invariant_backward(a, b, out=out)
    elif a.ndim == 4 and b.ndim == 4:
        # Handle 4D attention tensors
        batch, heads, seq_a, dim_a = a.shape
        _, _, dim_b, seq_b = b.shape

        a_3d = a.reshape(batch * heads, seq_a, dim_a)
        b_3d = b.reshape(batch * heads, dim_b, seq_b)

        result_3d = bmm_batch_invariant_backward(a_3d, b_3d)
        result = result_3d.reshape(batch, heads, seq_a, seq_b)

        if out is not None:
            out.copy_(result)
            return out
        return result
    else:
        raise ValueError(
            f"matmul_batch_invariant_backward only supports 2D x 2D, 3D x 2D, 3D x 3D, and 4D x 4D, "
            f"got shapes {a.shape} and {b.shape}"
        )


# ============================================================================
# Softmax/LogSoftmax Backward
# ============================================================================

class LogSoftmaxBatchInvariantFunction(torch.autograd.Function):
    """Autograd function for log_softmax with batch invariance."""

    @staticmethod
    def forward(ctx, input, dim, _half_to_float):
        """Forward: log_softmax(input, dim)"""
        from vllm.model_executor.layers.batch_invariant import log_softmax

        assert not _half_to_float, "not implemented"
        output = log_softmax(input, dim=dim)
        ctx.save_for_backward(output)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward: d/dx log_softmax = grad_output - exp(log_softmax) * sum(grad_output)"""
        output, = ctx.saved_tensors
        dim = ctx.dim

        # grad = grad_output - exp(output) * sum(grad_output, dim)
        sum_grad = grad_output.sum(dim=dim, keepdim=True)
        grad_input = grad_output - torch.exp(output) * sum_grad

        return grad_input, None, None


def _log_softmax_batch_invariant_backward(input, dim, _half_to_float):
    """Log softmax with backward support."""
    return LogSoftmaxBatchInvariantFunction.apply(input, dim, _half_to_float)


class SoftmaxBatchInvariantFunction(torch.autograd.Function):
    """Autograd function for softmax with batch invariance."""

    @staticmethod
    def forward(ctx, input, dim, dtype=None):
        """Forward: softmax(input, dim)"""
        from vllm.model_executor.layers.batch_invariant import softmax_batch_invariant

        output = softmax_batch_invariant(input, dim, dtype)
        ctx.save_for_backward(output)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward: d/dx softmax = softmax * (grad_output - sum(softmax * grad_output))"""
        output, = ctx.saved_tensors
        dim = ctx.dim

        # grad = output * (grad_output - sum(output * grad_output, dim))
        grad_output_times_output = grad_output * output
        sum_grad = grad_output_times_output.sum(dim=dim, keepdim=True)
        grad_input = output * (grad_output - sum_grad)

        return grad_input, None, None


def softmax_batch_invariant_backward(input, dim, dtype=None):
    """Softmax with backward support."""
    return SoftmaxBatchInvariantFunction.apply(input, dim, dtype)


# ============================================================================
# Mean Backward
# ============================================================================

class MeanBatchInvariantFunction(torch.autograd.Function):
    """Autograd function for mean with batch invariance."""

    @staticmethod
    def forward(ctx, input, dim, keepdim=False, dtype=None):
        """Forward: mean(input, dim)"""
        from vllm.model_executor.layers.batch_invariant import mean_batch_invariant

        ctx.input_shape = input.shape
        ctx.dim = dim
        ctx.keepdim = keepdim

        output = mean_batch_invariant(input, dim, keepdim, dtype)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward: gradient is distributed equally across reduced dimension."""
        input_shape = ctx.input_shape
        dim = ctx.dim
        keepdim = ctx.keepdim

        # Compute the number of elements in the reduced dimension
        if len(dim) == 0:
            dim = [i for i in range(len(input_shape))]

        # Calculate total elements in reduced dimensions
        n_elements = 1
        for d in dim:
            n_elements *= input_shape[d % len(input_shape)]

        # Expand grad_output back to input shape
        if not keepdim:
            # Add back the reduced dimensions
            for d in sorted([d % len(input_shape) for d in dim]):
                grad_output = grad_output.unsqueeze(d)

        # Broadcast to input shape and divide by number of elements
        grad_input = grad_output.expand(input_shape) / n_elements

        return grad_input, None, None, None


def mean_batch_invariant_backward(input, dim, keepdim=False, dtype=None):
    """Mean with backward support."""
    return MeanBatchInvariantFunction.apply(input, dim, keepdim, dtype)


# ============================================================================
# Registration
# ============================================================================

_batch_invariant_backward_MODE = False
_batch_invariant_backward_LIB = None


def enable_batch_invariant_backward_mode():
    """Enable batch invariant mode with backward support.

    This mode provides deterministic forward passes using vLLM's Triton kernels,
    combined with standard PyTorch backward passes (which don't need to be deterministic).
    """
    global _batch_invariant_backward_MODE, _batch_invariant_backward_LIB

    if _batch_invariant_backward_MODE:
        return

    _batch_invariant_backward_MODE = True
    _batch_invariant_backward_LIB = torch.library.Library("aten", "IMPL")

    # Override CUDA implementations with our backward-enabled versions
    # We only override the lowest-level matrix operations (mm, addmm, bmm)
    # PyTorch's autograd will automatically use these for higher-level ops
    _batch_invariant_backward_LIB.impl("aten::mm", mm_batch_invariant_backward, "CUDA")
    _batch_invariant_backward_LIB.impl("aten::addmm", addmm_batch_invariant_backward, "CUDA")
    _batch_invariant_backward_LIB.impl("aten::bmm", bmm_batch_invariant_backward, "CUDA")

    # Don't override these - let PyTorch's autograd handle them:
    # - aten::matmul (composed of mm/bmm under the hood)
    # - aten::linear (uses addmm under the hood)
    # - aten::_log_softmax, aten::softmax (computed correctly from primitives)
    # - aten::mean.dim (uses our deterministic mm for any matmul operations)


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
