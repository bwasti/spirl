# Single-file Deterministic RL Training with vLLM & TorchTitan

This is a project demonstrating simple, readable RL.
It achieves bitwise-deterministic reinforcement learning for language models using vLLM for generation and TorchTitan for training.

The single file is [here](https://github.com/bwasti/spirl/blob/main/simple_rl.py).

## Overview

This project demonstrates:
1. **vLLM rollouts with TorchTitan training** converting and updating weights between them
2. **Bitwise-deterministic training** using TorchTitan with vLLM-compatible kernels
3. **Exact matching** between generation and training forward passes (0.0000000 difference)
4. **Simple RL training** with GRPO-style policy gradients

## Quick Start

### 1. Install Dependencies

Install vLLM nightly:
```bash
pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
```

Install TorchTitan from the custom fork:
```bash
git clone -b shim https://github.com/bwasti/torchtitan.git
cd torchtitan
pip install -e .
cd ..
```

Install other dependencies:
```bash
pip install torch safetensors huggingface_hub transformers tensorboard
```

### 2. Run the training script!

```bash
VLLM_BATCH_INVARIANT=1 VLLM_FLASH_ATTN_VERSION=3 python simple_rl.py
```

### 3. Monitor Training with TensorBoard

```bash
# In a separate terminal
tensorboard --logdir=./runs

# Then open: http://localhost:6006
```

## Training Output

During training, you'll see determinism checks at each step:

```
================================================================================
Forward Pass Determinism Check (Training vs Generation)
================================================================================
Token 1: ID=12095 [✓]
  vLLM gen (bf16):     -0.6266400814  [hex: bf20]
  Titan train (bf16):  -0.6266400814  [hex: bf20]
  Δ (fp32 math):       0.000000000000000

Token 2: ID=25445 [✓]
  vLLM gen (bf16):     -1.2345678901  [hex: bf9d]
  Titan train (bf16):  -1.2345678901  [hex: bf9d]
  Δ (fp32 math):       0.000000000000000

Token 3: ID=11234 [✓]
  vLLM gen (bf16):     -0.9876543210  [hex: bf7d]
  Titan train (bf16):  -0.9876543210  [hex: bf7d]
  Δ (fp32 math):       0.000000000000000

Bitwise Summary over 20 tokens:
  Bitwise identical (bf16): True
  Different tokens: 0 / 20
  ✓✓✓ EXACT BITWISE MATCH!
================================================================================

Step 1/100:
  Loss: 2.3456
  Reward Mean: 0.15
  PG Loss: 1.234
  KL Div: 0.012
  Entropy: 4.567
```

**Interpretation:**
- **[✓]**: Token logprobs match bitwise between generation and training
- **[hex: ...]**: Raw bfloat16 bit representation for verification
- **Bitwise identical**: Uses native bf16 comparison (no fp32 conversion)
- **✓✓✓ EXACT BITWISE MATCH**: Training uses exact same forward pass as generation

## TensorBoard Metrics

The training logs the following metrics to TensorBoard:

### **Loss Metrics**
- `loss`: Total training loss (policy gradient + KL + entropy)
- `pg_loss`: Policy gradient loss
- `kl_div`: KL divergence between current and reference policy
- `entropy`: Policy entropy (higher = more exploration)

### **Reward Metrics**
- `reward_mean`: Average reward across samples
- `reward_std`: Reward standard deviation

### **Policy Metrics**
- `ratio_mean`: Mean importance sampling ratio (exp(logprob_new - logprob_old))
- `ratio_clipped_frac`: Fraction of ratios clipped by PPO (high = policy changing too fast)

### **Weight Deltas**
- `weight_delta/{layer}/magnitude`: L2 norm of weight changes per layer
- `weight_delta/{layer}/relative_change`: Relative weight change (normalized by weight magnitude)

### **Determinism Verification**
- Per-step determinism checks show exact bitwise match between generation and training


## How It Works

### **1. Rollouts with vLLM**
```python
# Generate samples using vLLM (fast!)
vllm_engine = VLLMRolloutEngine(model_path)
completions, logprobs = vllm_engine.generate(prompts)
```

### **2. Training with TorchTitan**
```python
# Train using TorchTitan (exact same forward pass!)
model = Qwen3VLLMCompatModel(model_args)  # Uses vLLM's kernels
loss = compute_policy_gradient_loss(model, completions, advantages)
loss.backward()
optimizer.step()
```

## Technical Details

### **Bitwise Determinism Achieved By:**

1. **vLLM's exact kernels**:
   - `SiluAndMul`: Custom CUDA kernel (`torch.ops._C.silu_and_mul`)
   - `RMSNorm`: Custom Triton kernel
   - `matmul_persistent`: Deterministic matrix multiplication
   - Flash Attention with `num_splits=1`

2. **Batch-invariant mode**:
   - `VLLM_BATCH_INVARIANT=1`: Enables deterministic vLLM operations
   - `batch_invariant_backward.py`: Adds gradient support

3. **Merged projections**:
   - vLLM-compat uses `gate_up_proj = [w1; w3]` (merged)
   - More numerically stable than separate w1/w3
   - Exactly matches vLLM's architecture

### **Weight Format**

**Standard TorchTitan:**
```
layers.0.feed_forward.w1.weight  [hidden_dim, dim]
layers.0.feed_forward.w2.weight  [dim, hidden_dim]
layers.0.feed_forward.w3.weight  [hidden_dim, dim]
```

**vLLM-Compat:**
```
layers.0.feed_forward.gate_up_proj.weight  [hidden_dim * 2, dim]  # [w1; w3] merged
layers.0.feed_forward.down_proj.weight     [dim, hidden_dim]
```

## Model Support

Currently tested with:
- **Qwen3-1.7B** ✅

Should work with other Qwen3 models with the same architecture.

## License

BSD 3-Clause License (same as TorchTitan and vLLM)
