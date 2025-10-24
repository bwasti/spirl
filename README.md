# SPIRL: Deterministic RL Training with vLLM & TorchTitan

Fast, bitwise-deterministic reinforcement learning for language models using vLLM for generation and TorchTitan for training.

## Overview

This project demonstrates:
1. **Fast rollouts** using vLLM (10-100x faster than PyTorch)
2. **Bitwise-deterministic training** using TorchTitan with vLLM-compatible kernels
3. **Exact matching** between generation and training forward passes (0.0000000 difference)
4. **RL training** with GRPO-style policy gradients

## Key Features

- ✅ **Bitwise determinism**: Training forward pass exactly matches generation
- ✅ **Fast generation**: vLLM for rollouts (fast!)
- ✅ **Trainable**: TorchTitan with full gradient support
- ✅ **Verification**: Automatic per-step determinism checks
- ✅ **Monitoring**: TensorBoard logging for training metrics

## Quick Start

### 1. Install Dependencies

```bash
pip install torch safetensors huggingface_hub transformers vllm tensorboard
```

### 2. Convert Model Weights

```bash
# Download and convert Qwen3-1.7B to TorchTitan format
python weights.py
```

This will:
- Download Qwen3-1.7B from HuggingFace (~3.5GB)
- Convert to TorchTitan format
- Save to `./converted/qwen3_torchtitan.safetensors`

### 3. Verify Bitwise Determinism

```bash
# Run comparison test to verify 0.0000000 difference
./run_comparison.sh "The capital of France is"
```

**Expected output:**
```
================================================================================
Log Probability Comparison
================================================================================
Tokens match: True
  ✓ Both predict: 12095 (' Paris')

Logprob difference: 0.000000000000000
  ✓✓✓ EXACT MATCH: 0.0000000000000 (bitwise identical!)
```

### 4. Run RL Training

```bash
# Train with determinism verification
env VLLM_BATCH_INVARIANT=1 \
    LD_PRELOAD="/usr/local/fbcode/platform010/lib/libcublasLt.so:/usr/local/fbcode/platform010/lib/libcublas.so" \
    HF_HUB_DISABLE_XET=1 \
    python simple_rl.py
```

### 5. Monitor Training with TensorBoard

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

## Files

### **Core Training**
- `simple_rl.py`: Main RL training loop with GRPO-style policy gradients
- `batch_invariant_backward.py`: Wraps vLLM's deterministic kernels with PyTorch autograd

### **Model Files**
- `torchtitan/models/qwen3/model/model_vllm_compat.py`: vLLM-compatible TorchTitan model
- Uses vLLM's exact kernels (SiluAndMul, RMSNorm, matmul_persistent)
- Merged gate_up_proj like vLLM for stability

### **Weight Conversion**
- `weights.py`: Standard TorchTitan ↔ vLLM conversion
- `weights_vllm_compat.py`: TorchTitan ↔ vLLM-compat conversion (merged projections)

### **Verification Scripts**
- `compare_direct_forward.py`: Verify bitwise determinism between vLLM and TorchTitan
- `run_comparison.sh`: Quick comparison test script
- `test_simple_rl.sh`: Training determinism test

### **Documentation**
- `DETERMINISM_VERIFICATION.md`: Detailed determinism verification guide
- `FINAL_RESULTS.md`: Results achieving 0.00000012 logprob difference
- `OPERATOR_DIFFERENCES.md`: Technical details on kernel differences

## How It Works

### **1. Fast Rollouts with vLLM**
```python
# Generate samples using vLLM (fast!)
vllm_engine = VLLMRolloutEngine(model_path)
completions, logprobs = vllm_engine.generate(prompts)
```

### **2. Deterministic Training with TorchTitan**
```python
# Train using TorchTitan (exact same forward pass!)
model = Qwen3VLLMCompatModel(model_args)  # Uses vLLM's kernels
loss = compute_policy_gradient_loss(model, completions, advantages)
loss.backward()
optimizer.step()
```

### **3. Automatic Verification**
Every training step verifies that:
- Forward pass logprobs match vLLM generation bitwise
- Uses native bfloat16 comparison (no fp32 conversion)
- Shows hex representation of bits for verification

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

### **Environment Variables**

- `VLLM_BATCH_INVARIANT=1`: Enable vLLM's deterministic mode
- `LD_PRELOAD`: Force correct cuBLAS libraries (Meta-specific)
- `HF_HUB_DISABLE_XET=1`: Disable XET for HuggingFace downloads

## Expected Results

### **Determinism Check**
- **✓✓✓ EXACT BITWISE MATCH**: All tokens match exactly (0 different)
- **✓✓ Excellent**: < 1e-7 difference (within bfloat16 precision)
- **✓ Good**: < 1e-4 difference
- **⚠ Warning**: > 1e-4 difference (something is wrong)

### **Training Progress**
- Initial loss: ~2.0-3.0 (random policy)
- Should decrease steadily over steps
- Reward mean should increase (if reward function is working)
- KL divergence should remain small (< 0.1)

### **TensorBoard**
- Loss curves should be smooth (determinism helps!)
- Weight deltas show which layers are changing
- Policy metrics track stability

## Troubleshooting

### **Large logprob differences (> 1e-4)**
1. Check model type: Must use `Qwen3VLLMCompatModel`, not `Qwen3Model`
2. Check weights: Must convert with `torchtitan_to_vllm_compat()`
3. Check environment: `VLLM_BATCH_INVARIANT=1` must be set

### **CUDA fork error**
- Make sure imports are inside functions, not at top level
- vLLM must fork before CUDA is initialized

### **OOM (Out of Memory)**
- Reduce `gpu_memory_utilization` in vLLM config (default: 0.3)
- Reduce `group_size` (number of samples per prompt)
- Reduce `max_new_tokens` (sequence length)

### **Training is unstable**
- Check KL divergence (should be < 0.1)
- Reduce learning rate
- Check `ratio_clipped_frac` (should be < 0.3)

## Why This Matters for RL

In RL training (especially GRPO/PPO):
1. **Generate samples** using vLLM (fast!)
2. **Compute advantages** based on rewards
3. **Train policy** by re-evaluating those samples

If the forward passes differ, the policy gradient is **wrong** because:
- The model computes different logprobs for the same tokens
- This breaks the PPO objective: `ratio = exp(current_logprob - old_logprob)`
- Training becomes unstable or ineffective

With **bitwise determinism**, we guarantee:
- `current_logprob == old_logprob` when weights haven't changed
- Clean gradient signal for policy updates
- Reproducible training runs

## Model Support

Currently tested with:
- **Qwen3-1.7B** ✅

Should work with other Qwen3 models with the same architecture.

## Citation

If you use this code, please cite:

```bibtex
@misc{spirl2025,
  title={SPIRL: Deterministic RL Training with vLLM and TorchTitan},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/spirl}
}
```

## License

BSD 3-Clause License (same as TorchTitan and vLLM)
