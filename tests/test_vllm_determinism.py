"""
Test vLLM determinism across multiple runs.

This test:
1. Loads model weights into vLLM
2. Runs inference to generate tokens
3. Destroys all context (engine, CUDA cache)
4. Repeats the process N times
5. Verifies that all runs produce identical tokens
"""

import os
import sys
import torch
from transformers import AutoConfig
from safetensors.torch import load_file, save_file
from huggingface_hub import snapshot_download
import shutil
import glob

# Add spirl directory to Python path to find modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'spirl'))

from vllm import LLM, SamplingParams
from vllm.model_executor.layers.batch_invariant import init_batch_invariance

# Import converters
from weights_vllm_compat import torchtitan_to_vllm_compat, vllm_compat_to_torchtitan
from weights.converter import torchtitan_to_vllm, vllm_to_torchtitan

init_batch_invariance()


def prepare_vllm_model_dir(model_path: str, titan_checkpoint_path: str, output_dir: str = "./converted/vllm_test"):
    """
    Prepare a directory with vLLM-compatible weights.

    Args:
        model_path: Path to HuggingFace model (for config/tokenizer)
        titan_checkpoint_path: Path to TorchTitan checkpoint
        output_dir: Directory to save vLLM weights

    Returns:
        vllm_model_dir: Path to directory with vLLM weights
    """
    vllm_model_dir = os.path.abspath(output_dir)
    os.makedirs(vllm_model_dir, exist_ok=True)

    # Copy config/tokenizer files from base model
    for file in ["config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "merges.txt", "vocab.json"]:
        src = os.path.join(model_path, file)
        if os.path.exists(src):
            shutil.copy2(src, vllm_model_dir)

    # Load TorchTitan weights and convert to vLLM format
    titan_state = load_file(titan_checkpoint_path)
    vllm_compat_state = torchtitan_to_vllm_compat(titan_state)
    titan_state_converted = vllm_compat_to_torchtitan(vllm_compat_state)
    vllm_state = torchtitan_to_vllm(titan_state_converted)

    # Check if original model has shards
    original_shard_files = sorted(glob.glob(os.path.join(model_path, "model-*.safetensors")))
    index_file = os.path.join(model_path, "model.safetensors.index.json")

    if len(original_shard_files) == 2 and os.path.exists(index_file):
        # Copy index file
        shutil.copy2(index_file, vllm_model_dir)

        # Load the index to see which weights go in which shard
        import json
        with open(index_file, 'r') as f:
            index_data = json.load(f)

        weight_map = index_data['weight_map']

        # Split weights according to the index
        shard1_weights = {}
        shard2_weights = {}

        for key, value in vllm_state.items():
            shard_file = weight_map.get(key, original_shard_files[0])
            if 'model-00001-of-00002' in shard_file:
                shard1_weights[key] = value
            else:
                shard2_weights[key] = value

        # Ensure weights stay in bfloat16
        shard1_weights = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                          for k, v in shard1_weights.items()}
        shard2_weights = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                          for k, v in shard2_weights.items()}

        # Save to the shard files
        shard1_path = os.path.join(vllm_model_dir, os.path.basename(original_shard_files[0]))
        shard2_path = os.path.join(vllm_model_dir, os.path.basename(original_shard_files[1]))
        save_file(shard1_weights, shard1_path)
        save_file(shard2_weights, shard2_path)
    else:
        # Save as single file
        vllm_state = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                      for k, v in vllm_state.items()}
        checkpoint_path = os.path.join(vllm_model_dir, "model.safetensors")
        save_file(vllm_state, checkpoint_path)

    return vllm_model_dir


def run_vllm_inference(vllm_model_dir: str, prompts: list[str], seed: int = 42,
                       max_tokens: int = 20, temperature: float = 1.0,
                       n_samples: int = 4) -> tuple[list[list[int]], list[list[float]]]:
    """
    Run vLLM inference and return generated tokens.

    Args:
        vllm_model_dir: Directory with vLLM model weights
        prompts: List of prompt strings
        seed: Random seed
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        n_samples: Number of samples per prompt

    Returns:
        all_token_ids: List of token ID lists for each sample
        all_token_logprobs: List of logprob lists for each sample
    """
    # Create vLLM engine
    llm = LLM(
        model=vllm_model_dir,
        trust_remote_code=True,
        max_model_len=2048,
        dtype="bfloat16",
        gpu_memory_utilization=0.3,
        seed=seed,
        enforce_eager=True,  # Disable CUDA graphs for determinism
    )

    # Sampling params
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=n_samples,
        logprobs=1,
        seed=seed,  # Also set seed in sampling params
    )

    # Generate
    outputs = llm.generate(prompts, sampling_params)

    # Extract tokens and logprobs
    all_token_ids = []
    all_token_logprobs = []

    for output in outputs:
        for sample in output.outputs:
            token_ids = sample.token_ids
            all_token_ids.append(token_ids)

            # Extract per-token log probs
            per_token_logprobs = [
                list(logprob_dict.values())[0].logprob
                for logprob_dict in sample.logprobs
            ]
            all_token_logprobs.append(per_token_logprobs)

    # Cleanup
    del llm
    torch.cuda.empty_cache()

    return all_token_ids, all_token_logprobs


def test_determinism(model_name: str = "Qwen/Qwen3-1.7B",
                     cache_dir: str = None,
                     output_dir: str = None,
                     num_runs: int = 3,
                     skip_download: bool = False):
    """
    Test vLLM determinism across multiple runs.

    Args:
        model_name: HuggingFace model name
        cache_dir: Directory to cache downloaded model
        output_dir: Directory for converted weights
        num_runs: Number of runs to test
        skip_download: If True, reuse existing files instead of downloading
    """
    # Set default paths relative to spirl directory
    spirl_dir = os.path.join(os.path.dirname(__file__), '..', 'spirl')
    if cache_dir is None:
        cache_dir = os.path.join(spirl_dir, "models")
    if output_dir is None:
        output_dir = os.path.join(spirl_dir, "converted")

    print("=" * 80)
    print(f"Testing vLLM Determinism - {num_runs} runs")
    print("=" * 80)

    # Check if we can skip download and reuse existing files
    titan_checkpoint_path = os.path.join(output_dir, "qwen3_torchtitan.safetensors")
    vllm_test_dir = os.path.join(output_dir, "vllm_test")

    if skip_download and os.path.exists(titan_checkpoint_path):
        print(f"\nReusing existing TorchTitan checkpoint: {titan_checkpoint_path}")

        # Find model path from cache
        cache_model_dir = os.path.join(cache_dir, "models--Qwen--Qwen3-1.7B")
        if os.path.exists(cache_model_dir):
            # Find the snapshot directory
            snapshots_dir = os.path.join(cache_model_dir, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshot_dirs = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                if snapshot_dirs:
                    model_path = os.path.join(snapshots_dir, snapshot_dirs[0])
                    print(f"  Found cached model: {model_path}")
                else:
                    raise FileNotFoundError("No snapshot found in cache")
            else:
                raise FileNotFoundError("No snapshots directory in cache")
        else:
            raise FileNotFoundError(f"Model not found in cache: {cache_model_dir}")
    else:
        # Download model
        print(f"\nDownloading {model_name}...")
        model_path = snapshot_download(
            model_name,
            cache_dir=cache_dir,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer.model"],
        )
        print(f"  Downloaded to: {model_path}")

        # Convert to TorchTitan format
        print("\nConverting to TorchTitan format...")
        titan_state = vllm_to_torchtitan(model_path)
        save_file(titan_state, titan_checkpoint_path)
        print(f"  Saved to: {titan_checkpoint_path}")

    # Prepare vLLM model directory
    print("\nPreparing vLLM model directory...")
    vllm_model_dir = prepare_vllm_model_dir(model_path, titan_checkpoint_path, output_dir=vllm_test_dir)
    print(f"  vLLM model dir: {vllm_model_dir}")

    # Test prompts
    prompts = [
        "The capital of France is",
        "What is 7 times 8?",
        "The first president of the United States was",
    ]

    print(f"\nRunning {num_runs} inference runs with determinism test...")
    print(f"Prompts: {len(prompts)}, Samples per prompt: 4")
    print(f"Total samples per run: {len(prompts) * 4}")

    # Run multiple times
    all_runs_tokens = []
    all_runs_logprobs = []

    for run_idx in range(num_runs):
        print(f"\n--- Run {run_idx + 1}/{num_runs} ---")

        token_ids, token_logprobs = run_vllm_inference(
            vllm_model_dir,
            prompts,
            seed=42,
            max_tokens=20,
            temperature=1.0,
            n_samples=4,
        )

        all_runs_tokens.append(token_ids)
        all_runs_logprobs.append(token_logprobs)

        # Show first sample
        print(f"  First sample tokens: {token_ids[0][:10]}...")
        print(f"  First sample logprobs: {[f'{lp:.6f}' for lp in token_logprobs[0][:5]]}...")

    # Verify determinism
    print("\n" + "=" * 80)
    print("DETERMINISM VERIFICATION")
    print("=" * 80)

    reference_tokens = all_runs_tokens[0]
    reference_logprobs = all_runs_logprobs[0]

    all_tokens_match = True
    all_logprobs_match = True

    for run_idx in range(1, num_runs):
        # Check tokens
        tokens_match = reference_tokens == all_runs_tokens[run_idx]

        if tokens_match:
            print(f"\n✓ Run {run_idx + 1} tokens MATCH reference (Run 1)")
        else:
            print(f"\n✗ Run {run_idx + 1} tokens DIFFER from reference (Run 1)")
            all_tokens_match = False

            # Show differences
            for sample_idx, (ref_toks, run_toks) in enumerate(zip(reference_tokens, all_runs_tokens[run_idx])):
                if ref_toks != run_toks:
                    print(f"  Sample {sample_idx}: tokens differ")
                    print(f"    Reference: {ref_toks[:10]}...")
                    print(f"    Run {run_idx + 1}:     {run_toks[:10]}...")
                    break

        # Check logprobs (approximate comparison due to floating point)
        logprobs_match = True
        max_delta = 0.0
        avg_delta = 0.0
        total_tokens = 0

        for ref_lps, run_lps in zip(reference_logprobs, all_runs_logprobs[run_idx]):
            for ref_lp, run_lp in zip(ref_lps, run_lps):
                delta = abs(ref_lp - run_lp)
                max_delta = max(max_delta, delta)
                avg_delta += delta
                total_tokens += 1

                if delta > 1e-6:  # Tolerance for floating point
                    logprobs_match = False

        avg_delta = avg_delta / total_tokens if total_tokens > 0 else 0.0

        if logprobs_match:
            print(f"✓ Run {run_idx + 1} logprobs MATCH reference (within tolerance)")
            print(f"  Max delta: {max_delta:.10e}, Avg delta: {avg_delta:.10e}")
        else:
            print(f"✗ Run {run_idx + 1} logprobs DIFFER from reference")
            print(f"  Max delta: {max_delta:.10e}, Avg delta: {avg_delta:.10e}")
            all_logprobs_match = False

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    if all_tokens_match:
        print("✓ TOKENS: All runs produced IDENTICAL tokens")
    else:
        print("✗ TOKENS: Runs produced DIFFERENT tokens")

    if all_logprobs_match:
        print("✓ LOGPROBS: All runs produced IDENTICAL logprobs (within tolerance)")
    else:
        print("✗ LOGPROBS: Runs produced DIFFERENT logprobs")

    if all_tokens_match and all_logprobs_match:
        print("\n🎉 SUCCESS: vLLM is deterministic across runs!")
        return True
    else:
        print("\n⚠️  FAILURE: vLLM is NOT deterministic across runs")
        return False


if __name__ == "__main__":
    success = test_determinism(num_runs=3, skip_download=True)
    exit(0 if success else 1)
