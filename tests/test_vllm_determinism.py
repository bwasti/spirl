"""
Test vLLM determinism across multiple runs.

Tests two aspects of determinism:
1. Cross-run determinism: Same prompts produce identical outputs across multiple runs
2. Batch-size independence: Same prompt produces identical output regardless of batch size

Minimal configuration required:
- VLLM_BATCH_INVARIANT=1 (environment variable)
- seed=42 in LLM() and SamplingParams()
"""

import hashlib
import os
import torch
from huggingface_hub import snapshot_download

from vllm import LLM, SamplingParams
from vllm.model_executor.layers.batch_invariant import init_batch_invariance

init_batch_invariance()


# Expected fingerprint for batch-size independence test
# Update this if you change the model or prompts
EXPECTED_FIRST_PROMPT_FINGERPRINT = "47dbff62d55b557f378d29582b81d871"


def compute_fingerprint(completions: list[str]) -> str:
    """Compute MD5 hash of completions for determinism verification."""
    return hashlib.md5("|||".join(completions).encode()).hexdigest()


def run_vllm_inference(
    model_path: str,
    prompts: list[str],
    seed: int = 42,
    max_tokens: int = 20,
    temperature: float = 1.0,
    n_samples: int = 4,
) -> tuple[list[list[int]], list[list[float]]]:
    """
    Run vLLM inference and return generated tokens.

    Args:
        model_path: Path to HuggingFace model
        prompts: List of prompt strings
        seed: Random seed
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        n_samples: Number of samples per prompt

    Returns:
        all_token_ids: List of token ID lists for each sample
        all_token_logprobs: List of logprob lists for each sample
    """
    # Create fresh vLLM engine
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=2048,
        dtype="bfloat16",
        gpu_memory_utilization=0.3,
        seed=seed,
        enable_prefix_caching=True,
    )

    # Sampling params with seed
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=n_samples,
        logprobs=1,
        seed=seed,
    )

    # Generate
    outputs = llm.generate(prompts, sampling_params)

    # Extract tokens and logprobs
    all_token_ids = []
    all_token_logprobs = []

    for output in outputs:
        for sample in output.outputs:
            all_token_ids.append(sample.token_ids)
            per_token_logprobs = [
                list(logprob_dict.values())[0].logprob
                for logprob_dict in sample.logprobs
            ]
            all_token_logprobs.append(per_token_logprobs)

    # Cleanup
    del llm
    torch.cuda.empty_cache()

    return all_token_ids, all_token_logprobs


def run_vllm_inference_with_text(
    model_path: str,
    prompts: list[str],
    seed: int = 42,
    max_tokens: int = 20,
    temperature: float = 1.0,
    n_samples: int = 4,
) -> list[str]:
    """
    Run vLLM inference and return generated text completions.

    Args:
        model_path: Path to HuggingFace model
        prompts: List of prompt strings
        seed: Random seed
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        n_samples: Number of samples per prompt

    Returns:
        completions: List of text completions (n_samples per prompt)
    """
    # Create fresh vLLM engine
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=2048,
        dtype="bfloat16",
        gpu_memory_utilization=0.3,
        seed=seed,
        enable_prefix_caching=True,
    )

    # Sampling params with seed
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=n_samples,
        seed=seed,
    )

    # Generate
    outputs = llm.generate(prompts, sampling_params)

    # Extract text completions
    completions = []
    for output in outputs:
        for sample in output.outputs:
            completions.append(sample.text)

    # Cleanup
    del llm
    torch.cuda.empty_cache()

    return completions


def test_batch_size_independence(
    model_name: str = "Qwen/Qwen3-1.7B",
    cache_dir: str = "./models",
):
    """
    Test that the same prompt produces identical output regardless of batch size.

    Tests inference with 1, 2, and 3 prompts, verifying the first prompt always
    produces the same output.

    Args:
        model_name: HuggingFace model name
        cache_dir: Directory to cache downloaded model

    Returns:
        True if batch-size independent, False otherwise
    """
    print("=" * 80)
    print("Testing Batch-Size Independence")
    print("=" * 80)

    # Download model from HuggingFace
    print(f"\nDownloading {model_name}...")
    model_path = snapshot_download(
        model_name,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer.model"],
    )
    print(f"Model path: {model_path}")

    # Test prompts
    prompts = [
        "The capital of France is",
        "What is 7 times 8?",
        "The first president of the United States was",
    ]

    print("\nRunning inference with varying batch sizes...")
    print(f"Testing first prompt: '{prompts[0]}'")

    # Store completions for first prompt from each batch size
    first_prompt_completions = {}

    # Test with 1, 2, and 3 prompts
    for batch_size in [1, 2, 3]:
        print(f"\n--- Batch size: {batch_size} ---")
        batch_prompts = prompts[:batch_size]

        # Run inference
        completions = run_vllm_inference_with_text(
            model_path,
            batch_prompts,
            seed=42,
            max_tokens=20,
            temperature=1.0,
            n_samples=4,
        )

        # Extract completions for first prompt (first 4 samples)
        first_prompt_samples = completions[:4]
        first_prompt_completions[batch_size] = first_prompt_samples

        # Compute fingerprint
        fingerprint = compute_fingerprint(first_prompt_samples)
        print(f"  Fingerprint: {fingerprint}")
        print(f"  Sample: {first_prompt_samples[0]}")

    # Verify all batch sizes produce identical outputs
    print("\n" + "=" * 80)
    print("BATCH-SIZE INDEPENDENCE VERIFICATION")
    print("=" * 80)

    reference_completions = first_prompt_completions[1]
    reference_fingerprint = compute_fingerprint(reference_completions)

    all_match = True
    for batch_size in [2, 3]:
        completions = first_prompt_completions[batch_size]
        fingerprint = compute_fingerprint(completions)

        if completions == reference_completions:
            print(f"\n✓ Batch size {batch_size}: Outputs MATCH batch size 1")
        else:
            print(f"\n✗ Batch size {batch_size}: Outputs DIFFER from batch size 1")
            all_match = False

            # Show differences
            for i, (ref, test) in enumerate(zip(reference_completions, completions)):
                if ref != test:
                    print(f"  Sample {i} differs:")
                    print(f"    Batch 1: {ref}")
                    print(f"    Batch {batch_size}: {test}")

    # Verify against expected fingerprint
    print("\n" + "=" * 80)
    print("FINGERPRINT VERIFICATION")
    print("=" * 80)

    print(f"\nExpected fingerprint: {EXPECTED_FIRST_PROMPT_FINGERPRINT}")
    print(f"Actual fingerprint:   {reference_fingerprint}")

    if reference_fingerprint == EXPECTED_FIRST_PROMPT_FINGERPRINT:
        print("✓ Fingerprint MATCHES expected value")
        fingerprint_match = True
    else:
        print("✗ Fingerprint DIFFERS from expected value")
        print("  (This is expected if you changed the model or prompts)")
        fingerprint_match = False

    # Final result
    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    if all_match and fingerprint_match:
        print("✓ SUCCESS: vLLM is batch-size independent and matches expected output!")
        return True
    elif all_match:
        print("⚠ PARTIAL SUCCESS: vLLM is batch-size independent but fingerprint differs")
        print("  Update EXPECTED_FIRST_PROMPT_FINGERPRINT if model/prompts changed")
        return True
    else:
        print("✗ FAILURE: vLLM outputs differ across batch sizes")
        return False


def test_determinism(
    model_name: str = "Qwen/Qwen3-1.7B",
    cache_dir: str = "./models",
    num_runs: int = 3,
):
    """
    Test vLLM determinism across multiple runs.

    Args:
        model_name: HuggingFace model name
        cache_dir: Directory to cache downloaded model
        num_runs: Number of runs to test
    """
    print("=" * 80)
    print(f"Testing vLLM Cross-Run Determinism - {num_runs} runs")
    print("=" * 80)

    # Download model from HuggingFace
    print(f"\nDownloading {model_name}...")
    model_path = snapshot_download(
        model_name,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer.model"],
    )
    print(f"Model path: {model_path}")

    # Test prompts
    prompts = [
        "The capital of France is",
        "What is 7 times 8?",
        "The first president of the United States was",
    ]

    print(f"\nRunning {num_runs} inference runs...")
    print(f"Prompts: {len(prompts)}, Samples per prompt: 4")

    # Run multiple times
    all_runs_tokens = []
    all_runs_logprobs = []

    for run_idx in range(num_runs):
        print(f"\n--- Run {run_idx + 1}/{num_runs} ---")

        token_ids, token_logprobs = run_vllm_inference(
            model_path,
            prompts,
            seed=42,
            max_tokens=20,
            temperature=1.0,
            n_samples=4,
        )

        all_runs_tokens.append(token_ids)
        all_runs_logprobs.append(token_logprobs)

        print(f"  First sample tokens: {token_ids[0][:10]}...")
        print(f"  First sample logprobs: {[f'{lp:.6f}' for lp in token_logprobs[0][:5]]}")

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
            print(f"\n✓ Run {run_idx + 1}: Tokens MATCH")
        else:
            print(f"\n✗ Run {run_idx + 1}: Tokens DIFFER")
            all_tokens_match = False

        # Check logprobs
        logprobs_match = True
        max_delta = 0.0
        total_delta = 0.0
        total_tokens = 0

        for ref_lps, run_lps in zip(reference_logprobs, all_runs_logprobs[run_idx]):
            for ref_lp, run_lp in zip(ref_lps, run_lps):
                delta = abs(ref_lp - run_lp)
                max_delta = max(max_delta, delta)
                total_delta += delta
                total_tokens += 1
                if delta > 1e-6:
                    logprobs_match = False

        avg_delta = total_delta / total_tokens if total_tokens > 0 else 0.0

        if logprobs_match:
            print(f"✓ Run {run_idx + 1}: Logprobs MATCH (max Δ: {max_delta:.2e})")
        else:
            print(f"✗ Run {run_idx + 1}: Logprobs DIFFER (max Δ: {max_delta:.2e})")
            all_logprobs_match = False

    # Final result
    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    if all_tokens_match and all_logprobs_match:
        print("✓ SUCCESS: vLLM is deterministic across runs!")
        return True
    else:
        print("✗ FAILURE: vLLM is NOT deterministic")
        if not all_tokens_match:
            print("  - Tokens differ across runs")
        if not all_logprobs_match:
            print("  - Logprobs differ across runs")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("VLLM DETERMINISM TEST SUITE")
    print("=" * 80)
    print("\nThis suite tests two aspects of determinism:")
    print("1. Batch-size independence: Same prompt → same output regardless of batch size")
    print("2. Cross-run determinism: Same prompts → identical outputs across multiple runs")
    print("\n")

    # Test 1: Batch-size independence
    batch_independence_success = test_batch_size_independence()

    print("\n\n")

    # Test 2: Cross-run determinism
    cross_run_success = test_determinism(num_runs=3)

    # Final summary
    print("\n\n" + "=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)
    print(f"Batch-size independence: {'✓ PASS' if batch_independence_success else '✗ FAIL'}")
    print(f"Cross-run determinism:   {'✓ PASS' if cross_run_success else '✗ FAIL'}")

    if batch_independence_success and cross_run_success:
        print("\n✓ ALL TESTS PASSED")
        exit(0)
    else:
        print("\n✗ SOME TESTS FAILED")
        exit(1)
