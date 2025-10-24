"""
Test script to download Qwen3-1.7B from HuggingFace and convert weights.

This demonstrates:
1. Downloading Qwen3-1.7B using huggingface_hub
2. Converting from vLLM/HF format to TorchTitan format
3. Converting back from TorchTitan to vLLM format
4. Verifying the round-trip conversion
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
import torch

import sys
from pathlib import Path

# Add parent directory to Python path for tests
sys.path.insert(0, str(Path(__file__).parent.parent))

from weights import vllm_to_torchtitan, torchtitan_to_vllm


def download_qwen3_model(cache_dir: str = "./models") -> str:
    """
    Download Qwen3-1.7B from HuggingFace.

    Args:
        cache_dir: Directory to cache the model

    Returns:
        Path to the downloaded model
    """
    print("=" * 80)
    print("Downloading Qwen3-1.7B from HuggingFace...")
    print("=" * 80)

    model_name = "Qwen/Qwen3-1.7B"

    # Download the model (only safetensors and config files)
    model_path = snapshot_download(
        model_name,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json"],  # Only download weights and configs
    )

    print(f"\nModel downloaded to: {model_path}")
    return model_path


def test_conversion(model_path: str, output_dir: str = "./converted"):
    """
    Test the weight conversion pipeline.

    Args:
        model_path: Path to the downloaded vLLM/HF model
        output_dir: Directory to save converted weights
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("Step 1: Convert vLLM → TorchTitan")
    print("=" * 80)

    # Convert vLLM to TorchTitan
    titan_state = vllm_to_torchtitan(model_path)

    # Save TorchTitan format
    titan_path = os.path.join(output_dir, "qwen3_torchtitan.safetensors")
    print(f"Saving TorchTitan weights to {titan_path}...")
    save_file(titan_state, titan_path)
    print(f"Saved! File size: {os.path.getsize(titan_path) / 1e9:.2f} GB")

    print("\n" + "=" * 80)
    print("Step 2: Convert TorchTitan → vLLM (round-trip test)")
    print("=" * 80)

    # Convert back to vLLM format
    vllm_state_converted = torchtitan_to_vllm(titan_state)

    # Save converted vLLM format
    vllm_converted_path = os.path.join(output_dir, "qwen3_vllm_converted.safetensors")
    print(f"Saving converted vLLM weights to {vllm_converted_path}...")
    save_file(vllm_state_converted, vllm_converted_path)
    print(f"Saved! File size: {os.path.getsize(vllm_converted_path) / 1e9:.2f} GB")

    print("\n" + "=" * 80)
    print("Step 3: Verify round-trip conversion")
    print("=" * 80)

    # Load original vLLM weights for comparison
    from safetensors.torch import load_file
    original_files = sorted(Path(model_path).glob("*.safetensors"))
    original_files = [f for f in original_files if "index" not in f.name]

    if original_files:
        print(f"Loading original weights from {len(original_files)} files...")
        original_state = {}
        for f in original_files:
            original_state.update(load_file(str(f)))

        # Compare keys
        original_keys = set(
            k for k in original_state.keys() if "rotary_emb.inv_freq" not in k
        )
        converted_keys = set(vllm_state_converted.keys())

        missing_keys = original_keys - converted_keys
        extra_keys = converted_keys - original_keys

        print(f"\nOriginal model: {len(original_keys)} weights")
        print(f"Converted model: {len(converted_keys)} weights")

        if missing_keys:
            print(f"\n⚠️  Missing keys in converted model: {len(missing_keys)}")
            for key in sorted(list(missing_keys)[:5]):
                print(f"  - {key}")
            if len(missing_keys) > 5:
                print(f"  ... and {len(missing_keys) - 5} more")

        if extra_keys:
            print(f"\n⚠️  Extra keys in converted model: {len(extra_keys)}")
            for key in sorted(list(extra_keys)[:5]):
                print(f"  - {key}")
            if len(extra_keys) > 5:
                print(f"  ... and {len(extra_keys) - 5} more")

        # Compare tensor values for common keys
        common_keys = original_keys & converted_keys
        if common_keys:
            print(f"\nComparing {len(common_keys)} common weights...")
            max_diff = 0.0
            mismatched_keys = []

            for key in sorted(common_keys):
                original_tensor = original_state[key]
                converted_tensor = vllm_state_converted[key]

                if original_tensor.shape != converted_tensor.shape:
                    mismatched_keys.append(
                        f"{key}: shape {original_tensor.shape} vs {converted_tensor.shape}"
                    )
                    continue

                diff = (original_tensor - converted_tensor).abs().max().item()
                max_diff = max(max_diff, diff)

                if diff > 1e-6:
                    mismatched_keys.append(f"{key}: max diff {diff}")

            print(f"Maximum absolute difference: {max_diff}")

            if max_diff < 1e-6:
                print("✅ Round-trip conversion successful! All weights match.")
            else:
                print(
                    f"⚠️  Some weights have differences > 1e-6. Mismatched: {len(mismatched_keys)}"
                )
                for msg in mismatched_keys[:5]:
                    print(f"  - {msg}")
                if len(mismatched_keys) > 5:
                    print(f"  ... and {len(mismatched_keys) - 5} more")

    print("\n" + "=" * 80)
    print("Inspection: Sample TorchTitan keys")
    print("=" * 80)

    sample_keys = sorted(titan_state.keys())[:10]
    for key in sample_keys:
        tensor = titan_state[key]
        print(f"  {key}: {tuple(tensor.shape)} {tensor.dtype}")

    print(f"\nTotal TorchTitan weights: {len(titan_state)}")


if __name__ == "__main__":
    import sys

    # Allow custom cache directory
    cache_dir = sys.argv[1] if len(sys.argv) > 1 else "./models"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./converted"

    print("Qwen3-1.7B Weight Converter Test")
    print("=" * 80)

    # Download model
    model_path = download_qwen3_model(cache_dir)

    # Test conversion
    test_conversion(model_path, output_dir)

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  - {output_dir}/qwen3_torchtitan.safetensors")
    print(f"  - {output_dir}/qwen3_vllm_converted.safetensors")
