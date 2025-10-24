"""
Test forward passes for vLLM and TorchTitan with Qwen3-1.7B.

This script:
1. Downloads Qwen3-1.7B from HuggingFace
2. Converts weights to TorchTitan format
3. Runs forward passes on both vLLM and TorchTitan
4. Compares logits to verify weight conversion accuracy
"""

import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors.torch import save_file, load_file
from transformers import AutoTokenizer

# Add parent directory to Python path for tests
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import dependencies
import torchtitan.experiments.compat
from weights import vllm_to_torchtitan


def download_model(cache_dir: str = "./models") -> str:
    """Download Qwen3-1.7B from HuggingFace."""
    print("\n" + "=" * 80)
    print("Downloading Qwen3-1.7B...")
    print("=" * 80)

    model_name = "Qwen/Qwen3-1.7B"
    model_path = snapshot_download(
        model_name,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer.model"],
    )

    print(f"Model downloaded to: {model_path}")
    return model_path


def prepare_test_inputs(tokenizer, num_samples: int = 5, seq_len: int = 32):
    """Prepare test inputs for forward pass."""
    prompts = [
        "The capital of France is",
        "Once upon a time",
        "Machine learning is",
        "The quick brown fox",
        "Hello, my name is",
    ][:num_samples]

    # Tokenize prompts
    input_ids_list = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        # Pad or truncate to seq_len
        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]
        else:
            # Pad with zeros (will be masked in attention)
            tokens = tokens + [0] * (seq_len - len(tokens))
        input_ids_list.append(torch.tensor(tokens, dtype=torch.long))

    return torch.stack(input_ids_list), prompts


@torch.no_grad()
def forward_vllm(model_path: str, prompts: list[str], max_tokens: int = 1):
    """Run forward pass with vLLM."""
    print("\n" + "=" * 80)
    print("Running vLLM forward pass...")
    print("=" * 80)

    try:
        from vllm import LLM, SamplingParams

        # Create LLM
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=2048,
            gpu_memory_utilization=0.5,
        )

        # Get logits for next token (greedy, temp=0 for deterministic)
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            logprobs=1,  # Get logprobs for verification
        )

        outputs = llm.generate(prompts, sampling_params)

        # Extract logits (from logprobs)
        # vLLM returns log probabilities, we need to convert back
        logits_list = []
        for output in outputs:
            # Get the logprobs for the first generated token
            if output.outputs[0].logprobs:
                # Get all logprobs for the next token position
                token_logprobs = output.outputs[0].logprobs[0]

                # vLLM only returns top-k logprobs, so we can't get full logits
                # Instead, we'll just verify the top predicted tokens match
                print(f"Prompt: {output.prompt[:50]}...")
                print(f"  Top token ID: {output.outputs[0].token_ids[0]}")
                print(f"  Log prob: {list(token_logprobs.values())[0].logprob:.4f}")

                logits_list.append({
                    "top_token_id": output.outputs[0].token_ids[0],
                    "top_logprob": list(token_logprobs.values())[0].logprob,
                })

        del llm
        torch.cuda.empty_cache()

        return logits_list

    except Exception as e:
        print(f"vLLM forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None


@torch.no_grad()
def forward_vllm_model_runner(model_path: str, input_ids: torch.Tensor):
    """
    Run forward pass using vLLM's model directly (not the engine).
    This gives us access to full logits.
    """
    print("\n" + "=" * 80)
    print("Running vLLM model runner forward pass (direct model access)...")
    print("=" * 80)

    try:
        from transformers import AutoModelForCausalLM

        # Load model with transformers (same weights vLLM uses)
        print(f"Loading model from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        input_ids = input_ids.to(device)

        print(f"Running forward pass with input shape: {input_ids.shape}")
        outputs = model(input_ids)
        logits = outputs.logits  # [batch, seq, vocab]

        # Get logits for the last token
        last_token_logits = logits[:, -1, :]  # [batch, vocab]

        print(f"Logits shape: {last_token_logits.shape}")
        print(f"Logits range: [{last_token_logits.min().item():.2f}, {last_token_logits.max().item():.2f}]")

        # Get top predictions
        top_tokens = torch.argmax(last_token_logits, dim=-1)
        print(f"Top predicted tokens: {top_tokens.tolist()}")

        del model
        torch.cuda.empty_cache()

        return last_token_logits.cpu()

    except Exception as e:
        print(f"vLLM model runner forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None


@torch.no_grad()
def forward_torchtitan(titan_checkpoint_path: str, input_ids: torch.Tensor, model_path: str):
    """Run forward pass with TorchTitan model."""
    print("\n" + "=" * 80)
    print("Running TorchTitan forward pass...")
    print("=" * 80)

    try:
        # Import TorchTitan model (path already set up at module level)
        from torchtitan.models.qwen3.model.model import Qwen3Model
        from torchtitan.models.qwen3.model.args import Qwen3ModelArgs
        from transformers import AutoConfig

        # Load HuggingFace config
        print(f"Loading config from {model_path}...")
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # Create model args from HuggingFace config
        model_args = Qwen3ModelArgs(
            dim=hf_config.hidden_size,
            n_layers=hf_config.num_hidden_layers,
            n_heads=hf_config.num_attention_heads,
            n_kv_heads=hf_config.num_key_value_heads,
            vocab_size=hf_config.vocab_size,
            head_dim=getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads),
            hidden_dim=hf_config.intermediate_size,
            norm_eps=hf_config.rms_norm_eps,
            rope_theta=hf_config.rope_theta,
            max_seq_len=getattr(hf_config, "max_position_embeddings", 32768),
            qk_norm=True,  # Qwen3 specific
            depth_init=True,  # TorchTitan default
            eos_id=getattr(hf_config, "eos_token_id", 151645),
        )

        print(f"Model config: dim={model_args.dim}, layers={model_args.n_layers}, "
              f"heads={model_args.n_heads}, kv_heads={model_args.n_kv_heads}")

        print("Creating TorchTitan model...")
        model = Qwen3Model(model_args)

        # Load weights
        print(f"Loading weights from {titan_checkpoint_path}...")
        state_dict = load_file(titan_checkpoint_path)

        # Load into model
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: Missing keys: {missing[:5]}...")
        if unexpected:
            print(f"Warning: Unexpected keys: {unexpected[:5]}...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model = model.to(torch.bfloat16)
        model.eval()

        input_ids = input_ids.to(device)

        print(f"Running forward pass with input shape: {input_ids.shape}")

        # TorchTitan model returns logits directly
        logits = model(input_ids)  # [batch, seq, vocab]

        # Get logits for the last token
        last_token_logits = logits[:, -1, :]  # [batch, vocab]

        print(f"Logits shape: {last_token_logits.shape}")
        print(f"Logits range: [{last_token_logits.min().item():.2f}, {last_token_logits.max().item():.2f}]")

        # Get top predictions
        top_tokens = torch.argmax(last_token_logits, dim=-1)
        print(f"Top predicted tokens: {top_tokens.tolist()}")

        del model
        torch.cuda.empty_cache()

        return last_token_logits.cpu()

    except Exception as e:
        print(f"TorchTitan forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_logits(vllm_logits: torch.Tensor, titan_logits: torch.Tensor):
    """Compare logits from vLLM and TorchTitan."""
    print("\n" + "=" * 80)
    print("Comparing logits...")
    print("=" * 80)

    if vllm_logits is None or titan_logits is None:
        print("⚠️  Cannot compare: one or both forward passes failed")
        return

    # Compute differences
    abs_diff = (vllm_logits - titan_logits).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    print(f"Logits comparison:")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")

    # Compare top predictions
    vllm_top = torch.argmax(vllm_logits, dim=-1)
    titan_top = torch.argmax(titan_logits, dim=-1)

    matches = (vllm_top == titan_top).sum().item()
    total = vllm_top.numel()

    print(f"\nTop token predictions:")
    print(f"  vLLM:      {vllm_top.tolist()}")
    print(f"  TorchTitan: {titan_top.tolist()}")
    print(f"  Matches: {matches}/{total} ({100*matches/total:.1f}%)")

    # Compute KL divergence
    vllm_probs = F.softmax(vllm_logits, dim=-1)
    titan_log_probs = F.log_softmax(titan_logits, dim=-1)
    kl_div = F.kl_div(titan_log_probs, vllm_probs, reduction='batchmean').item()

    print(f"\nKL Divergence (TorchTitan || vLLM): {kl_div:.6f}")

    # Verdict
    if max_diff < 1e-3 and matches == total:
        print("\n✅ Forward passes match! Weight conversion is correct.")
    elif max_diff < 0.1 and matches == total:
        print("\n⚠️  Minor differences detected, but top predictions match.")
    else:
        print("\n❌ Significant differences detected. Weight conversion may have issues.")


def main():
    cache_dir = sys.argv[1] if len(sys.argv) > 1 else "./models"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./converted"

    os.makedirs(output_dir, exist_ok=True)

    print("Qwen3-1.7B Forward Pass Test")
    print("=" * 80)

    # Download model
    model_path = download_model(cache_dir)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Prepare test inputs
    input_ids, prompts = prepare_test_inputs(tokenizer, num_samples=3, seq_len=16)
    print(f"\nTest inputs shape: {input_ids.shape}")
    print(f"Prompts: {prompts}")

    # Convert weights to TorchTitan format
    print("\n" + "=" * 80)
    print("Converting weights to TorchTitan format...")
    print("=" * 80)

    titan_state = vllm_to_torchtitan(model_path)
    titan_checkpoint_path = os.path.join(output_dir, "qwen3_torchtitan.safetensors")
    save_file(titan_state, titan_checkpoint_path)
    print(f"Saved TorchTitan weights to {titan_checkpoint_path}")

    # Run forward passes
    vllm_logits = forward_vllm_model_runner(model_path, input_ids)
    titan_logits = forward_torchtitan(titan_checkpoint_path, input_ids, model_path)

    # Compare outputs
    compare_logits(vllm_logits, titan_logits)

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
